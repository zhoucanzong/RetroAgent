"""RetroPlanner: the central reasoning agent, inheriting mini-swe-agent's DefaultAgent.

This is the ONLY component that makes decisions. It:
1. Reads the blackboard state
2. Decides which tools to call and in what order
3. Interprets tool results and adjusts search strategy
4. Determines when a complete route has been found

Dead-Loop Monitor (Phase 0): detects cycling, semantic repeats, stagnation, and
suggests early exit — all heuristic checks that don't add LLM calls.

Think Tool support (Phase 1): gives the LLM explicit reasoning space via a virtual
"think" tool, formatted as <thinking> blocks in the conversation.
"""

import json
import logging
import time
from collections import deque
from pathlib import Path

import yaml
from jinja2 import StrictUndefined, Template

from retroagent.agents.config import PlannerConfig
from retroagent.blackboard import SharedBlackboard


class RetroPlanner:
    """
    Reimplements the DefaultAgent control flow from mini-swe-agent but adapted
    for retrosynthetic planning with structured chemical tools.

    Control flow:
        run(target_smiles):
            initialize blackboard
            render system + user messages
            while not done:
                step():
                    query() -> LLM proposes action(s)
                    execute_actions() -> run tools, update blackboard
                    add observation to messages
            return result
    """

    def __init__(self, model, env, blackboard: SharedBlackboard | None = None,
                 *, config_class: type = PlannerConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("retroagent")
        self.cost = 0.0
        self.n_calls = 0
        self.n_consecutive_format_errors = 0
        self._start_time = time.time()
        self.blackboard = blackboard or SharedBlackboard()
        # Inner Loop state
        self._action_history: list[tuple[str, str]] = []
        self._low_score_count = 0
        self._last_best_score: float | None = None
        self._tool_schemas: dict[str, dict] = {}
        self._load_tool_schemas()
        # Multi-perspective review state
        self._review_count = 0
        self._last_reviewed_round = -1
        # Dead-Loop Monitor state
        self._tool_sequence: deque = deque(maxlen=self.config.cycling_window)
        self._semantic_call_counts: dict[tuple[str, str], int] = {}
        self._last_progress_round: int = 0
        self._last_stagnation_warning_round: int = -10  # avoid warning on first rounds
        self._evaluate_call_count: int = 0
        self._last_eval_route_ids: set = set()
        # Complexity auto-tune
        self._complexity_level: str = "moderate"
        # Design Auditor state
        self._design_audit_count = 0
        self._last_design_audit_round: int = -3

    def get_template_vars(self, **kwargs) -> dict:
        """Collect all template variables for Jinja2 rendering."""
        result = {
            "n_model_calls": self.n_calls,
            "model_cost": self.cost,
            "elapsed_seconds": int(time.time() - self._start_time),
            "max_search_depth": self.config.max_search_depth,
            "search_strategy": self.config.search_strategy,
            "tools_spec": self.env.get_tools_spec(),
        }
        # Merge config dump
        result |= self.config.model_dump()
        # Merge blackboard state
        result |= self.blackboard.to_template_vars()
        # Merge extra vars
        result |= self.extra_template_vars
        result |= kwargs
        return result

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(str(messages)[:500])
        self.messages.extend(messages)
        return list(messages)

    def run(self, task: str = "", mode: str = "retrosynthesis", **kwargs) -> dict:
        """Main entry point. Supports two modes:

        - 'retrosynthesis': given a target SMILES, plan a synthetic route.
        - 'design': given natural-language constraints, design a molecule/ligand.
        """
        self.extra_template_vars |= {"task": task, "mode": mode, **kwargs}
        self.messages = []

        # Initialize blackboard if target provided
        if task:
            self.blackboard.initialize(task, mode=mode)

        # Auto-tune parameters based on molecular complexity
        if self.config.enable_auto_tune and mode == "retrosynthesis":
            self._auto_tune_from_complexity()

        # Add system and user messages
        system_content = self._render_template(self.config.system_template)
        instance_content = self._render_template(self.config.instance_template)
        self.add_messages(
            self._format_message("system", system_content),
            self._format_message("user", instance_content),
        )

        while True:
            try:
                self.step()
                self.n_consecutive_format_errors = 0
            except _InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except _FormatError as e:
                self.n_consecutive_format_errors += 1
                if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *e.messages,
                        self._format_message("user", "RepeatedFormatError — exiting",
                                             extra={"exit_status": "RepeatedFormatError", "submission": ""}),
                    )
                    break
                else:
                    self.add_messages(*e.messages)
            except Exception as e:
                self.add_messages(
                    self._format_message("user", f"Fatal error: {e}",
                                         extra={"exit_status": type(e).__name__, "submission": ""})
                )
                break
            finally:
                self._save_trajectory()
            # Check for exit signal: look for COMPLETE_TASK in recent assistant messages only
            if self.messages:
                recent = [m for m in self.messages[-3:] if m.get("role") == "assistant"]
                if any("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in m.get("content", "") for m in recent):
                    break
                # Or check exit_status in extra dict on any recent message
                if any(m.get("extra", {}).get("exit_status") for m in self.messages[-3:]):
                    break

        # Find and return the submission
        for m in reversed(self.messages):
            extra = m.get("extra", {})
            if extra.get("exit_status"):
                return extra
        return {"exit_status": "Unknown", "submission": ""}

    def step(self) -> list[dict]:
        """One iteration: query LLM -> execute tools -> add observations."""
        msg = self.query()
        result = self.execute_actions(msg)

        # Phase 0F: Design Auditor — review catalyst/ligand designs for chemical errors
        if self.blackboard.mode == "design":
            audit_msg = self._audit_design()
            if audit_msg:
                result.extend(self.add_messages(self._format_message("user", audit_msg)))

        return result

    def query(self) -> dict:
        """Query the model and return the assistant message."""
        # Check limits
        if 0 < self.config.step_limit <= self.n_calls:
            raise _LimitsExceeded([self._format_message(
                "exit", "LimitsExceeded", extra={"exit_status": "LimitsExceeded", "submission": ""})])
        if 0 < self.config.cost_limit <= self.cost:
            raise _LimitsExceeded([self._format_message(
                "exit", "CostLimitExceeded", extra={"exit_status": "LimitsExceeded", "submission": ""})])
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            raise _TimeExceeded([self._format_message(
                "exit", "TimeExceeded", extra={"exit_status": "TimeExceeded", "submission": ""})])

        self.n_calls += 1
        message = self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def _review_route(self) -> list[dict] | None:
        """Multi-perspective review: switch the LLM to a 'reviewer' role to critique the
        current best route(s) before submission.

        Only trigger when:
          - The agent has found at least one route (scores present or routes on blackboard)
          - It's been at least 2 rounds since the last review
          - We haven't already reviewed too many times

        The reviewer prompt makes the model adopt a critical stance, identifying:
          1. Missing chirality / stereochemistry considerations
          2. Competing reactive sites (chemoselectivity)
          3. Unrealistic or unvalidated reaction mechanisms
          4. Missing literature precedent
          5. Over-reliance on non-matching templates
        """
        if not self.config.enable_reflection:
            return None

        # Trigger condition: have routes or scores, and spaced reviews
        has_routes = (
            (self.blackboard.proposal_results and len(self.blackboard.proposal_results) > 0)
            or (self.blackboard.scores and len(self.blackboard.scores) > 0)
            or (self.blackboard.routes and len(self.blackboard.routes) > 0)
        )
        if not has_routes:
            return None

        if self._review_count >= 2:
            return None

        current_round = self.blackboard.iteration_count
        if current_round - self._last_reviewed_round < 3:
            return None

        self._last_reviewed_round = current_round
        self._review_count += 1

        # Build the reviewer prompt from current blackboard state
        route_summary = self._summarize_routes_for_review()

        reviewer_prompt = (
            "### Role Switch: Peer Reviewer Mode ###\n\n"
            "You are now acting as a **critical peer reviewer** for the retrosynthetic analysis "
            "that was just proposed. Your ONLY task is to identify flaws, omissions, and risks. "
            "Do NOT propose fixes — just audit.\n\n"
            f"#### Current Proposed Route(s):\n{route_summary}\n\n"
            "#### Review Checklist (answer each explicitly):\n"
            "1. **Stereochemistry** — Does the target have chiral centers? If so, does the proposed "
            "route address enantioselectivity? Which enantiomer is needed?\n"
            "2. **Chemoselectivity** — Are there competing functional groups that could react "
            "under the proposed conditions?\n"
            "3. **Mechanism plausibility** — Are the bond disconnections chemically sound? "
            "Could the reaction actually proceed as written?\n"
            "4. **Template quality** — Are the templates used substructure-matched? "
            "If templates had `matching=false`, flag this as a risk.\n"
            "5. **Literature gap** — Is this route known from precedents, or is it exploratory?\n"
            "6. **Stock realism** — Are all precursors commercially available?\n\n"
            "#### Output format:\n"
            "For each issue found, write one line starting with `ISSUE:`. "
            "Start each issue with a **severity tag**: [CRITICAL], [WARNING], or [INFO].\n"
            "End with a single line: `VERDICT: ` followed by one of: "
            "PROCEED / NEEDS_FIX / ABANDON.\n\n"
            "If NO issues are found, write `VERDICT: PROCEED — No issues detected.`"
        )

        try:
            review_message = self.model.query(
                self.messages
                + [self._format_message("user", reviewer_prompt)]
            )
            review_content = review_message.get("content", "")
            self.cost += review_message.get("extra", {}).get("cost", 0.0)
            self.n_calls += 1

            # Inject the reviewer's output as an observation so the planner sees it
            formatted = (
                "<role:reviewer>\n"
                f"{review_content}\n"
                "</role:reviewer>\n\n"
                "You are now back in **Planner mode**. Consider the reviewer's feedback above. "
                "If the reviewer found CRITICAL or WARNING issues, you MUST address them "
                "before submitting. Adjust your route or backtrack if needed."
            )
            return self.add_messages(self._format_message("user", formatted))
        except Exception as e:
            self.logger.warning(f"Review step failed: {e}")
            return None

    def _summarize_routes_for_review(self) -> str:
        """Extract a concise summary of current routes from the blackboard."""
        parts = []
        bb = self.blackboard

        parts.append(f"Target: {bb.target_smiles}\n")

        # Scores
        if bb.scores:
            for rid, s in list(bb.scores.items())[:3]:
                total = s["total"] if isinstance(s, dict) else s
                feasibility = s.get("feasibility", "?") if isinstance(s, dict) else total
                stock = s.get("stock_availability", "?") if isinstance(s, dict) else "?"
                parts.append(f"Route {rid}: total={total:.4f} feasibility={feasibility} stock={stock}")
            parts.append("")

        # Proposal results: show precursor SMILES
        if bb.proposal_results:
            for rxn in bb.proposal_results[:5]:
                precursors = rxn.get("precursors", [])
                classification = rxn.get("classification", "?")
                tid = rxn.get("template_index", "?")
                parts.append(
                    f"Template {tid} [{classification}]: "
                    f"{' + '.join(precursors) if precursors else 'no precursors'}"
                )
            parts.append("")

        # Disconnections: matching vs non-matching
        if bb.disconnection_results:
            for b in bb.disconnection_results[:3]:
                parts.append(
                    f"Disconnection rank={b.get('rank')} template#{b.get('template_index')} "
                    f"score={b.get('score')} matching={b.get('matching')} "
                    f"class={b.get('classification')}"
                )

        # Stock hits
        if bb.stock_hits:
            parts.append(f"\nStock hits ({len(bb.stock_hits)}): {', '.join(list(bb.stock_hits)[:6])}")

        return "\n".join(parts) if parts else "(no routes yet)"

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute tool calls from the LLM response, update blackboard, return observations."""
        actions = message.get("extra", {}).get("actions", [])
        if not actions:
            return []

        outputs = []
        for action in actions:
            result = self.env.execute(action)
            outputs.append(result)
            # Update blackboard if JSON returned
            if result["returncode"] == 0:
                try:
                    parsed = json.loads(result["output"])
                    tool_name = action.get("tool", "bash")
                    self.blackboard.update(tool_name, parsed)
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass  # bash output is not always JSON

        return self.add_messages(*self._format_observations(message, outputs))

    def _format_message(self, role: str, content: str, extra: dict | None = None) -> dict:
        return {"role": role, "content": content, "extra": extra or {}}

    def _format_observations(self, message: dict, outputs: list[dict]) -> list[dict]:
        """Format tool execution outputs as observation messages."""
        obs = []
        for i, out in enumerate(outputs):
            output_text = out.get("output", "")
            # Truncate long outputs
            if len(output_text) > 8000:
                output_text = output_text[:4000] + f"\n... [{len(output_text) - 8000} chars elided] ...\n" + output_text[-4000:]
            obs.append(self._format_message(
                "user",
                f"<returncode>{out['returncode']}</returncode>\n"
                f"<output>\n{output_text}\n</output>"
                + (f"\n<exception>{out['exception_info']}</exception>" if out.get("exception_info") else ""),
            ))
        return obs

    def _load_tool_schemas(self) -> None:
        for spec in self.env.get_tools_spec():
            self._tool_schemas[spec["name"]] = spec.get("parameters", {})

    # ------------------------------------------------------------------
    # Dead-Loop Monitor (Phase 0) — heuristic loop-detection without LLM calls
    # ------------------------------------------------------------------

    def _auto_tune_from_complexity(self) -> None:
        """Analyze target molecule and adjust loop parameters."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(self.blackboard.target_smiles)
            if mol is None:
                return
            n_atoms = mol.GetNumAtoms()
            n_rings = mol.GetRingInfo().NumRings()
            n_chiral = len(Chem.FindMolChiralCenters(mol))

            if n_atoms < 15 and n_rings <= 1 and n_chiral <= 1:
                self._complexity_level = "simple"
                self.config.step_limit = min(self.config.step_limit, 30)
                self._review_count = 2  # effectively skip review for simple molecules
            elif n_atoms < 30 and n_rings <= 3:
                self._complexity_level = "moderate"
                self.config.step_limit = max(self.config.step_limit, 100)
            else:
                self._complexity_level = "complex"
                self.config.step_limit = max(self.config.step_limit, 150)
                self.config.backtrack_patience = max(self.config.backtrack_patience, 3)

            self._last_progress_round = 0  # reset after auto-tune
            self.logger.info(
                f"Auto-tuned for {self._complexity_level} molecule "
                f"(atoms={n_atoms}, rings={n_rings}, chiral={n_chiral}): "
                f"step_limit={self.config.step_limit}, "
                f"backtrack_patience={self.config.backtrack_patience}"
            )
        except Exception:
            pass

    def _check_tool_type_cycling(self) -> str:
        """0A — Detect tool-type cycling patterns without progress."""
        if not self.config.enable_dead_loop_monitor:
            return ""

        seq = list(self._tool_sequence)
        if len(seq) < self.config.cycling_window:
            return ""

        # Pattern 1: alternating disconnect → propose without evaluate
        non_think_seq = [t for t in seq if t != "think"]
        if len(non_think_seq) >= 6:
            recent = non_think_seq[-6:]
            if "evaluate" not in recent:
                if recent.count("disconnect") >= 2 and recent.count("propose") >= 2:
                    return (
                        "Dead loop detected: You have been alternating between disconnect "
                        "and propose for multiple rounds without evaluating any routes. "
                        "STOP exploring new bonds — call evaluate NOW to assess your best "
                        "candidates, then submit or backtrack."
                    )

        # Pattern 2: consecutive evaluate without propose
        if len(non_think_seq) >= 4 and all(t == "evaluate" for t in non_think_seq[-4:]):
            return (
                "Dead loop detected: You have called evaluate 4+ times consecutively "
                "without any new proposals. If scores are sufficient, SUBMIT the route. "
                "If not, call propose with different template_indices or use_fallback=True."
            )

        # Pattern 3: disconnect-disconnect-disconnect without any propose
        if len(non_think_seq) >= 3 and all(t == "disconnect" for t in non_think_seq[-3:]):
            return (
                "You have called disconnect 3+ times consecutively without proposing any "
                "reactions. Pick the top matching template(s) and call propose now."
            )

        return ""

    def _check_semantic_repeats(self, action: dict) -> str:
        """0B — Detect semantically equivalent repeat calls (same tool + same key molecule)."""
        if not self.config.enable_dead_loop_monitor:
            return ""

        tool = action.get("tool", "")
        params = action.get("parameters", {})

        # Extract the key molecular identifier from params
        key_smiles = params.get("smiles", "") or params.get("target", "") or ""
        key_id = json.dumps(params.get("reactions", params.get("route_ids", "")),
                            sort_keys=True, ensure_ascii=False)

        semantic_key = (tool, key_smiles or key_id)
        if not semantic_key[1]:  # no identifiable key, skip
            return ""

        self._semantic_call_counts[semantic_key] = \
            self._semantic_call_counts.get(semantic_key, 0) + 1

        count = self._semantic_call_counts[semantic_key]
        if count >= self.config.semantic_repeat_threshold:
            # Reset to allow a few more after warning
            self._semantic_call_counts[semantic_key] = 0
            if key_smiles:
                return (
                    f"Semantic repeat detected: You have called `{tool}` on the same "
                    f"target ({key_smiles}) {count} times with minor parameter variations. "
                    f"Stop fine-tuning parameters — make a decision based on existing results."
                )
            else:
                return (
                    f"Semantic repeat detected: You have called `{tool}` with the same "
                    f"content {count} times. Move forward or backtrack."
                )

        return ""

    def _check_stagnation(self) -> str:
        """0C — Check if no substantive progress has been made for N rounds.

        In retrosynthesis mode: checks for new evaluations, stock hits, or proposals.
        In design mode: checks for new candidates, chirality analyses, or ligand classifications.
        """
        if not self.config.enable_dead_loop_monitor:
            return ""

        current_round = self.blackboard.iteration_count

        # Determine if progress was made — depends on mode
        has_progress = False
        if self.blackboard.mode == "retrosynthesis":
            has_progress = (
                bool(self.blackboard.scores)
                or bool(self.blackboard.stock_hits)
            )
        else:  # design mode
            has_progress = (
                bool(self.blackboard.design_candidates)
                or bool(self.blackboard.design_evaluations)
            )
        # Also consider proposals as progress in either mode
        if not has_progress:
            has_progress = bool(self.blackboard.proposal_results)

        if has_progress:
            if current_round > self._last_progress_round:
                self._last_progress_round = current_round
                return ""

        rounds_since_progress = current_round - self._last_progress_round

        # Don't warn too early or too frequently
        if current_round < 5 or \
           current_round - self._last_stagnation_warning_round < self.config.stagnation_rounds:
            return ""

        if rounds_since_progress >= self.config.stagnation_rounds:
            self._last_stagnation_warning_round = current_round
            return (
                f"No substantive progress in the last {rounds_since_progress} rounds "
                f"(no new evaluations or stock hits). Consider: "
                f"(1) committing to your best route and submitting, "
                f"(2) backtracking to an alternative disconnection strategy, "
                f"or (3) using think to re-evaluate your approach."
            )

        return ""

    def _check_early_exit(self) -> str:
        """0D — Suggest early exit when results are good enough.

        In retrosynthesis mode: checks route scores and stock availability.
        In design mode: checks if candidates have been validated by chirality + classification tools.
        """
        if not self.config.enable_early_exit_hint:
            return ""

        # Retrosynthesis mode: check scores
        if self.blackboard.mode == "retrosynthesis":
            scores = self.blackboard.scores
            stock = self.blackboard.stock_hits
            if not scores:
                return ""
            best_score = 0.0
            for rid, s in scores.items():
                total = s["total"] if isinstance(s, dict) else float(s)
                if total > best_score:
                    best_score = total
            if best_score >= self.config.early_exit_score_threshold:
                stock_count = len(stock) if stock else 0
                return (
                    f"<system_hint>\n"
                    f"Good routes detected (best score: {best_score:.3f}, "
                    f"{stock_count} precursors in stock, {len(scores)} routes evaluated).\n"
                    f"If you are satisfied with the current routes, you can submit now by "
                    f"including COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT in your response.\n"
                    f"Otherwise, continue exploring alternative disconnection strategies.\n"
                    f"</system_hint>"
                )

        # Design mode: check if candidates have been generated and validated
        else:
            candidates = self.blackboard.design_candidates
            evaluated = self.blackboard.design_evaluations
            if candidates and evaluated and len(candidates) >= 2 and len(evaluated) >= 2:
                return (
                    f"<system_hint>\n"
                    f"You have generated {len(candidates)} candidates and run "
                    f"{len(evaluated)} evaluations. If the best candidate meets the "
                    f"design constraints (symmetry, chirality, scaffold), you can submit now. "
                    f"Include COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT in your response.\n"
                    f"</system_hint>"
                )

        return ""

    def _check_evaluate_loop(self, action: dict) -> str:
        """0E — Detect repeated evaluate on the same route_ids."""
        if not self.config.enable_dead_loop_monitor:
            return ""

        if action.get("tool", "") != "evaluate":
            return ""

        params = action.get("parameters", {})
        route_ids = set(params.get("route_ids", []) or [])
        if not route_ids:
            # If no explicit route_ids, check if reactions list is the same
            reactions = params.get("reactions", []) or []
            route_ids = {json.dumps(r, sort_keys=True, ensure_ascii=False) for r in reactions}

        if route_ids == self._last_eval_route_ids and route_ids:
            self._evaluate_call_count += 1
        else:
            self._evaluate_call_count = 1
            self._last_eval_route_ids = route_ids

        if self._evaluate_call_count >= 3:
            self._evaluate_call_count = 0
            self._last_eval_route_ids = set()
            return (
                "The same routes have been evaluated 3+ times. Scores won't change. "
                "Make a decision NOW: submit the best route if scores are adequate, "
                "or backtrack to a completely different disconnection strategy."
            )

        return ""

    def _run_dead_loop_checks(self, action: dict | None = None) -> list[str]:
        """Aggregate all dead-loop monitor warnings. Returns list of non-empty messages."""
        warnings = []
        msg = self._check_tool_type_cycling()
        if msg:
            warnings.append(msg)
        if action:
            msg = self._check_semantic_repeats(action)
            if msg:
                warnings.append(msg)
            msg = self._check_evaluate_loop(action)
            if msg:
                warnings.append(msg)
        msg = self._check_stagnation()
        if msg:
            warnings.append(msg)
        msg = self._check_early_exit()
        if msg:
            warnings.append(msg)
        return warnings

    def _validate_action(self, action: dict) -> tuple[bool, str]:
        """Validate action parameters against tool JSON schema."""
        if not self.config.enable_schema_validation:
            return True, ""
        tool_name = action.get("tool", "")
        schema = self._tool_schemas.get(tool_name)
        if not schema:
            return True, ""
        params = action.get("parameters", {})
        required = schema.get("required", [])
        for key in required:
            if key not in params:
                return False, f"Missing required parameter '{key}' for tool '{tool_name}'"
        return True, ""

    def _check_repeated_action(self, action: dict) -> bool:
        """Detect if the exact same tool+parameters was already called too many times."""
        if not self.config.enable_repeated_action_guard:
            return False
        key = (action.get("tool", ""), json.dumps(action.get("parameters", {}), sort_keys=True, ensure_ascii=False))
        count = sum(1 for past in self._action_history if past == key)
        return count >= self.config.max_repeated_actions

    def _record_action(self, action: dict) -> None:
        key = (action.get("tool", ""), json.dumps(action.get("parameters", {}), sort_keys=True, ensure_ascii=False))
        self._action_history.append(key)

    def _check_backtracking(self) -> tuple[bool, str]:
        """Check if the agent is stuck in a low-score region and should backtrack."""
        if not self.config.enable_backtracking:
            return False, ""
        scores = self.blackboard.scores
        if not scores:
            return False, ""
        best = max(v["total"] if isinstance(v, dict) else float(v) for v in scores.values())
        if self._last_best_score is not None and best <= self._last_best_score + 1e-6:
            if best < self.config.backtrack_score_threshold:
                self._low_score_count += 1
        else:
            self._low_score_count = 0
        self._last_best_score = best
        if self._low_score_count >= self.config.backtrack_patience:
            self._low_score_count = 0
            return True, (
                f"Best route score has stayed below {self.config.backtrack_score_threshold} "
                f"for {self.config.backtrack_patience} rounds. Consider backtracking to alternative "
                f"bonds or strategies."
            )
        return False, ""

    def _track_branch_from_action(self, action: dict, result: dict) -> None:
        """Phase 4: Automatically update branch tracking based on tool results.

        Detects when the LLM starts a new branch (disconnect → propose),
        evaluates routes, or checks stock, and updates the blackboard branch table.
        """
        tool = action.get("tool", "")
        params = action.get("parameters", {})
        bb = self.blackboard

        if tool == "disconnect":
            # A new disconnection round = potential new branches
            bonds = result.get("bonds", [])
            for bond_info in bonds[:5]:
                tid = bond_info.get("template_index", 0)
                classification = bond_info.get("classification", "?")
                # Check if we already have a branch for this template
                existing = any(
                    b.get("template_index") == tid for b in bb.branches
                )
                if not existing:
                    bb.track_branch(
                        bond_name=classification,
                        template_index=tid,
                        classification=classification,
                        status="exploring",
                    )

        elif tool == "propose":
            # After proposing, mark related branches as having precursors
            reactions = result.get("reactions", [])
            tids_seen = set()
            for rxn in reactions:
                tid = rxn.get("template_index", 0)
                tids_seen.add(tid)
                precursors = rxn.get("precursors", [])
                # Find matching branch
                for i, b in enumerate(bb.branches):
                    if b.get("template_index") == tid:
                        bb.update_branch(i, precursor_count=len(precursors))

        elif tool == "evaluate":
            # After evaluating, update scores on branches
            scores = result.get("scores", {})
            for rid, s in scores.items():
                total = s.get("total", s) if isinstance(s, dict) else float(s)
                # Try to match route_id to a branch by score magnitude
                # Simple approach: update the highest-scoring branch that hasn't been evaluated
                for i, b in enumerate(bb.branches):
                    if b.get("status") in ("exploring",) and b.get("best_score") is None:
                        bb.update_branch(i, best_score=total, status="evaluated")
                        break

        elif tool == "check_stock":
            # Update stock counts on branches
            in_stock = result.get("in_stock", {})
            stock_count = len(in_stock)
            if stock_count > 0:
                for i, b in enumerate(bb.branches):
                    if b.get("status") in ("exploring", "evaluated"):
                        bb.update_branch(i, stock_count=stock_count)
                        if b.get("best_score", 0) >= self.config.early_exit_score_threshold:
                            bb.update_branch(i, status="high_score")

        elif tool in ("analyze_chirality", "classify_ligand", "design_ligand"):
            # Phase 4 extended: track design evaluations
            pass  # design tools update design_evaluations/candidates via blackboard.update()

    # ------------------------------------------------------------------
    # Phase 0F — Design Auditor: chemical correctness review for design mode
    # ------------------------------------------------------------------

    def _audit_design(self) -> str | None:
        """Review the current best design candidate for chemical correctness.

        Only triggers in design mode when the LLM has validated candidates.
        Checks: symmetry claims vs structure, coordination saturation,
        chirality source, auxiliary ligand suitability, literature comparability.
        """
        if not self.config.enable_design_auditor:
            return None
        if self.blackboard.mode != "design":
            return None

        # Trigger: have at least 2 evaluations (chirality + classification done)
        evaluated = self.blackboard.design_evaluations
        if len(evaluated) < 2:
            return None

        # Don't audit too frequently
        if self._design_audit_count >= 2:
            return None
        current_round = self.blackboard.iteration_count
        if current_round - self._last_design_audit_round < 3:
            return None

        self._last_design_audit_round = current_round
        self._design_audit_count += 1

        # Build a summary of what's been designed so far
        summary_parts = []
        candidates = self.blackboard.design_candidates
        if candidates:
            summary_parts.append(f"Generated {len(candidates)} candidate(s).")
            for c in candidates[-3:]:  # last 3
                summary_parts.append(f"  - {json.dumps(c, ensure_ascii=False)[:300]}")
        for ev in evaluated[-3:]:
            summary_parts.append(f"Evaluation: {json.dumps(ev, ensure_ascii=False)[:300]}")

        design_summary = "\n".join(summary_parts) if summary_parts else "(no design data yet)"

        # Build the auditor prompt
        auditor_prompt = (
            "### Role Switch: Catalyst Design Auditor ###\n\n"
            "You are now acting as an **adversarial chemistry reviewer** for the catalyst "
            "design proposal above. Your ONLY task is to identify chemical flaws. "
            "Be skeptical — challenge every claim.\n\n"
            f"#### Current Design Data:\n{design_summary}\n\n"
            "#### Design Audit Checklist (answer each explicitly):\n\n"
            "1. **Symmetry** — What symmetry does the user request? (C2 / C3 / etc.) "
            "Does each candidate ACTUALLY have that symmetry? Count the ligands "
            "and their arrangement. If the user asked for C2 and the candidate has "
            "3 identical ligands (C3), flag it as WRONG SYMMETRY.\n\n"
            "2. **Coordination Saturation** — Is the metal coordinatively saturated "
            "(18 e⁻, all sites occupied)? If so, WHERE does the substrate bind? "
            "A catalyst that has NO open coordination sites is catalytically DEAD. "
            "Flag any candidate with no labile ligands.\n\n"
            "3. **Chirality Source** — Where does the chirality come from? "
            "(a) metal-centered Δ/Λ, (b) ligand point-chirality, "
            "(c) atropisomerism/axial? If metal-centered, does the candidate's "
            "ligand arrangement actually create a chiral environment?\n\n"
            "4. **Auxiliary Ligands** — Are the labile/ancillary ligands correct for "
            "the target reaction? For C-H activation: MeCN, H₂O, or weakly-coordinating "
            "anions (OTf⁻, BF₄⁻, PF₆⁻) are GOOD. acac⁻, Cp*, strongly-bound Cl⁻ "
            "are BAD — they block the metal center. Check each candidate.\n\n"
            "5. **Literature Precedent** — Is there a known catalyst in the literature "
            "with similar architecture? If the candidate has identical architecture to "
            "a known OLED luminophore but the user asked for a C-H activation catalyst, "
            "flag this as a MISMATCHED APPLICATION.\n\n"
            "6. **Valence/Electron Count** — Rapid check: does the candidate violate "
            "the 18-electron rule or common oxidation states for this metal?\n\n"
            "#### Output format:\n"
            "For each issue found, write: `ISSUE: [SEVERITY] <description>`\n"
            "Severity: [CRITICAL] = fundamentally wrong, [WARNING] = suboptimal, "
            "[INFO] = minor concern.\n\n"
            "Then write a **FIX_SUGGESTIONS** section with concrete recommendations. "
            "Name specific structures, ligand types, coordination geometries, and "
            "precursors. For example: 'Use fac-[Ir(C^N)2(MeCN)2]+ where the two MeCN "
            "are labile and create vacant sites for substrate C-H binding. The facial "
            "arrangement of two C^N ligands gives C2 symmetry.'\n\n"
            "End with: `AUDIT_VERDICT: APPROVE / REVISE / REJECT`\n\n"
            "CRITICAL issues = REJECT. Multiple WARNINGs = REVISE. No issues = APPROVE."
        )

        try:
            # Use isolated review with tools disabled. We send the auditor prompt
            # via model.query() which always passes tools. So wrap with a no-tools query.
            # Hack: temporarily set env to empty tools for this call, then restore.
            # Better approach: inject a strong instruction to NOT call tools.
            audit_prompt_wrapped = (
                auditor_prompt
                + "\n\n**CRITICAL: Do NOT call any tools in your response. "
                "This is a text-only audit. Output ONLY the audit report with "
                "ISSUE lines and AUDIT_VERDICT. No JSON, no tool calls.**"
            )
            audit_message = self.model.query(
                self.messages + [self._format_message("user", audit_prompt_wrapped)]
            )
            audit_content = audit_message.get("content", "")
            self.cost += audit_message.get("extra", {}).get("cost", 0.0)
            self.n_calls += 1

            formatted = (
                "<role:design_auditor>\n"
                f"{audit_content}\n"
                "</role:design_auditor>\n\n"
                "You are now back in **Designer mode**. Address all CRITICAL and WARNING "
                "issues from the auditor before submitting. If the auditor says REJECT, "
                "you MUST redesign the catalyst. If REVISE, fix the issues. "
                "If APPROVE, you may submit."
            )
            return formatted
        except Exception as e:
            self.logger.warning(f"Design audit failed: {e}")
            return None

    def _reflection_prompt(self, action: dict, output: dict) -> str:
        """Generate a short reflection prompt after tool execution."""
        if not self.config.enable_reflection:
            return ""
        return (
            f"Reflect briefly on the result of calling `{action['tool']}`:\n"
            f"- Was the output useful and chemically sensible?\n"
            f"- Are there contradictions with previous results or expected chemistry?\n"
            f"- What should the next step be? Avoid repeating the same tool with the same parameters."
        )

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute tool calls from the LLM response, update blackboard, return observations."""
        actions = message.get("extra", {}).get("actions", [])
        if not actions:
            return []

        filtered_actions = []
        warnings = []
        for action in actions:
            valid, reason = self._validate_action(action)
            if not valid:
                warnings.append(f"Invalid action skipped: {reason}")
                continue
            if self._check_repeated_action(action):
                warnings.append(
                    f"Repeated action detected for `{action['tool']}`; skipping to avoid loops."
                )
                continue
            filtered_actions.append(action)
            self._record_action(action)

        outputs = []
        for action in filtered_actions:
            result = self.env.execute(action)
            outputs.append((action, result))

            # Track tool sequence for cycling detection
            self._tool_sequence.append(action.get("tool", "bash"))

            if result["returncode"] == 0:
                try:
                    parsed = json.loads(result["output"])
                    tool_name = action.get("tool", "bash")
                    # Think tool does not increment iteration count
                    if tool_name != "think":
                        self.blackboard.update(tool_name, parsed)
                    # Phase 4: Track branch exploration
                    self._track_branch_from_action(action, parsed)
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass

        # Run dead-loop monitor per action
        for action in filtered_actions:
            loop_warnings = self._run_dead_loop_checks(action)
            if loop_warnings:
                warnings.extend(loop_warnings)

        should_backtrack, backtrack_msg = self._check_backtracking()
        return self.add_messages(
            *self._format_observations(outputs, warnings, should_backtrack, backtrack_msg)
        )

    def _format_observations(
        self,
        outputs: list[tuple[dict, dict]],
        warnings: list[str] | None = None,
        backtrack: bool = False,
        backtrack_msg: str = "",
    ) -> list[dict]:
        """Format tool execution outputs as observation messages.

        Think tool outputs are formatted as <thinking> blocks.
        Other tool outputs get reflection + enhanced observation prompts.
        """
        obs = []
        for action, out in outputs:
            tool_name = action.get("tool", "bash")

            # Phase 1: Think tool gets special formatting
            if tool_name == "think":
                params = action.get("parameters", {})
                thought = params.get("thought", "")
                concern = params.get("concern", "")
                next_action = params.get("next_action", "")

                parts = ["<thinking>", thought]
                if concern:
                    parts.append(f"\n⚠ Concern: {concern}")
                if next_action:
                    parts.append(f"\n→ Next planned action: {next_action}")
                parts.append("</thinking>")
                obs.append(self._format_message("user", "\n".join(parts)))
                continue

            output_text = out.get("output", "")
            if len(output_text) > 8000:
                output_text = output_text[:4000] + f"\n... [{len(output_text) - 8000} chars elided] ...\n" + output_text[-4000:]

            parts = [
                f"<tool>{tool_name}</tool>",
                f"<returncode>{out['returncode']}</returncode>",
                f"<output>\n{output_text}\n</output>",
            ]
            if out.get("exception_info"):
                parts.append(f"<exception>{out['exception_info']}</exception>")

            # Enhanced observation prompt (Phase 2)
            summary = self._summarize_result(tool_name, output_text)
            if self.config.enable_enhanced_observation:
                parts.append(
                    f"<observe>\n"
                    f"Tool `{tool_name}` completed.\n"
                    f"{summary}\n"
                    f"Now think through:\n"
                    f"1. Is this result chemically sensible?\n"
                    f"2. Does anything conflict with prior results?\n"
                    f"3. Should you continue this branch, backtrack, or submit?\n"
                    f"Use `think` if you need to reason through complex decisions.\n"
                    f"</observe>"
                )
            else:
                # Original reflection prompt (kept for backward compat)
                reflection = self._reflection_prompt(action, out)
                if reflection:
                    parts.append(f"<reflection>\n{reflection}\n</reflection>")

            obs.append(self._format_message("user", "\n".join(parts)))

        if warnings:
            obs.append(self._format_message(
                "user",
                "<system_warning>\n" + "\n".join(f"- {w}" for w in warnings) + "\n</system_warning>"
            ))

        if backtrack:
            obs.append(self._format_message(
                "user",
                f"<system_backtrack>\n{backtrack_msg}\n</system_backtrack>"
            ))

        # Phase 4: Dynamically inject branch tracking table when branches exist
        branch_table = self._render_branch_status()
        if branch_table:
            obs.append(self._format_message("user", branch_table))

        return obs

    def _render_branch_status(self) -> str:
        """Render the current branch exploration table for dynamic injection."""
        if not self.blackboard.branches:
            return ""
        table = self.blackboard._build_branch_table()
        active = sum(1 for b in self.blackboard.branches if b.get("status") not in ("submitted", "abandoned"))
        return (
            "<branch_status>\n"
            f"{table}\n"
            f"\nUse this table to track which branches you've explored. "
            f"Avoid re-exploring abandoned branches. Focus on high-score or "
            f"promising branches. When all active branches are resolved, submit.\n"
            f"</branch_status>"
        )

    def _summarize_result(self, tool_name: str, output_text: str) -> str:
        """Extract a short summary from tool output for the observation prompt."""
        try:
            data = json.loads(output_text)
        except (json.JSONDecodeError, TypeError):
            return f"Output: {output_text[:300]}..."

        if tool_name == "disconnect":
            bonds = data.get("bonds", [])
            matching = sum(1 for b in bonds if b.get("matching"))
            return (
                f"Key results: {len(bonds)} disconnection sites found, "
                f"{matching} substructure-matched. Top bond: "
                f"template #{bonds[0].get('template_index', '?') if bonds else 'none'} "
                f"({bonds[0].get('classification', '?') if bonds else 'none'})."
            )
        elif tool_name == "propose":
            reactions = data.get("reactions", [])
            precursors = set()
            for r in reactions[:5]:
                precursors.update(r.get("precursors", []))
            return (
                f"Key results: {len(reactions)} reactions generated, "
                f"{len(precursors)} unique precursors from top 5 reactions."
            )
        elif tool_name == "evaluate":
            scores = data.get("scores", {})
            if scores:
                best = max(
                    (v["total"] if isinstance(v, dict) else float(v))
                    for v in scores.values()
                )
                return (
                    f"Key results: {len(scores)} routes evaluated. "
                    f"Best score: {best:.3f}."
                )
            return f"Key results: {len(scores)} routes evaluated."
        elif tool_name == "check_stock":
            in_stock = data.get("in_stock", {})
            total = data.get("total_checked", len(in_stock))
            return (
                f"Key results: {len(in_stock)}/{total} precursors in stock."
            )
        elif tool_name == "search_literature":
            routes = data.get("known_routes", [])
            return f"Key results: {len(routes)} literature precedents found."
        else:
            return f"Output: {output_text[:300]}..."


    def _save_trajectory(self) -> None:
        """Save the trajectory to the output path if configured."""
        if self.config.output_path:
            try:
                import json
                path = Path(self.config.output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(self.messages, indent=2, ensure_ascii=False))
            except Exception:
                pass


# --- Exception classes (mirroring mini-swe-agent's exception hierarchy) ---

class _InterruptAgentFlow(Exception):
    def __init__(self, messages: list[dict]):
        self.messages = messages


class _FormatError(_InterruptAgentFlow):
    pass


class _LimitsExceeded(_InterruptAgentFlow):
    pass


class _TimeExceeded(_LimitsExceeded):
    pass
