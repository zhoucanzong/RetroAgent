"""RetroPlanner: the central reasoning agent, inheriting mini-swe-agent's DefaultAgent.

This is the ONLY component that makes decisions. It:
1. Reads the blackboard state
2. Decides which tools to call and in what order
3. Interprets tool results and adjusts search strategy
4. Determines when a complete route has been found
"""

import json
import logging
import time
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
                        self._format_message("exit", "RepeatedFormatError",
                                             extra={"exit_status": "RepeatedFormatError", "submission": ""}),
                    )
                else:
                    self.add_messages(*e.messages)
            except Exception as e:
                self.add_messages(
                    self._format_message("exit", str(e),
                                         extra={"exit_status": type(e).__name__, "submission": ""})
                )
            finally:
                self._save_trajectory()
            if self.messages and self.messages[-1].get("role") == "exit":
                break

        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict]:
        """One iteration: query LLM -> execute tools -> add observations."""
        result = self.execute_actions(self.query())

        # After executing tools, optionally run the peer reviewer
        review = self._review_route()
        if review:
            result.extend(review)

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
            if result["returncode"] == 0:
                try:
                    parsed = json.loads(result["output"])
                    tool_name = action.get("tool", "bash")
                    self.blackboard.update(tool_name, parsed)
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass

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
        """Format tool execution outputs as observation messages, with reflection and backtracking hints."""
        obs = []
        for action, out in outputs:
            output_text = out.get("output", "")
            if len(output_text) > 8000:
                output_text = output_text[:4000] + f"\n... [{len(output_text) - 8000} chars elided] ...\n" + output_text[-4000:]

            parts = [
                f"<tool>{action.get('tool', 'bash')}</tool>",
                f"<returncode>{out['returncode']}</returncode>",
                f"<output>\n{output_text}\n</output>",
            ]
            if out.get("exception_info"):
                parts.append(f"<exception>{out['exception_info']}</exception>")

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

        return obs

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
