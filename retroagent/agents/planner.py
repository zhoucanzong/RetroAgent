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
        return self.execute_actions(self.query())

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
