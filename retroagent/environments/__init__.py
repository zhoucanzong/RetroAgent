"""RetroEnvironment: tool dispatcher implementing mini-swe-agent's Environment protocol.

This is the unified entry point for all tool execution. The Planner (LLM) calls
tools through this environment, which distributes calls to the appropriate tool.
"""

import json
import os
import signal
import subprocess
from typing import Any

from pydantic import BaseModel


class RetroEnvConfig(BaseModel):
    timeout: int = 120
    cwd: str = ""
    env: dict[str, str] = {}


class RetroEnvironment:
    """Implements the Environment protocol from mini-swe-agent.

    action = {"tool": "disconnect", "parameters": {"smiles": "..."}}
    -> executes DisconnectionTool -> returns structured observation dict.
    """

    def __init__(self, *, config_class: type = RetroEnvConfig, **kwargs):
        self.config = config_class(**kwargs)
        self._tools: dict[str, Any] = {}

    def register(self, name: str, tool: Any) -> None:
        self._tools[name] = tool

    def execute(self, action: dict) -> dict[str, Any]:
        tool_name = action.get("tool", "bash")
        parameters = action.get("parameters", {})

        # bash is handled specially — subprocess.run
        if tool_name == "bash":
            return self._execute_bash(parameters)

        tool = self._tools.get(tool_name)
        if tool is None:
            return {
                "output": json.dumps({"error": f"Unknown tool: {tool_name}"}),
                "returncode": -1,
                "exception_info": f"Unknown tool: {tool_name}",
            }

        try:
            result = tool.execute(parameters)
            return {"output": result, "returncode": 0, "exception_info": ""}
        except Exception as e:
            return {
                "output": json.dumps({"error": str(e)}),
                "returncode": -1,
                "exception_info": f"{type(e).__name__}: {e}",
            }

    def get_tools_spec(self) -> list[dict]:
        specs = []
        for name, tool in self._tools.items():
            try:
                schema = tool.parameters_schema()
            except (AttributeError, NotImplementedError):
                schema = {"type": "object", "properties": {}}
            spec = {
                "name": name,
                "description": getattr(tool, "description", ""),
                "parameters": schema,
            }
            # Include examples if the tool provides them (context-engineering hint)
            examples = getattr(tool, "examples", None)
            if examples:
                spec["examples"] = examples
            specs.append(spec)
        # Always include bash
        specs.append({
            "name": "bash",
            "description": "Execute arbitrary shell commands (RDKit validation, file ops, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"}
                },
                "required": ["command"],
            },
        })
        return specs

    def _execute_bash(self, parameters: dict) -> dict[str, Any]:
        command = parameters.get("command", "")
        cwd = parameters.get("cwd", self.config.cwd or os.getcwd())
        timeout = parameters.get("timeout", self.config.timeout)

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                text=True,
                cwd=cwd,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
            try:
                stdout, _ = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL) if os.name == "posix" else process.kill()
                stdout, _ = process.communicate()
                return {
                    "output": stdout,
                    "returncode": -1,
                    "exception_info": f"Command timed out after {timeout}s",
                }
            return {"output": stdout, "returncode": process.returncode, "exception_info": ""}
        except Exception as e:
            return {"output": "", "returncode": -1, "exception_info": str(e)}

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }
