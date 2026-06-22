"""BashTool: execute shell commands. Wraps subprocess.run like mini-swe-agent's
LocalEnvironment. Always available as the catch-all tool."""

import json
import os
import signal
import subprocess
import platform


class BashTool:
    name = "bash"
    description = (
        "Execute arbitrary shell commands. Use for RDKit validation, file operations, "
        "molecule property calculation, or any ad-hoc computation not covered by "
        "the specialized chemistry tools."
    )

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def execute(self, parameters: dict) -> str:
        command = parameters.get("command", "")
        cwd = parameters.get("cwd", os.getcwd())
        timeout = parameters.get("timeout", self.timeout)

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
                return json.dumps({
                    "output": stdout,
                    "returncode": -1,
                    "error": f"Command timed out after {timeout}s",
                })

            return json.dumps({
                "output": stdout,
                "returncode": process.returncode,
            })
        except Exception as e:
            return json.dumps({
                "output": "",
                "returncode": -1,
                "error": str(e),
            })

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory for the command"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30)"
                },
            },
            "required": ["command"]
        }
