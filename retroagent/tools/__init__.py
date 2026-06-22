"""RetroTool protocol and base classes. Each tool is a pure function that returns
structured JSON results — no decision logic lives inside tools."""

from typing import Protocol


class RetroTool(Protocol):
    """Protocol for retrosynthetic tools.

    Key constraint: Tools do NOT make decisions. They return complete,
    ranked result sets, and the Planner (LLM) chooses what to use.
    """

    name: str
    description: str

    def execute(self, parameters: dict) -> str:
        """Execute the tool and return a JSON string of structured results."""
        ...

    def parameters_schema(self) -> dict:
        """Return a JSON Schema describing the tool's input parameters."""
        ...
