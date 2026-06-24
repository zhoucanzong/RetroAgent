"""ThinkTool — a virtual "think" tool that gives the LLM explicit reasoning space.

Inspired by Claude's "think" tool and extended thinking architecture.
When the model "calls" this tool, no actual computation happens — the reasoning
content is simply recorded and formatted back into the conversation, giving the
model a structured place to pause and analyze before the next real action.

This does NOT consume an extra LLM call; the model can invoke think + the next
real tool in the same response via native function calling (multiple tool_calls).
"""

import json


class ThinkTool:
    """Virtual reasoning tool. Records structured thinking without side effects."""

    name = "think"
    description = (
        "Pause and reason through the current situation before taking action. "
        "Use this to analyze tool results, plan strategy, compare options, and "
        "catch potential issues BEFORE calling other tools. This tool has no side "
        "effects — it simply records your reasoning for future reference."
    )

    def execute(self, parameters: dict) -> str:
        thought = parameters.get("thought", "")
        concern = parameters.get("concern", "")
        next_action = parameters.get("next_action", "")

        result = {
            "status": "thought_recorded",
            "char_count": len(thought) + len(concern) + len(next_action),
        }
        if concern:
            result["concern_noted"] = True
        if next_action:
            result["next_action_planned"] = next_action

        return json.dumps(result)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": (
                        "Your detailed reasoning about the current state: analysis of "
                        "previous results, strategic plan for the next steps, comparison "
                        "of alternative approaches, or chemical reasoning about bond "
                        "disconnections and precursor feasibility."
                    ),
                },
                "concern": {
                    "type": "string",
                    "description": (
                        "Any specific concerns, risks, or potential pitfalls you have "
                        "identified (e.g., chemoselectivity risks, missing stereochemistry, "
                        "non-matching templates). Optional."
                    ),
                },
                "next_action": {
                    "type": "string",
                    "description": (
                        "What specific tool you plan to call next and why. "
                        "This helps maintain strategic focus. Optional but recommended."
                    ),
                },
            },
            "required": ["thought"],
        }
