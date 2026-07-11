"""Deterministic negative-feedback oracle used by closed-loop benchmarks."""

import json


class FeedbackOracleTool:
    name = "feedback_oracle"
    description = (
        "Reveal the benchmark's withheld computational feedback for the currently "
        "preferred route. Use the returned failure evidence to update route beliefs and replan."
    )

    def __init__(self, feedback: dict, *, expected_route_id: str = "route_a"):
        self.feedback = dict(feedback)
        self.expected_route_id = expected_route_id

    def execute(self, parameters: dict) -> str:
        queried_route_id = parameters.get("route_id", "")
        if queried_route_id != self.expected_route_id:
            return json.dumps({
                "error": f"No withheld feedback is registered for route: {queried_route_id}",
                "queried_route_id": queried_route_id,
            }, ensure_ascii=True, sort_keys=True)
        return json.dumps({
            "queried_route_id": queried_route_id,
            "failure": dict(self.feedback),
        }, ensure_ascii=True, sort_keys=True)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "route_id": {"type": "string", "description": "Route being tested."},
            },
            "required": ["route_id"],
        }
