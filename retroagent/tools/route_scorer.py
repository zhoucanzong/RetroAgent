"""RouteScorerTool: enumerate and score candidate routes from reaction records."""

import json

from retroagent.route_graph import RouteGraph


class RouteScorerTool:
    name = "score_routes"
    description = (
        "Build an AND-OR route graph from proposed reaction records, enumerate "
        "candidate complete or partial routes, and score them by route closure, "
        "step count, unresolved precursors, diversity, convergence, and "
        "protection/deprotection burden. Use after propose/evaluate/check_stock "
        "to compare whole routes rather than isolated reactions."
    )

    def execute(self, parameters: dict) -> str:
        target_smiles = parameters.get("target_smiles", "")
        reactions = parameters.get("reactions", [])
        stock_hits = parameters.get("stock_hits", {})
        max_depth = int(parameters.get("max_depth", 4) or 4)
        max_routes = int(parameters.get("max_routes", 20) or 20)

        graph = RouteGraph()
        if target_smiles:
            graph.add_target(target_smiles)
        for reaction in reactions:
            graph.add_reaction(reaction, target=target_smiles)
        for smiles, source in stock_hits.items():
            graph.mark_stock(smiles, source=str(source))

        routes = graph.scored_routes(max_depth=max_depth, max_routes=max_routes)
        return json.dumps({
            "target_smiles": target_smiles,
            "route_count": len(routes),
            "routes": routes,
            "route_graph": graph.to_dict(),
            "summary": {
                "solved_route_count": sum(1 for route in routes if route["metadata"].get("solved")),
                "partial_route_count": sum(1 for route in routes if not route["metadata"].get("solved")),
                "best_score": routes[0]["score"] if routes else None,
            },
        }, ensure_ascii=False)

    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "target_smiles": {"type": "string", "description": "Target molecule SMILES."},
                "reactions": {
                    "type": "array",
                    "description": "Reaction records from propose.",
                    "items": {"type": "object"},
                },
                "stock_hits": {
                    "type": "object",
                    "description": "In-stock precursor mapping from check_stock.",
                },
                "max_depth": {"type": "integer", "description": "Maximum recursive route depth."},
                "max_routes": {"type": "integer", "description": "Maximum routes to return."},
            },
            "required": ["target_smiles", "reactions"],
        }
