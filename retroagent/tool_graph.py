"""ToolGraph metadata for chemistry workflow planning.

The graph captures lightweight dependencies and recommended next steps. It does
not execute tools; it gives the Planner a structured map of how tools compose.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolNode:
    name: str
    category: str
    input_types: tuple[str, ...] = ()
    output_types: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    suggests_next: tuple[str, ...] = ()
    fallback_tools: tuple[str, ...] = ()
    safety_level: str = "low"
    purpose: str = ""


DEFAULT_TOOL_GRAPH: dict[str, ToolNode] = {
    "convert_identifier": ToolNode(
        name="convert_identifier",
        category="grounding",
        input_types=("name", "iupac", "cas", "smiles"),
        output_types=("smiles", "iupac", "cas"),
        suggests_next=("molecule_properties", "functional_groups"),
        purpose="Ground natural-language molecule identifiers into structure identifiers.",
    ),
    "molecule_properties": ToolNode(
        name="molecule_properties",
        category="grounding",
        input_types=("smiles",),
        output_types=("descriptors", "reactivity_alerts"),
        suggests_next=("functional_groups", "disconnect", "check_patent"),
        purpose="Compute deterministic RDKit descriptors and reactivity alerts.",
    ),
    "functional_groups": ToolNode(
        name="functional_groups",
        category="grounding",
        input_types=("smiles",),
        output_types=("functional_groups",),
        suggests_next=("disconnect", "recommend_conditions", "map_reaction"),
        purpose="Identify functional groups for chemoselectivity and compatibility checks.",
    ),
    "search_literature": ToolNode(
        name="search_literature",
        category="evidence",
        input_types=("smiles", "query"),
        output_types=("literature_precedents",),
        fallback_tools=("web_search", "fetch_url"),
        suggests_next=("research_memory", "disconnect", "propose"),
        purpose="Find known or analog route precedents.",
    ),
    "research_memory": ToolNode(
        name="research_memory",
        category="memory",
        input_types=("operation", "target_smiles", "query", "hypotheses", "critic_report", "experiment_plan"),
        output_types=("memory_records", "memory_summary", "record_id"),
        suggests_next=("search_literature", "disconnect", "chemical_critic"),
        fallback_tools=("web_search", "fetch_url"),
        purpose="Record, search, or summarize local JSONL memory for route hypotheses, critic findings, experiments, and outcomes.",
    ),
    "web_search": ToolNode(
        name="web_search",
        category="evidence",
        input_types=("query",),
        output_types=("web_results",),
        suggests_next=("fetch_url",),
        safety_level="medium",
        purpose="Search public web/scientific sources when local literature tools are insufficient.",
    ),
    "fetch_url": ToolNode(
        name="fetch_url",
        category="evidence",
        input_types=("url",),
        output_types=("document_text",),
        requires=("web_search",),
        suggests_next=("think",),
        safety_level="medium",
        purpose="Fetch and clean a source document for evidence grounding.",
    ),
    "disconnect": ToolNode(
        name="disconnect",
        category="retrosynthesis",
        input_types=("smiles",),
        output_types=("template_candidates", "disconnections"),
        suggests_next=("propose", "think"),
        fallback_tools=("functional_groups",),
        purpose="Generate ranked retrosynthetic template/disconnection candidates.",
    ),
    "propose": ToolNode(
        name="propose",
        category="retrosynthesis",
        input_types=("smiles", "template_indices"),
        output_types=("reaction_records",),
        requires=("disconnect",),
        suggests_next=("evaluate", "check_stock", "score_routes", "map_reaction"),
        fallback_tools=("disconnect",),
        purpose="Apply templates or fallback scans to generate precursor reactions.",
    ),
    "evaluate": ToolNode(
        name="evaluate",
        category="validation",
        input_types=("reaction_records",),
        output_types=("route_scores",),
        requires=("propose",),
        suggests_next=("check_stock", "recommend_conditions", "search_literature"),
        purpose="Score reaction feasibility and precursor availability heuristics.",
    ),
    "check_stock": ToolNode(
        name="check_stock",
        category="validation",
        input_types=("smiles_list",),
        output_types=("stock_hits",),
        suggests_next=("score_routes", "propose", "search_literature"),
        purpose="Check recursive termination candidates against available stock data.",
    ),
    "score_routes": ToolNode(
        name="score_routes",
        category="route_scoring",
        input_types=("target_smiles", "reaction_records", "stock_hits"),
        output_types=("scored_routes", "route_graph", "route_closure_metrics"),
        requires=("propose",),
        suggests_next=("chemical_critic", "design_experiments", "think"),
        fallback_tools=("evaluate", "check_stock"),
        purpose="Enumerate and score whole candidate routes from an AND-OR route graph.",
    ),
    "recommend_conditions": ToolNode(
        name="recommend_conditions",
        category="validation",
        input_types=("reaction_class", "reaction_smiles"),
        output_types=("conditions",),
        suggests_next=("map_reaction", "think"),
        purpose="Suggest reaction conditions or identify condition gaps.",
    ),
    "map_reaction": ToolNode(
        name="map_reaction",
        category="validation",
        input_types=("reaction_smiles",),
        output_types=("atom_mapping", "bond_changes"),
        suggests_next=("think", "evaluate", "chemical_critic"),
        purpose="Map atoms and expose bond changes for mechanistic auditing.",
    ),
    "check_patent": ToolNode(
        name="check_patent",
        category="risk",
        input_types=("smiles",),
        output_types=("patent_or_purchase_signal",),
        suggests_next=("think", "chemical_critic", "build_research_report"),
        purpose="Check novelty/purchasability signal via molbloom.",
    ),
    "chemical_critic": ToolNode(
        name="chemical_critic",
        category="risk",
        input_types=("target_smiles", "hypotheses", "reaction_records", "route_scores", "stock_hits", "route_graph", "scored_routes"),
        output_types=("critic_report", "audit_issues", "revision_required"),
        requires=("propose", "score_routes"),
        suggests_next=("think", "design_experiments", "build_research_report"),
        fallback_tools=("molecule_properties", "functional_groups", "evaluate", "check_stock"),
        purpose="Audit route hypotheses and route graphs for evidence, validation, precursor, mechanism, and strategy risks.",
    ),
    "design_experiments": ToolNode(
        name="design_experiments",
        category="experiment",
        input_types=("target_smiles", "hypotheses", "critic_report", "reaction_records", "route_graph"),
        output_types=("experiment_plan", "controls", "success_criteria", "failure_diagnostics"),
        requires=("chemical_critic",),
        suggests_next=("build_research_report", "think"),
        fallback_tools=("recommend_conditions", "map_reaction", "search_literature"),
        safety_level="medium",
        purpose="Convert critic findings into prioritized validation experiments with controls and decision points.",
    ),
    "build_research_report": ToolNode(
        name="build_research_report",
        category="reporting",
        input_types=("target_smiles", "hypotheses", "evidence", "risk_flags", "next_experiments"),
        output_types=("structured_research_report", "completeness_audit"),
        requires=("think", "chemical_critic", "design_experiments"),
        suggests_next=("research_memory", "think"),
        fallback_tools=("think", "chemical_critic", "design_experiments"),
        purpose="Normalize route hypotheses and check evidence/risk/experiment completeness before submission.",
    ),
    "think": ToolNode(
        name="think",
        category="reasoning",
        input_types=("thought",),
        output_types=("reasoning_note",),
        purpose="Explicit reasoning scratchpad between major decisions.",
    ),
    "design_ligand": ToolNode(
        name="design_ligand",
        category="design",
        input_types=("constraints",),
        output_types=("ligand_candidates",),
        suggests_next=("analyze_chirality", "classify_ligand", "molecule_properties"),
        purpose="Generate ligand candidates from structured or natural-language constraints.",
    ),
    "analyze_chirality": ToolNode(
        name="analyze_chirality",
        category="design_validation",
        input_types=("smiles",),
        output_types=("chirality_report",),
        suggests_next=("classify_ligand", "evaluate"),
        purpose="Analyze point/axial chirality and stereocenters.",
    ),
    "classify_ligand": ToolNode(
        name="classify_ligand",
        category="design_validation",
        input_types=("smiles",),
        output_types=("ligand_classification",),
        suggests_next=("check_stock", "evaluate"),
        purpose="Classify ligand family, denticity, and donor atoms.",
    ),
    "design_catalyst": ToolNode(
        name="design_catalyst",
        category="design",
        input_types=("structured_catalyst_descriptor",),
        output_types=("coordination_facts",),
        suggests_next=("think", "search_literature"),
        safety_level="medium",
        purpose="Compute catalyst coordination facts without judging suitability.",
    ),
}


def available_tool_graph(tool_names: list[str]) -> dict[str, ToolNode]:
    return {name: DEFAULT_TOOL_GRAPH[name] for name in tool_names if name in DEFAULT_TOOL_GRAPH}


def summarize_tool_graph(tool_names: list[str]) -> str:
    graph = available_tool_graph(tool_names)
    if not graph:
        return "_No ToolGraph metadata available._"
    lines = [
        "| Tool | Category | Requires | Suggested next | Safety |",
        "|------|----------|----------|----------------|--------|",
    ]
    for name in sorted(graph):
        node = graph[name]
        requires = ", ".join(node.requires) if node.requires else "-"
        next_tools = ", ".join(node.suggests_next) if node.suggests_next else "-"
        lines.append(
            f"| `{node.name}` | {node.category} | {requires} | {next_tools} | {node.safety_level} |"
        )
    return "\n".join(lines)


def next_tools(tool_name: str, tool_names: list[str]) -> list[str]:
    graph = available_tool_graph(tool_names)
    node = graph.get(tool_name)
    if not node:
        return []
    available = set(graph)
    return [name for name in node.suggests_next if name in available]
