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
        suggests_next=("propose", "ensemble_disconnections", "think"),
        fallback_tools=("functional_groups",),
        purpose="Generate ranked retrosynthetic template/disconnection candidates.",
    ),
    "propose": ToolNode(
        name="propose",
        category="retrosynthesis",
        input_types=("smiles", "template_indices"),
        output_types=("reaction_records",),
        requires=("disconnect",),
        suggests_next=("ensemble_disconnections", "evaluate", "check_stock", "score_routes", "map_reaction"),
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
        suggests_next=("chemical_critic", "build_route_portfolio", "design_experiments", "think"),
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
        suggests_next=("think", "design_experiments", "select_experiment", "build_research_report"),
        fallback_tools=("molecule_properties", "functional_groups", "evaluate", "check_stock"),
        purpose="Audit route hypotheses and route graphs for evidence, validation, precursor, mechanism, and strategy risks.",
    ),
    "design_experiments": ToolNode(
        name="design_experiments",
        category="experiment",
        input_types=("target_smiles", "hypotheses", "critic_report", "reaction_records", "route_graph"),
        output_types=("experiment_plan", "controls", "success_criteria", "failure_diagnostics"),
        requires=("chemical_critic",),
        suggests_next=("select_experiment", "build_research_report", "think"),
        fallback_tools=("recommend_conditions", "map_reaction", "search_literature"),
        safety_level="medium",
        purpose="Convert critic findings into prioritized validation experiments with controls and decision points.",
    ),
    "select_experiment": ToolNode(
        name="select_experiment",
        category="active_learning",
        input_types=("hypotheses", "candidate_experiments", "outcome_likelihoods", "costs", "risks"),
        output_types=("ranked_experiments", "expected_information_gain", "selected_experiment"),
        requires=("design_experiments",),
        suggests_next=("update_hypotheses",),
        fallback_tools=("think", "chemical_critic"),
        safety_level="medium",
        purpose="Choose the next discriminating experiment by expected information gain adjusted for cost and risk.",
    ),
    "update_hypotheses": ToolNode(
        name="update_hypotheses",
        category="active_learning",
        input_types=("hypotheses", "observation", "likelihoods"),
        output_types=("posterior_probabilities", "entropy_reduction"),
        requires=("select_experiment",),
        suggests_next=("build_route_portfolio", "design_experiments", "think"),
        purpose="Apply an observed result to the competing hypotheses with an auditable Bayesian update.",
    ),
    "neuralloc_predict": ToolNode(
        name="neuralloc_predict", category="retrosynthesis",
        input_types=("target_smiles", "checkpoint"),
        output_types=("reaction_records", "ensemble_source"),
        suggests_next=("transformer_predict", "ensemble_disconnections", "evaluate"),
        fallback_tools=("disconnect", "propose"),
        purpose="Optional NeuralLoc-only template and localization predictions, chosen by the Agent.",
    ),
    "transformer_predict": ToolNode(
        name="transformer_predict", category="retrosynthesis",
        input_types=("target_smiles", "checkpoint"),
        output_types=("reaction_records", "ensemble_source"),
        suggests_next=("neuralloc_predict", "ensemble_disconnections", "evaluate"),
        fallback_tools=("disconnect", "propose"),
        purpose="Optional R-SMILES 2 de-novo predictions, chosen by the Agent.",
    ),
    "retrochimera_predict": ToolNode(
        name="retrochimera_predict",
        category="retrosynthesis",
        input_types=("target_smiles", "retrochimera_checkpoint"),
        output_types=("ranked_reaction_records", "constituent_ranks", "ensemble_probabilities"),
        suggests_next=("ensemble_disconnections", "build_route_portfolio", "validate_route_portfolio"),
        fallback_tools=("disconnect", "propose"),
        safety_level="medium",
        purpose="Call an optional RetroChimera checkpoint while preserving constituent-model provenance.",
    ),
    "ensemble_disconnections": ToolNode(
        name="ensemble_disconnections",
        category="route_discovery",
        input_types=("ranked_prediction_sources", "source_rank_weights"),
        output_types=("consensus_reactions", "individual_ranks", "agreement_scores"),
        suggests_next=("build_route_portfolio", "evaluate", "map_reaction"),
        fallback_tools=("chemical_critic", "think"),
        purpose="Fuse complementary retrosynthesis sources by canonical reactant-set agreement and rank weights.",
    ),
    "build_route_portfolio": ToolNode(
        name="build_route_portfolio",
        category="route_discovery",
        input_types=("target_smiles", "multi_source_route_candidates", "known_reactions", "ensemble_provenance"),
        output_types=("deduplicated_routes", "novelty_scores", "model_disagreement"),
        suggests_next=("validate_route_portfolio",),
        fallback_tools=("score_routes", "disconnect", "think"),
        purpose="Combine template, template-free, mechanism, and analogy routes while preserving provenance and disagreement.",
    ),
    "validate_route_portfolio": ToolNode(
        name="validate_route_portfolio",
        category="independent_validation",
        input_types=("routes", "mapping_reports", "forward_predictions", "conditions", "stock_hits"),
        output_types=("validation_reports", "validated_routes"),
        requires=("build_route_portfolio",),
        suggests_next=("build_research_report", "research_memory", "think"),
        fallback_tools=("map_reaction", "recommend_conditions", "check_stock", "chemical_critic"),
        safety_level="medium",
        purpose="Gate generated routes using evidence checks independent of generator scores.",
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
