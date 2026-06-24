"""CLI for RetroAgent. Phase 1: tool testing. Phase 2: LLM-driven planning."""

import json
import logging
import time
from pathlib import Path

import typer

from retroagent import package_dir, models_dir, get_config
from retroagent.agents.planner import RetroPlanner
from retroagent.blackboard import SharedBlackboard
from retroagent.environments import RetroEnvironment
from retroagent.tools.disconnect import DisconnectionTool
from retroagent.tools.propose import ProposalTool
from retroagent.tools.evaluate import EvaluationTool
from retroagent.tools.stock import StockTool
from retroagent.tools.literature import LiteratureTool
from retroagent.tools.condition import ConditionTool
from retroagent.tools.chirality import ChiralityTool
from retroagent.tools.ligand_category import LigandCategoryTool
from retroagent.tools.conditional_ligand import ConditionalLigandTool

app = typer.Typer(rich_markup_mode="rich")


# ---------------------------------------------------------------------------
# LLM Client (OpenAI-compatible, follows OptimizerLLM pattern)
# ---------------------------------------------------------------------------

class LLMClient:
    """OpenAI-compatible LLM client with retry and tool-use support."""

    def __init__(self, model: str, api_key: str, base_url: str,
                 temperature: float = 0.3, max_tokens: int = 32000,
                 max_retries: int = 3):
        from openai import OpenAI
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def query(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        """Send a chat completion request with optional tool use.

        Returns dict with keys: role, content, extra (with actions list, cost, etc.)
        """
        for attempt in range(self.max_retries + 1):
            try:
                kwargs = dict(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=False,
                )
                if tools:
                    # Convert internal specs to OpenAI tool schema if needed
                    openai_tools = []
                    for t in tools:
                        if "function" in t:
                            openai_tools.append(t)
                        else:
                            openai_tools.append({
                                "type": "function",
                                "function": {
                                    "name": t["name"],
                                    "description": t["description"],
                                    "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                                },
                            })
                    kwargs["tools"] = openai_tools
                    kwargs["tool_choice"] = "auto"

                response = self.client.chat.completions.create(**kwargs)

                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

                msg = response.choices[0].message
                content = msg.content or ""

                # Parse tool calls from native function-calling
                raw_tool_calls = getattr(msg, "tool_calls", None) or []
                actions = []
                for tc in raw_tool_calls:
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    actions.append({
                        "tool": tc.function.name,
                        "parameters": args,
                    })

                # Fallback: parse JSON action blocks from markdown text
                # (useful for models that do not reliably emit native tool_calls)
                if not actions:
                    actions = self._parse_actions_from_text(content)

                if actions:
                    content += "\n[Tool calls: " + ", ".join(a["tool"] for a in actions) + "]"

                # Check for completion signal
                if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in content:
                    return {
                        "role": "exit",
                        "content": content,
                        "extra": {
                            "actions": actions,
                            "exit_status": "Submitted",
                            "submission": content.split("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", 1)[-1].strip(),
                            "cost": 0.0,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                        },
                    }

                return {
                    "role": "assistant",
                    "content": content,
                    "extra": {
                        "actions": actions,
                        "cost": 0.0,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                }
            except Exception as e:
                logger = logging.getLogger("retroagent")
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                if attempt < self.max_retries:
                    delay = 2 ** attempt
                    logger.info(f"Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"LLM call failed after {self.max_retries + 1} attempts: {e}")

        raise RuntimeError("LLM query failed — unreachable")

    @staticmethod
    def _parse_actions_from_text(content: str) -> list[dict]:
        """Parse tool calls from markdown JSON blocks in the LLM response.

        Supports two patterns:
        1. A single ```json block containing a list of actions:
           [{"tool": "...", "parameters": {...}}, ...]
        2. A single ```json block containing one action object:
           {"tool": "...", "parameters": {...}}
        """
        import re
        actions = []
        for block in re.findall(r"```json\s*(.*?)\s*```", content, re.DOTALL):
            try:
                data = json.loads(block)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "tool" in item:
                            actions.append({"tool": item["tool"], "parameters": item.get("parameters", {})})
                elif isinstance(data, dict) and "tool" in data:
                    actions.append({"tool": data["tool"], "parameters": data.get("parameters", {})})
            except json.JSONDecodeError:
                continue
        return actions


# ---------------------------------------------------------------------------
# Environment builder (reads from config)
# ---------------------------------------------------------------------------

def _build_environment() -> RetroEnvironment:
    """Build RetroEnvironment from config."""
    cfg = get_config()
    env = RetroEnvironment(timeout=cfg.env_timeout)

    # Disconnection tool (ONNX expansion policy)
    if cfg.expansion_model_path and cfg.expansion_model_path.exists():
        env.register("disconnect", DisconnectionTool(
            model_path=str(cfg.expansion_model_path),
            templates_path=str(cfg.templates_path) if cfg.templates_path else "",
            ringbreaker_model_path=str(cfg.ringbreaker_model_path) if cfg.ringbreaker_model_path else None,
            ringbreaker_templates_path=str(cfg.ringbreaker_templates_path) if cfg.ringbreaker_templates_path else None,
        ))

    # Proposal tool (template application)
    if cfg.templates_path and cfg.templates_path.exists():
        env.register("propose", ProposalTool(
            templates_path=str(cfg.templates_path),
            filter_model_path=str(cfg.filter_model_path) if cfg.filter_model_path else None,
        ))

    # Evaluation tool (feasibility + stock scoring)
    if cfg.filter_model_path and cfg.filter_model_path.exists():
        env.register("evaluate", EvaluationTool(
            filter_model_path=str(cfg.filter_model_path),
            stock_path=str(cfg.stock_path) if cfg.stock_path else None,
        ))

    # Stock tool
    if cfg.stock_path and cfg.stock_path.exists():
        env.register("check_stock", StockTool(str(cfg.stock_path)))

    # Literature tool
    if cfg.templates_path and cfg.templates_path.exists():
        env.register("search_literature", LiteratureTool(templates_path=str(cfg.templates_path)))

    # Condition tool (always available, rule-based)
    env.register("recommend_conditions", ConditionTool())

    # New: chiral ligand design tools (always available, lightweight)
    env.register("analyze_chirality", ChiralityTool())
    env.register("classify_ligand", LigandCategoryTool())
    # Conditional ligand design tool with optional CIC-DB few-shot examples
    design_tool = ConditionalLigandTool(
        examples_path=str(cfg.cic_db_conditional_examples_path)
        if cfg.cic_db_conditional_examples_path and cfg.cic_db_conditional_examples_path.exists()
        else None,
        n_examples=3,
    )
    env.register("design_ligand", design_tool)

    return env


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

@app.command()
def run(
    task: str = typer.Argument(..., help="Target molecule SMILES or design constraints"),
    mode: str = typer.Option("retrosynthesis", "--mode", help="Mode: retrosynthesis | design"),
    model: str | None = typer.Option(None, "-m", "--model", help="LLM model name (overrides config)"),
    api_key: str | None = typer.Option(None, "--api-key", help="API key (overrides config)"),
    base_url: str | None = typer.Option(None, "--base-url", help="API base URL (overrides config)"),
    output: Path | None = typer.Option(None, "-o", "--output", help="Save trajectory to file"),
    max_steps: int = typer.Option(30, "--max-steps", help="Maximum planning steps"),
) -> None:
    """Run LLM-driven planning or design on a target SMILES / constraints."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

    cfg = get_config()

    # Resolve LLM settings (CLI > env > config > defaults)
    llm_model = model or cfg.llm_model
    llm_api_key = api_key or cfg.llm_api_key
    llm_base_url = base_url or cfg.llm_base_url

    if not llm_api_key:
        typer.echo(
            "ERROR: No API key set.\n"
            "  - Create retroagent/config/config.local.yaml with:\n"
            "      llm:\n"
            "        api_key: \"sk-...\"\n"
            "  - Or set LLM_API_KEY environment variable\n"
            "  - Or pass --api-key", err=True
        )
        raise typer.Exit(1)

    # Build components
    env = _build_environment()
    llm = LLMClient(
        model=llm_model, api_key=llm_api_key, base_url=llm_base_url,
        temperature=cfg.llm_temperature, max_tokens=cfg.llm_max_tokens,
        max_retries=cfg.llm_max_retries,
    )
    blackboard = SharedBlackboard()

    planner = RetroPlanner(
        llm, env, blackboard,
        system_template=cfg.agent_system_template,
        instance_template=cfg.agent_instance_template,
        step_limit=max_steps,
        cost_limit=cfg.agent_cost_limit,
        wall_time_limit_seconds=cfg.agent_wall_time_limit_seconds,
        max_search_depth=cfg.agent_max_search_depth,
        search_strategy=cfg.agent_search_strategy,
        max_consecutive_format_errors=cfg.agent_max_consecutive_format_errors,
        output_path=output,
        enable_reflection=cfg.agent_enable_reflection,
        enable_backtracking=cfg.agent_enable_backtracking,
        backtrack_score_threshold=cfg.agent_backtrack_score_threshold,
        backtrack_patience=cfg.agent_backtrack_patience,
        enable_repeated_action_guard=cfg.agent_enable_repeated_action_guard,
        max_repeated_actions=cfg.agent_max_repeated_actions,
        enable_schema_validation=cfg.agent_enable_schema_validation,
    )

    print(f"\n{'='*60}")
    print(f"RetroAgent v0.1 — Mode: {mode}")
    print(f"Task: {task}")
    print(f"Model: {llm_model}")
    print(f"Tools available: {[t['name'] for t in env.get_tools_spec()]}")
    print(f"{'='*60}\n")

    result = planner.run(task=task, mode=mode)
    print(f"\nResult: exit_status={result.get('exit_status')}")

    if output:
        print(f"Trajectory saved to {output}")


@app.command()
def test_tools(
    smiles: str = typer.Argument("CC(=O)Oc1ccccc1C(=O)O", help="Target SMILES"),
) -> None:
    """Test all tools against a target molecule without LLM."""
    env = _build_environment()

    print(f"\nTesting tools on: {smiles}\n")

    # 1. Disconnect
    print("--- DisconnectionTool ---")
    r = json.loads(env.execute({"tool": "disconnect", "parameters": {"smiles": smiles}})["output"])
    fg_info = {}
    if "molecule_info" in r:
        fg_info = r["molecule_info"]
        print(f"  Molecule: {fg_info.get('num_atoms')} atoms, {fg_info.get('rings')} rings")
        print(f"  Functional groups present: {fg_info.get('functional_groups')}")
        print(f"  Functional groups absent:  {fg_info.get('functional_groups_absent')}")
        print()
    if "bonds" in r:
        matching_count = sum(1 for b in r["bonds"] if b.get("matching"))
        print(f"  {len(r['bonds'])} templates, {matching_count} substructure-matched:")
        for b in r["bonds"][:5]:
            match_flag = "✓" if b.get("matching") else "✗"
            print(f"    {match_flag} Rank {b['rank']}: score={b['score']:.4f} class={b['classification']}")
    elif "error" in r:
        print(f"  Error: {r['error']}")

    # 2. Literature
    print("--- LiteratureTool ---")
    r = json.loads(env.execute({"tool": "search_literature", "parameters": {"smiles": smiles}})["output"])
    print(f"  Known routes: {len(r.get('known_routes', []))}")

    # 3. Propose — let LLM decide fallback strategy
    print("--- ProposalTool ---")
    dis_r = json.loads(env.execute({"tool": "disconnect", "parameters": {"smiles": smiles}})["output"])
    reactions = []
    if dis_r.get("bonds"):
        matching_tids = [b["template_index"] for b in dis_r["bonds"] if b.get("matching")]
        nonmatching_tids = [b["template_index"] for b in dis_r["bonds"] if not b.get("matching")]
        print(f"  Model predictions: {len(matching_tids)} matching, {len(nonmatching_tids)} non-matching")

        if nonmatching_tids:
            print(f"  Model predictions mostly non-matching → testing LLM's decision point:")
            print(f"    The LLM should notice: molecule has no amide, model predicts 'N-acylation'")
            print(f"    → LLM should call propose() with use_fallback=True")

        # Test with model-only first
        r = json.loads(env.execute({"tool": "propose", "parameters": {
            "smiles": smiles, "template_indices": [b["template_index"] for b in dis_r["bonds"][:10]],
            "use_fallback": False, "max_results": 20
        }})["output"])
        print(f"  propose(use_fallback=False): {r.get('count', 0)} reactions")

        # Test with fallback — simulates LLM deciding to fall back
        r = json.loads(env.execute({"tool": "propose", "parameters": {
            "smiles": smiles, "template_indices": [b["template_index"] for b in dis_r["bonds"][:10]],
            "use_fallback": True, "max_results": 20
        }})["output"])
        reactions = r.get("reactions", [])
        print(f"  propose(use_fallback=True):  {r.get('count', 0)} reactions  ← LLM chooses")
        for rx in reactions[:3]:
            print(f"    [{rx['classification']} occ={rx['library_occurrence']}] {rx['precursors']}")

    # 4. Evaluate
    print("--- EvaluationTool ---")
    if reactions:
        r = json.loads(env.execute({"tool": "evaluate", "parameters": {"reactions": reactions[:5]}})["output"])
        scores = r.get("scores", {})
        for rid, s in list(scores.items())[:3]:
            print(f"    {rid}: feasibility={s['feasibility']:.4f} stock={s['stock_availability']:.4f} total={s['total']:.4f}")

    # 5. Stock check
    print("--- StockTool ---")
    all_precursors = []
    for rx in reactions[:3]:
        all_precursors.extend(rx.get("precursors", []))
    if all_precursors:
        r = json.loads(env.execute({"tool": "check_stock", "parameters": {"smiles_list": all_precursors[:10]}})["output"])
        print(f"  In stock: {len(r.get('in_stock', {}))}/{r.get('total_checked', 0)}")
        for smi in r.get("in_stock", {}):
            print(f"    {smi}: {r['in_stock'][smi]}")


if __name__ == "__main__":
    app()
