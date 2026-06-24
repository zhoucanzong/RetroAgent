"""PlannerConfig and base agent configuration for RetroAgent."""

from pathlib import Path
from pydantic import BaseModel


class PlannerConfig(BaseModel):
    """Configuration for RetroPlanner. Mirrors mini-swe-agent's AgentConfig."""

    system_template: str = ""
    """Template for the system message (first message). Injected with tool specs + blackboard state."""

    instance_template: str = "Plan a synthetic route for the following molecule: {{task}}"
    """Template for the first user message specifying the target."""

    step_limit: int = 100
    """Maximum number of Planner steps (LLM calls + tool executions)."""

    cost_limit: float = 5.0
    """Stop agent after exceeding this cost (USD)."""

    wall_time_limit_seconds: int = 600
    """Stop agent after this many seconds. 0 means no limit."""

    max_consecutive_format_errors: int = 3
    """Exit after this many format errors in a row."""

    output_path: Path | None = None
    """Save the trajectory to this path."""

    max_search_depth: int = 20
    """Maximum number of retrosynthetic search iterations."""

    search_strategy: str = "auto"
    """Search strategy hint for the Planner. 'explore', 'exploit', or 'auto'."""

    # --- Inner Loop Engineering settings ---
    enable_reflection: bool = True
    """After each tool execution, ask the LLM to reflect on the result."""

    enable_backtracking: bool = True
    """Detect low scores / dead-ends and force the LLM to backtrack."""

    backtrack_score_threshold: float = 0.3
    """If best route score stays below this for N rounds, trigger backtracking."""

    backtrack_patience: int = 2
    """Number of consecutive low-score rounds before backtracking."""

    enable_repeated_action_guard: bool = True
    """Detect repeated identical tool calls and terminate early."""

    max_repeated_actions: int = 2
    """Allow the same tool+parameters combination up to this many times."""

    enable_schema_validation: bool = True
    """Validate LLM tool parameters against declared JSON schemas before execution."""

    # --- Dead-Loop Monitor (Phase 0) ---
    enable_dead_loop_monitor: bool = True
    """Enable detection of tool-type cycling, semantic repeats, and stagnation."""

    cycling_window: int = 6
    """Sliding window size for tool-type cycling detection."""

    stagnation_rounds: int = 5
    """Rounds without substantive progress before stagnation warning."""

    semantic_repeat_threshold: int = 3
    """Max same-(tool, semantic_key) calls before warning."""

    enable_early_exit_hint: bool = True
    """Inject a hint when route quality is good enough to submit."""

    early_exit_score_threshold: float = 0.7
    """Score threshold for early exit hint."""

    # --- Enhanced Observation (Phase 2) ---
    enable_enhanced_observation: bool = True
    """Replace simple reflection with structured observation+reasoning prompts."""

    # --- Auto-tune (Phase 3) ---
    enable_auto_tune: bool = True
    """Auto-adjust step_limit and other params based on molecular complexity."""

    # --- Design Auditor (Phase 0F) ---
    enable_design_auditor: bool = True
    """In design mode, run an adversarial chemical correctness review of catalyst/ligand designs."""
