"""Configuration loader for RetroAgent.

Loads config/default.yaml (version-controlled defaults), then merges
config.local.yaml (local overrides, gitignored for API keys), then env var overrides.

Priority (lowest → highest):
  1. default.yaml (in config/ directory)
  2. config.local.yaml
  3. Environment variables (LLM_API_KEY, LLM_BASE_URL, LLM_MODEL)

Usage:
  from retroagent.config import load_config
  cfg = load_config()
  cfg.llm_model          # "deepseek-v4-flash"
  cfg.expansion_model    # "uspto_model.onnx"
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_CONFIG_DIR = Path(__file__).resolve().parent
_DEFAULT_CONFIG = _CONFIG_DIR / "default.yaml"
_LOCAL_CONFIG = _CONFIG_DIR / "config.local.yaml"

# Project root is retroagent package dir's parent
PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_DIR / "models"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, returning {} if not found or unreadable."""
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError):
        return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override keys win."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _resolve_model_path(filename: str | None) -> Path | None:
    """Resolve a model filename to an absolute path under models_dir."""
    if filename is None:
        return None
    return MODELS_DIR / filename


@dataclass
class RetroAgentConfig:
    """Merged configuration for RetroAgent.

    Holds model paths, LLM settings, agent settings, and environment settings.
    """

    # --- Model paths (absolute) ---
    expansion_model_path: Path | None = None
    filter_model_path: Path | None = None
    ringbreaker_model_path: Path | None = None
    templates_path: Path | None = None
    templates_fallback_path: Path | None = None
    stock_path: Path | None = None

    # --- LLM ---
    llm_model: str = "deepseek-v4-flash"
    llm_base_url: str = "https://api.deepseek.com"
    llm_api_key: str = ""
    llm_temperature: float = 0.3
    llm_max_tokens: int = 32000
    llm_max_retries: int = 3

    # --- Agent ---
    agent_system_template: str = ""
    agent_instance_template: str = "Plan a synthetic route for the following molecule: {{task}}"
    agent_step_limit: int = 30
    agent_cost_limit: float = 10.0
    agent_wall_time_limit_seconds: int = 600
    agent_max_search_depth: int = 20
    agent_search_strategy: str = "auto"
    agent_max_consecutive_format_errors: int = 3

    # --- CIC-DB / ligand design paths ---
    cic_db_conditional_examples_path: Path | None = None
    # Ring-breaker templates (CSV)
    ringbreaker_templates_path: Path | None = None

    # --- Inner Loop Engineering ---
    agent_enable_reflection: bool = True
    agent_enable_backtracking: bool = True
    agent_backtrack_score_threshold: float = 0.3
    agent_backtrack_patience: int = 2
    agent_enable_repeated_action_guard: bool = True
    agent_max_repeated_actions: int = 2
    agent_enable_schema_validation: bool = True

    # --- Environment ---
    env_timeout: int = 120

    # --- Paths ---
    project_dir: Path = field(default_factory=lambda: PROJECT_DIR)
    models_dir: Path = field(default_factory=lambda: MODELS_DIR)


def load_config() -> RetroAgentConfig:
    """Load merged config: default.yaml + config.local.yaml + env overrides."""
    cfg = _load_yaml(_DEFAULT_CONFIG)
    local = _load_yaml(_LOCAL_CONFIG)
    cfg = _deep_merge(cfg, local)

    # Env var overrides (highest priority)
    llm_section = cfg.get("llm", {})
    env_key = os.environ.get("RETROAGENT_API_KEY", "") or os.environ.get("LLM_API_KEY", "")
    if env_key:
        llm_section["api_key"] = env_key

    env_url = os.environ.get("LLM_BASE_URL", "")
    if env_url:
        llm_section["base_url"] = env_url

    env_model = os.environ.get("LLM_MODEL", "")
    if env_model:
        llm_section["model"] = env_model

    # Model paths
    models_section = cfg.get("models", {})
    agent_section = cfg.get("agent", {})
    env_section = cfg.get("environment", {})

    return RetroAgentConfig(
        # Model paths
        expansion_model_path=_resolve_model_path(models_section.get("expansion_model")),
        filter_model_path=_resolve_model_path(models_section.get("filter_model")),
        ringbreaker_model_path=_resolve_model_path(models_section.get("ringbreaker_model")),
        templates_path=_resolve_model_path(models_section.get("templates")),
        templates_fallback_path=_resolve_model_path(models_section.get("templates_fallback")),
        stock_path=_resolve_model_path(models_section.get("stock")),
        # CIC-DB / ringbreaker templates
        cic_db_conditional_examples_path=_resolve_model_path(models_section.get("cic_db_conditional_examples")),
        ringbreaker_templates_path=_resolve_model_path(models_section.get("ringbreaker_templates")),
        # LLM
        llm_model=llm_section.get("model", "deepseek-v4-flash"),
        llm_base_url=llm_section.get("base_url", "https://api.deepseek.com"),
        llm_api_key=llm_section.get("api_key", ""),
        llm_temperature=llm_section.get("temperature", 0.3),
        llm_max_tokens=llm_section.get("max_tokens", 32000),
        llm_max_retries=llm_section.get("max_retries", 3),
        # Agent
        agent_system_template=agent_section.get("system_template", ""),
        agent_instance_template=agent_section.get(
            "instance_template", "Plan a synthetic route for the following molecule: {{task}}"
        ),
        agent_step_limit=agent_section.get("step_limit", 30),
        agent_cost_limit=agent_section.get("cost_limit", 10.0),
        agent_wall_time_limit_seconds=agent_section.get("wall_time_limit_seconds", 600),
        agent_max_search_depth=agent_section.get("max_search_depth", 20),
        agent_search_strategy=agent_section.get("search_strategy", "auto"),
        agent_max_consecutive_format_errors=agent_section.get("max_consecutive_format_errors", 3),
        # Inner Loop Engineering
        agent_enable_reflection=agent_section.get("enable_reflection", True),
        agent_enable_backtracking=agent_section.get("enable_backtracking", True),
        agent_backtrack_score_threshold=agent_section.get("backtrack_score_threshold", 0.3),
        agent_backtrack_patience=agent_section.get("backtrack_patience", 2),
        agent_enable_repeated_action_guard=agent_section.get("enable_repeated_action_guard", True),
        agent_max_repeated_actions=agent_section.get("max_repeated_actions", 2),
        agent_enable_schema_validation=agent_section.get("enable_schema_validation", True),
        # Environment
        env_timeout=env_section.get("timeout", 120),
    )


# Global cached config
_config: RetroAgentConfig | None = None


def get_config() -> RetroAgentConfig:
    """Get the cached config, loading if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def validate_config() -> None:
    """Validate that required settings are present."""
    cfg = get_config()
    if not cfg.llm_api_key:
        raise RuntimeError(
            "LLM API key is not set.\n"
            "Options:\n"
            "  1. Create retroagent/config/config.local.yaml with:\n"
            "       llm:\n"
            "         api_key: \"sk-...\"\n"
            "  2. Or set environment variable:\n"
            "       export LLM_API_KEY='sk-...'\n"
            "       export RETROAGENT_API_KEY='sk-...'\n"
            "Also set LLM_BASE_URL if not using the default provider."
        )
    missing_models = []
    for name, path in [
        ("expansion_model", cfg.expansion_model_path),
        ("filter_model", cfg.filter_model_path),
        ("templates", cfg.templates_path),
        ("stock", cfg.stock_path),
    ]:
        if path is None or not path.exists():
            missing_models.append(name)
    if missing_models:
        raise RuntimeError(
            f"Missing model files: {', '.join(missing_models)}\n"
            f"Ensure the files exist in {cfg.models_dir}"
        )
