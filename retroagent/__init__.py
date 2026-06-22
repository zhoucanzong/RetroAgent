"""
RetroAgent: LLM-driven retrosynthetic planning with dedicated chemistry tools.

Built on mini-swe-agent's agent control loop and AiZynthFinder's chemistry models.
"""

__version__ = "0.1.0"

from pathlib import Path

package_dir = Path(__file__).resolve().parent
project_dir = package_dir.parent

# Models are stored flat under models/ (no subdirectories)
models_dir = project_dir / "models"

# Lazily import config
from retroagent.config import get_config, load_config, RetroAgentConfig, validate_config
