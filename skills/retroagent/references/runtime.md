# Runtime and Transport

Install the backend from this release's repository `skill` branch in an isolated
environment, then export this Skill into the host's supported skill location:

```sh
git clone --branch skill https://github.com/zhoucanzong/RetroAgent.git
cd RetroAgent
python -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/retroagent capability skill --output /path/to/host-skills/retroagent
```

The host performs the reasoning. Do not configure a second model API merely to
use this Skill. `agent run/resume` remain available for an explicit request to use
the standalone legacy Agent.

The bundled helper `scripts/retro.py` forwards commands to the installed backend.
Set `RETROAGENT_PYTHON` to the absolute path of the backend's Python executable if
the host runs a different environment. No model checkpoint is bundled.
Inspect `tool list --json` to discover the currently configured tools. Tool schema
discovery does not initialize neural or ONNX weights; calls may still fail if an
optional dependency or checkpoint is missing. Use absolute checkpoint paths.

## Persistent MCP

Run `retroagent mcp serve --project /absolute/project/path` from the backend's
environment. Configure that process through the host's MCP configuration.

- `retroagent_tools`: list chemistry tool schemas without loading model weights.
- `retroagent_tool`: execute a chemistry tool in the persistent process. Pass
  `tool`, `parameters`, `project`, and optional `target_ids`, `question`, `rationale`,
  `request_id`. The model cache is retained across calls.
- `retroagent_command`: run project/state commands such as `research context`,
  `batch`, `view`, `approve`, and `research finish`. This compatibility transport
  starts a CLI process; use `retroagent_tool` for expensive model calls.

The MCP server processes requests serially and releases loaded models at EOF.
CLI commands are separate processes and do not share a neural cache. Do not run
multiple legacy controllers against one project concurrently with host-driven work.

## Optional Neural Runtime

The repository's `requirements-retrochimera-cpu.txt` and
`docs/retrochimera-cpu.md` describe a verified Python 3.11 environment. Install
RetroChimera and weights only when a task needs them. Follow upstream checkpoint
instructions; any mirror must retain its provenance and integrity checks.
Never put weights, secrets, or run transcripts inside the installed Skill folder.

Tool execution errors produce nonzero CLI status or MCP `isError=true`, with the
failed execution retained in the ledger. Read diagnostics rather than silently
substituting invented outputs. Approval and runtime failures are different:
installing a missing model does not grant permission for external actions.
