"""Forward one command to a configured RetroAgent backend without a second Agent."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys


def main():
    python = os.environ.get("RETROAGENT_PYTHON")
    executable = shutil.which("retroagent")
    if python:
        command = [python, "-m", "retroagent.run.retro"]
    elif executable:
        command = [executable]
    elif importlib.util.find_spec("retroagent"):
        command = [sys.executable, "-m", "retroagent.run.retro"]
    else:
        print(json.dumps({"ok": False, "error": {"code": "E_RUNTIME_MISSING",
              "message": "Install the RetroAgent backend or set RETROAGENT_PYTHON to its interpreter."}}))
        return 2
    try:
        return subprocess.call([*command, *sys.argv[1:]])
    except OSError as exc:
        print(json.dumps({"ok": False, "error": {"code": "E_RUNTIME_MISSING", "message": str(exc)}}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
