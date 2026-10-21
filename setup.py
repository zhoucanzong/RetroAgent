"""Include the canonical Skill directory in wheels without duplicating its source."""

from pathlib import Path
import shutil
from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithSkill(build_py):
    def run(self):
        super().run()
        source = Path(__file__).parent / "skills" / "retroagent"
        target = Path(self.build_lib) / "retroagent" / "skill_assets"
        shutil.copytree(source, target, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


setup(cmdclass={"build_py": BuildWithSkill})
