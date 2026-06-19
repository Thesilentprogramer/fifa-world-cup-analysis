"""Runtime checks for project environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REEXEC_ENV = "FIFA_PROJECT_VENV_REEXEC"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _venv_python(project_root: Path) -> Path:
    return project_root / ".venv" / "bin" / "python"


def _running_in_project_venv(project_root: Path) -> bool:
    venv_dir = project_root / ".venv"
    if not venv_dir.exists():
        return False
    try:
        return Path(sys.prefix).resolve() == venv_dir.resolve()
    except OSError:
        return str(sys.executable).startswith(str(venv_dir.resolve() / "bin"))


def _missing_deps() -> list[str]:
    missing: list[str] = []
    for pkg, pip_name in (("pyarrow", "pyarrow"), ("xgboost", "xgboost"), ("sklearn", "scikit-learn")):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pip_name)
    return missing


def ensure_project_deps() -> None:
    """Use project venv automatically, or exit with setup instructions."""
    project_root = _project_root()
    venv_python = _venv_python(project_root)
    missing = _missing_deps()

    if not missing:
        return

    if (
        venv_python.exists()
        and not _running_in_project_venv(project_root)
        and os.environ.get(_REEXEC_ENV) != "1"
    ):
        os.environ[_REEXEC_ENV] = "1"
        os.execv(str(venv_python), [str(venv_python), *sys.argv])

    lines = [
        f"Wrong Python interpreter: {sys.executable}",
        f"Missing here: {', '.join(missing)}",
    ]
    if venv_python.exists():
        venv_cmd = project_root / ".venv" / "bin" / "python"
        lines.extend([
            "",
            "Dependencies are installed in the project venv. Use either:",
            "",
            f"  source {project_root}/.venv/bin/activate",
            "  python scripts/build_features.py",
            "  python scripts/train_model.py",
            "",
            "Or without activating:",
            f"  {venv_cmd} scripts/build_features.py",
            f"  {venv_cmd} scripts/train_model.py",
        ])
    else:
        lines.extend([
            "",
            f"  cd {project_root}",
            "  python3.11 -m venv .venv",
            "  source .venv/bin/activate",
            "  pip install -r requirements.txt",
        ])

    print("\n".join(lines), file=sys.stderr)
    sys.exit(1)
