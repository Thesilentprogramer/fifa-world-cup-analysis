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


def ensure_model_compatibility() -> None:
    """Ensure that the pickled models are compatible with the current environment.
    
    If there is a mismatch (e.g., different OS, python version, or package versions),
    the models are automatically retrained to prevent segmentation faults during loading.
    """
    project_root = _project_root()
    models_dir = project_root / "models"
    signature_path = models_dir / "model_env_signature.json"
    
    try:
        import joblib
        import sklearn
        import xgboost
    except ImportError:
        # If packages are not installed, we can't retrain or run; env_check will handle
        return

    current_sig = {
        "platform": sys.platform,
        "xgboost_version": xgboost.__version__,
        "scikit_learn_version": sklearn.__version__,
        "joblib_version": joblib.__version__
    }
    
    retrain = True
    if signature_path.exists():
        try:
            import json
            saved_sig = json.loads(signature_path.read_text(encoding="utf-8"))
            if (
                saved_sig.get("platform") == current_sig["platform"]
                and saved_sig.get("xgboost_version") == current_sig["xgboost_version"]
                and saved_sig.get("scikit_learn_version") == current_sig["scikit_learn_version"]
                and saved_sig.get("joblib_version") == current_sig["joblib_version"]
            ):
                retrain = False
        except Exception:
            pass
            
    if retrain:
        print("Model environment mismatch or missing signature. Retraining models for current environment...")
        try:
            # Run the training scripts main functions
            from scripts.train_model import main as train_outcome_main
            from scripts.train_xg_model import main as train_xg_main
            
            train_outcome_main()
            try:
                train_xg_main()
            except SystemExit as se:
                if se.code != 0:
                    print(f"Warning: train_xg_model exited with code {se.code}", file=sys.stderr)
            
            import json
            signature_path.write_text(json.dumps(current_sig, indent=2), encoding="utf-8")
            print("Model retraining completed and signature updated.")
        except Exception as e:
            print(f"Warning: Automatic model retraining failed: {e}", file=sys.stderr)

