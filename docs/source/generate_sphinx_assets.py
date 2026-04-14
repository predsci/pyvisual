import os
import subprocess
import sys
from pathlib import Path

def _generate_sphinx_assets():
    from dotenv import load_dotenv
    load_dotenv()

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _EXAMPLES_DIR = _REPO_ROOT / "examples"
    _INCLUDED_EXAMPLES = [
        _EXAMPLES_DIR / "99_advanced_plots" / "p07_faux_volume_render.py",
        # _EXAMPLES_DIR / "99_advanced_plots" / "p09_integrating_mhdweb.py",
    ]

    env = os.environ.copy()
    static_assets = os.environ.get("STATIC_ASSETS")
    if not Path(static_assets).exists():
        Path(static_assets).mkdir(parents=True, exist_ok=True)

    for example in _INCLUDED_EXAMPLES:
        print(f"Running {example.name} ...")
        subprocess.run(
            [sys.executable, str(example)],
            env=env,
            check=True,
        )


if __name__ == "__main__":
    _generate_sphinx_assets()