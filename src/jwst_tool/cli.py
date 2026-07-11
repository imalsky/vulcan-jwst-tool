"""Console entry point: ``jwst-tool`` launches the Streamlit GUI.

Equivalent to ``streamlit run src/jwst_tool/app.py``, but works
from anywhere once the package is installed. Preflight checks catch the two
external requirements with actionable messages instead of a mid-run stack
trace: an importable ``vulcan_jax`` package and the Pandeia backend env
(``$JWST_TOOL_PANDEIA_PYTHON``).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("jwst-tool: streamlit is not installed in this environment.\n"
              "Install the GUI extra:  pip install 'vulcan-jwst-tool[gui]'  "
              "(or, in a checkout:  pip install -e '.[gui]' --no-deps "
              "plus pip install streamlit pandas)", file=sys.stderr)
        return 2

    import importlib.util
    if importlib.util.find_spec("vulcan_jax") is None:
        print("jwst-tool: the vulcan_jax package is not installed in this "
              "environment.\n"
              "Install it from TestPyPI "
              "(pip install -i https://test.pypi.org/simple/ vulcan-jax), or "
              "from a checkout:\n"
              "  pip install -e <PROJECT_ROOT>/VULCAN-JAX --no-deps",
              file=sys.stderr)
        return 2

    from jwst_tool import instruments as ins
    if not Path(ins.PICASO_PYTHON).exists():
        print(f"jwst-tool: Pandeia backend python not found at {ins.PICASO_PYTHON}.\n"
              "Point JWST_TOOL_PANDEIA_PYTHON at a python with pandeia.engine 3.0 "
              "(and JWST_TOOL_PANDEIA_REFDATA at the matching refdata). The GUI "
              "still starts, but every noise calculation will refuse to run.",
              file=sys.stderr)

    app = Path(__file__).parent / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app)] + sys.argv[1:]
    return subprocess.call(cmd, env=os.environ.copy())


if __name__ == "__main__":
    sys.exit(main())
