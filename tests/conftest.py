import sys
from pathlib import Path

# run from a checkout without requiring the editable install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
