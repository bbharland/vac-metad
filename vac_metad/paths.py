from pathlib import Path
from joblib import Memory

# ROOT = main working directory (one up from source)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGS = ROOT / "notes/figures"
memory = Memory(DATA / "cache", verbose=1) # for @memory.cache

# This breaks the rule that importing a module should not touch disk
# FIGS.mkdir(parents=True, exist_ok=True)
