from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._bootstrap import bootstrap_local_venv


bootstrap_local_venv()

from spma.tune_sparse import main


if __name__ == "__main__":
    main()
