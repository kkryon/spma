from __future__ import annotations

import sys
from pathlib import Path


def _add_local_venv_site_packages() -> None:
    repo_root = Path(__file__).resolve().parent
    version_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = repo_root / ".venv" / "lib" / version_tag / "site-packages"
    if site_packages.exists():
        sys.path.insert(0, str(site_packages))


_add_local_venv_site_packages()

