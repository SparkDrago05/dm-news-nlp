from pathlib import Path


def get_project_root() -> Path:
    """Infer project root as two levels above this file: src/data/load.py -> project/"""
    return Path(__file__).resolve().parents[2]


def _resolve_path(p: str | Path, root: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (root / p)
