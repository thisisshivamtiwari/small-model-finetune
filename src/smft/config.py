import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def model_path(cfg: dict, key: str) -> Path:
    """Local MLX model folder. The 1.5B model is shared with agent-observatory, so it may live elsewhere."""
    name = cfg["models"][key]
    for base in [os.environ.get("MODEL_DIR", ""), "~/.cache/smft", "~/.cache/agent-observatory"]:
        if base:
            for candidate in (Path(base).expanduser() / name, Path(base).expanduser() / name.lower()):
                if candidate.exists():
                    return candidate
    raise FileNotFoundError(f"model {name} not found; set MODEL_DIR or download it (see README)")


def k_name(k: int | None) -> str:
    return "all" if k is None else f"k{k}"
