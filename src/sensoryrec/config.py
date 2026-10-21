"""Load config.yaml from the repository root."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path=ROOT / "config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def repo_path(relative):
    """Turn a path from config.yaml into an absolute path."""
    return ROOT / relative
