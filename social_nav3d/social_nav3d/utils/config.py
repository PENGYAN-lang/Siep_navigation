from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        cfg: Any = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError('YAML root must be a mapping')
    return cfg
