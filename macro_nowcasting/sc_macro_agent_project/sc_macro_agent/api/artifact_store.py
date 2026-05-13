"""
产物存储工具。

作用：
- 统一 run 目录命名
- 把 dataframe / json / text 按规范落盘
- 给 Agent / CLI / API 共用
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ..utils import save_json, save_text


class ArtifactStore:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self, prefix: str = "run") -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.base_dir / f"{prefix}_{ts}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_dataframe(self, run_dir: str | Path, name: str, df: pd.DataFrame) -> str:
        path = Path(run_dir) / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return str(path)

    def save_payload(self, run_dir: str | Path, name: str, payload: Dict[str, Any]) -> str:
        path = Path(run_dir) / f"{name}.json"
        save_json(path, payload)
        return str(path)

    def save_markdown(self, run_dir: str | Path, name: str, content: str) -> str:
        path = Path(run_dir) / f"{name}.md"
        save_text(path, content)
        return str(path)

    def save_text(self, run_dir: str | Path, name: str, content: str) -> str:
        path = Path(run_dir) / f"{name}.txt"
        save_text(path, content)
        return str(path)
