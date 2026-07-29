"""StrictDict + load/registry for LLM prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class StrictDict(dict):
    """str.format_map 专用：缺少键时抛出带键名的 KeyError。"""

    def __getitem__(self, key: str) -> Any:
        if key not in self:
            raise KeyError(f"Prompt template variable not provided: '{key}'")
        return super().__getitem__(key)


def _prompts_dir() -> Path:
    return Path(__file__).parent


def load_prompt(prompt_id: str, version: str | None = None) -> Dict[str, Any]:
    """从 YAML 加载提示词。version 指定时从 archive/ 读历史版本，否则读当前版本。"""
    import yaml

    if version is not None:
        path = _prompts_dir() / "archive" / f"{prompt_id}_{version}.yaml"
    else:
        path = _prompts_dir() / f"{prompt_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def render(prompt_id: str, version: str | None = None, **kwargs: Any) -> Dict[str, Any]:
    """加载并渲染提示词，返回 {id, version, system, user, temperature, max_tokens}。

    严格模式：缺变量抛 KeyError 并指明变量名。
    version 指定时从 archive/ 读历史版本。
    """
    prompt = load_prompt(prompt_id, version=version)
    sd = StrictDict(kwargs)
    return {
        "id": prompt_id,
        "version": prompt.get("version", "0.0.0"),
        "system": prompt["system"].format_map(sd),
        "user": prompt["user_template"].format_map(sd),
        "temperature": prompt.get("default_temperature") or 0.3,
        "max_tokens": prompt.get("default_max_tokens") or 1500,
    }
