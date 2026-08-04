"""CriticAgent 输出结构校验 schema。

用 pydantic v2 对 LLM 返回的 JSON 做宽松校验：
- 所有字段都有默认值，缺字段不报错
- severity 归一化到 {"high", "low"}
- issues 清洗掉非 dict 元素
- from_raw() 类方法保证永不抛异常
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator


class CriticIssue(BaseModel):
    """单条审阅问题。"""
    type: str = "未分类"
    severity: str = "low"
    quote: str = ""
    suggestion: str = ""

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, v: Any) -> str:
        """severity 归一化：不在 {"high","low"} 内的一律改成 "low"。"""
        if v not in ("high", "low"):
            return "low"
        return v


class CriticReview(BaseModel):
    """CriticAgent 审阅结果。"""
    passed: bool = False
    issues: list[CriticIssue] = []
    summary: str = ""
    critic_error: bool = False

    @field_validator("issues", mode="before")
    @classmethod
    def clean_issues(cls, v: Any) -> list:
        """宽松清洗 issues：
        - 非 list → 返回 []
        - list 内非 dict 的元素跳过（不抛异常）
        """
        if not isinstance(v, list):
            return []
        return [item for item in v if isinstance(item, dict)]

    @field_validator("passed", mode="before")
    @classmethod
    def normalize_passed(cls, v: Any) -> bool:
        """passed 必须偏保守：
        - 真 bool → 原样
        - 字符串：只有 "true"/"True" 才算 True，其余一律 False
        - 其他类型 → False
        """
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v in ("true", "True")
        return False

    @classmethod
    def from_raw(cls, raw: dict) -> "CriticReview":
        """从原始 LLM 输出的 dict 构建 CriticReview。

        内部 try/except，校验失败时返回降级对象
        （passed=False, critic_error=True），永不抛异常。
        """
        try:
            return cls(**raw)
        except Exception:
            return cls(
                passed=False,
                summary="审阅结果结构不合法",
                critic_error=True,
            )
