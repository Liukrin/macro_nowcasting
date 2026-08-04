"""AnalystAgent: 调用 LLM 生成经济简报。"""
from __future__ import annotations

from typing import Any, Dict

from ..config import AppConfig
from ..llm.briefing import BriefingGenerator
from ..logging_utils import get_logger
from ..prediction_engine import PredictionEngine


class AnalystAgent:
    def __init__(self, config: AppConfig) -> None:
        self.generator = BriefingGenerator(config)
        self.logger = get_logger("sc_macro_agent.analyst")

    def run(self, engine: PredictionEngine) -> Dict[str, Any]:
        """首次生成简报。返回 {"briefing", "inputs"}。"""
        inputs = self.generator.build_inputs(engine)
        briefing = self.generator.generate(inputs)
        return {"briefing": briefing, "inputs": inputs}

    def rewrite(self, original_briefing: str, issues: list) -> dict:
        """根据 Critic 反馈重写【已有】简报，仅做一次 LLM 调用。

        不碰 engine、不调 build_inputs。
        返回 {"briefing": str, "rewritten": bool}。
        """
        # 拒绝空原文：拿空串去重写必然产出元回复，白烧 token 还污染最终产物
        if not (original_briefing or "").strip():
            self.logger.warning("rewrite 收到空原文，拒绝重写，返回原文")
            return {"briefing": original_briefing, "rewritten": False}

        # 短路保护：无有效 issue → 直接返回原文
        valid_issues = [i for i in (issues or []) if isinstance(i, dict)]
        if not valid_issues:
            return {"briefing": original_briefing, "rewritten": False}

        feedback_text = "\n".join(
            f"- [{i.get('type', '未分类')}] {i.get('quote', '')} → {i.get('suggestion', '')}"
            for i in valid_issues
        )
        from ..prompts.registry import render
        prompt = render("analyst_rewrite",
            feedback_text=feedback_text,
            original=original_briefing,
        )
        try:
            rewritten = self.generator.llm.chat(
                prompt["system"], prompt["user"],
                temperature=prompt["temperature"], max_tokens=prompt["max_tokens"],
                prompt_id=prompt["id"], prompt_version=prompt["version"],
                caller="analyst_rewrite",
            )
            # 失败保护：mock 模式或空返回 → 保留原文
            if not rewritten or rewritten.startswith("[MOCK LLM]"):
                self.logger.warning("rewrite 返回 mock/空内容，保留原文")
                return {"briefing": original_briefing, "rewritten": False}
            return {"briefing": rewritten, "rewritten": True}
        except Exception as exc:
            self.logger.warning("rewrite LLM 调用失败: %s，保留原文", exc)
            return {"briefing": original_briefing, "rewritten": False}
