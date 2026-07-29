"""AnalystAgent: 调用 LLM 生成经济简报。"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..config import AppConfig
from ..llm.briefing import BriefingGenerator
from ..prediction_engine import PredictionEngine


class AnalystAgent:
    def __init__(self, config: AppConfig) -> None:
        self.generator = BriefingGenerator(config)

    def run(self, engine: PredictionEngine, critic_feedback: Optional[list] = None) -> Dict[str, Any]:
        inputs = self.generator.build_inputs(engine)
        briefing = self.generator.generate(inputs)

        if critic_feedback:
            # Rewrite with feedback
            briefing = self._rewrite(briefing, critic_feedback)

        return {"briefing": briefing, "inputs": inputs, "rewritten": critic_feedback is not None}

    def _rewrite(self, original: str, issues: list) -> str:
        """根据 Critic 反馈重写简报。"""
        feedback_text = "\n".join(
            f"- [{i['type']}] {i['quote']} → {i['suggestion']}"
            for i in issues if isinstance(i, dict)
        )
        from ..prompts.registry import render
        prompt = render("analyst_rewrite",
            feedback_text=feedback_text,
            original=original,
        )
        return self.generator.llm.chat(
            prompt["system"], prompt["user"],
            temperature=prompt["temperature"], max_tokens=prompt["max_tokens"],
            prompt_id=prompt["id"], prompt_version=prompt["version"],
            caller="analyst_rewrite",
        )
