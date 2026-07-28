"""AnalystAgent: 调用 LLM 生成经济简报。"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..llm.briefing import BriefingGenerator
from ..prediction_engine import PredictionEngine


class AnalystAgent:
    def __init__(self) -> None:
        self.generator = BriefingGenerator()

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
        prompt = (
            f"以下是原始简报，存在以下问题需要修正：\n\n{feedback_text}\n\n"
            f"原始简报：\n{original}\n\n"
            f"请修正上述问题后重新输出完整简报。保持四段结构。"
        )
        return self.generator.llm.chat(
            system="你是经济简报编辑。根据审阅意见修正简报，只输出修正后的全文。",
            user=prompt,
            temperature=0.2,
            max_tokens=2000,
        )
