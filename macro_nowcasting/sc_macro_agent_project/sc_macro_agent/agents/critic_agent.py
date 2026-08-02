"""CriticAgent: 对简报做语义审查，输出结构化 JSON。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from ..llm.client import LLMClient
from ..logging_utils import get_logger
from ..prompts.registry import render


class CriticAgent:
    def __init__(self, prompt_version: str | None = None) -> None:
        self.llm = LLMClient.get_instance()
        self.logger = get_logger("sc_macro_agent.critic")
        self._prompt_version = prompt_version  # 用于回归测试指定历史版本

    def review(self, briefing: str, structured_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """审查简报，返回结构化审阅结果。输出被截断时自适应提高 max_tokens 重试。"""
        try:
            pred_value = round(float(structured_inputs.get("pred_value")), 1)
        except (TypeError, ValueError):
            pred_value = "?"
        prompt = render("critic_review", version=self._prompt_version,
            as_of_q=structured_inputs.get("as_of_quarter", "?"),
            pred_q=structured_inputs.get("pred_quarter", "?"),
            actual_latest=structured_inputs.get("actual_latest", "?"),
            pred_value=pred_value,
            ci_lower=structured_inputs.get("ci_lower", "N/A"),
            ci_upper=structured_inputs.get("ci_upper", "N/A"),
            recent_data=structured_inputs.get("recent_data", ""),
            metrics=structured_inputs.get("metrics", ""),
            indicators=structured_inputs.get("indicators", ""),
            briefing=briefing,
        )

        cur_max_tokens = prompt["max_tokens"]

        for attempt in range(2):
            meta = self.llm.chat_with_meta(
                prompt["system"], prompt["user"],
                temperature=prompt["temperature"], max_tokens=cur_max_tokens,
                prompt_id=prompt["id"], prompt_version=prompt["version"],
                caller="critic",
            )
            raw = meta["response"]
            try:
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if match:
                    return json.loads(match.group())
            except json.JSONDecodeError:
                self.logger.warning("Critic JSON parse failed, attempt %d", attempt + 1)

            # 输出被长度限制截断 → 提高上限重试
            if meta.get("finish_reason") == "length":
                cur_max_tokens = int(cur_max_tokens * 1.5)
                self.logger.info("Output truncated (finish_reason=length), retrying with max_tokens=%d", cur_max_tokens)

        return {"passed": False, "issues": [], "summary": "critic_error: JSON解析失败", "critic_error": True}
