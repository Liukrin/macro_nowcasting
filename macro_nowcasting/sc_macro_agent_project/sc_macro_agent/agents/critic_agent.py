"""CriticAgent: 对简报做语义审查，输出结构化 JSON。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from ..llm.client import LLMClient
from ..logging_utils import get_logger
from ..prompts.registry import render


def _extract_json(raw: str) -> dict | None:
    """从 LLM 原始输出中稳健提取 JSON 对象。

    按优先级尝试：
      1. 直接 json.loads（纯 JSON 输出）
      2. 剥离 markdown 代码围栏（```json ... ``` 或 ``` ... ```）
      3. 括号配对扫描——从第一个 '{' 开始计数，到配对 '}' 结束，
         正确处理字符串内的花括号与转义字符

    任一成功即返回解析后的 dict，全部失败返回 None。
    """
    text = raw.strip()

    # 1) 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) 剥离 markdown 代码围栏
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3) 括号配对扫描：找到第一个完整的 JSON 对象
    start = text.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == '\\':
                escaped = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        return None  # 找到了配对花括号但不是有效 JSON
    return None


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
            result = _extract_json(raw)
            if result is not None:
                return result

            # 输出被长度限制截断 → 提高上限重试
            if meta.get("finish_reason") == "length":
                cur_max_tokens = int(cur_max_tokens * 1.5)
                self.logger.info("Output truncated (finish_reason=length), retrying with max_tokens=%d", cur_max_tokens)
            else:
                self.logger.warning(
                    "Critic JSON parse failed, attempt %d. Raw output (head 500): %s",
                    attempt + 1, raw[:500],
                )

        return {"passed": False, "issues": [], "summary": "审阅解析失败，简报未经有效审阅", "critic_error": True}

