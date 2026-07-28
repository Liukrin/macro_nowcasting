"""CriticAgent: 对简报做语义审查，输出结构化 JSON。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from ..llm.client import LLMClient
from ..logging_utils import get_logger


CRITIC_SCHEMA = """{
  "passed": true,
  "issues": [
    {
      "type": "时间一致性|数字-语境匹配|逻辑自洽|越权表述|不确定性披露",
      "severity": "high|low",
      "quote": "原文摘录",
      "suggestion": "修改建议"
    }
  ],
  "summary": "一句话总评"
}"""


class CriticAgent:
    def __init__(self) -> None:
        self.llm = LLMClient()
        self.logger = get_logger("sc_macro_agent.critic")

    def review(self, briefing: str, structured_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """审查简报，返回结构化审阅结果。"""
        # Build context from inputs
        pred_q = structured_inputs.get("pred_quarter", "?")
        as_of_q = structured_inputs.get("as_of_quarter", "?")
        actual_latest = structured_inputs.get("actual_latest", "?")
        pred_value = structured_inputs.get("pred_value", "?")

        context = (
            f"数据截至季度：{as_of_q}\n"
            f"该季度实际GDP同比增速：{actual_latest}%\n"
            f"下期预测季度：{pred_q}，预测值：{pred_value}%\n"
            f"重要区分：{as_of_q} 是已发生的实际数据，{pred_q} 是模型预测。\n"
        )

        system = (
            "你是经济简报的严格审阅者。逐项检查以下五项，输出必须为合法 JSON。\n\n"
            "检查项（必须逐一核对）：\n"
            "1. 时间一致性：同一季度是否既被当作实际又被当作预测？"
            "   {as_of_q} 只能是实际值，{pred_q} 只能是预测值。\n"
            "2. 数字-语境匹配：将简报中每个数字与输入数据逐项比对，"
            "数字值必须与对应指标的输入数据一致，不得将其他指标的数字挪用"
            "（如把指标A的值写成指标B的值）。\n"
            "3. 逻辑自洽：结论与数据方向是否矛盾。"
            "如果数据显示某指标大幅下行（如地产投资-8.5%），"
            "简报不得声称该领域\"回暖\"\"强劲\"或\"全面向好\"。\n"
            "4. 越权表述：是否出现输入中没有的政策/会议/文件引用。\n"
            "5. 不确定性披露：预测值是否标注了模型属性与不确定性。\n\n"
            f"输出 JSON schema：\n{CRITIC_SCHEMA}\n\n"
            "示例：\n"
            '{"passed":false,"issues":[{"type":"时间一致性","severity":"high",'
            '"quote":"2026Q1实际增速为5.4%","suggestion":"改为预测值"}],'
            '"summary":"1处时间一致性问题"}'
        )

        user = f"上下文：\n{context}\n\n简报：\n{briefing}"

        for attempt in range(2):
            raw = self.llm.chat(system=system, user=user, temperature=0.1, max_tokens=1000)
            try:
                # Extract JSON from response
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if match:
                    return json.loads(match.group())
            except json.JSONDecodeError:
                self.logger.warning("Critic JSON parse failed, attempt %d", attempt + 1)
        return {"passed": False, "issues": [], "summary": "critic_error: JSON解析失败", "critic_error": True}
