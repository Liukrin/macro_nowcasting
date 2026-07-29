"""
反向验证：构造故意有错的简报，确认 Critic 检查项真的生效。
"""
from __future__ import annotations

import json
import sys
sys.path.insert(0, "D:/PythonProject/macro_nowcasting/sc_macro_agent_project")

from dotenv import load_dotenv
load_dotenv()

from sc_macro_agent.agents.critic_agent import CriticAgent

# 构造三处植入错误的简报
bad_briefing = """**四川省宏观经济运行简报（2025年度）**

**（一）本期概况**

2025年第四季度，四川省经济运行总体平稳。全年GDP累计同比增速为5.5%。

**（二）主要指标动态**

2025年12月，规模以上工业增加值累计同比增长6.5%，GDP累计同比增速为6.5%，
与工业增速持平，显示经济增长强劲。房地产开发投资累计同比下降8.5%。

**（三）下期预测与依据**

2026年第一季度实际增速为5.4%，延续了2025年下半年的稳健增长态势。
模型预测，这一增速反映了根据近期出台的产业扶持政策所带来的积极效应。

**（四）风险提示**

外部环境不确定性依然存在。"""

# 真实 structured_inputs
structured_inputs = {
    "as_of_quarter": "2025Q4",
    "pred_quarter": "2026Q1",
    "actual_latest": 5.5,
    "pred_value": 5.37,
}

critic = CriticAgent()
result = critic.review(briefing=bad_briefing, structured_inputs=structured_inputs)

print(json.dumps(result, ensure_ascii=False, indent=2))
