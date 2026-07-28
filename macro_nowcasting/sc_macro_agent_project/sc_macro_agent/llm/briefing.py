"""经济简报生成器——修复时间边界：回顾+展望模式。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .client import LLMClient
from ..prediction_engine import PredictionEngine
from ..logging_utils import get_logger


class BriefingGenerator:
    """生成四川省宏观经济简报。

    采用"回顾+展望"模式：
      - 第一段（本期概况）：as_of_quarter 的实际值 + 模型回顾
      - 第三段（下期预测）：as_of_quarter 的下一个季度，来自 predict_next()
    """

    def __init__(self) -> None:
        self.logger = get_logger("sc_macro_agent.briefing")
        self.llm = LLMClient()

    def build_inputs(self, engine: "PredictionEngine") -> Dict[str, Any]:
        """装配简报所需的结构化数据。返回 dict。"""
        # --- 数据截至季度 ---
        bp_path = Path("artifacts/final/backtest_predictions.csv")
        bp_df = pd.read_csv(bp_path) if bp_path.exists() else pd.DataFrame()
        if not bp_df.empty:
            last_row = bp_df.iloc[-1]
            as_of_quarter = last_row["test_quarter"]
            actual_latest = last_row["actual"]
        else:
            as_of_quarter = "N/A"
            actual_latest = 0.0

        # --- 近期 4 季度实际值 ---
        recent_lines = []
        if not bp_df.empty:
            for _, r in bp_df.tail(4).iterrows():
                recent_lines.append(
                    f"{r['test_quarter']}: GDP累计同比实际值 {r['actual']:.1f}%"
                )

        # --- 下期预测（来自 predict_next，不是 backtest） ---
        try:
            pred = engine.predict_next()
            pred_quarter = pred.get("prediction_quarter", "N/A")
            pred_value = pred.get("prediction_value", 0.0)
            pred_model = pred.get("model_name", "unknown")
            ci = pred.get("confidence_interval", {})
        except Exception as exc:
            self.logger.warning("predict_next failed: %s", exc)
            pred_quarter = "N/A"
            pred_value = 0.0
            pred_model = "unknown"
            ci = {}

        # --- 回测指标 ---
        metrics_path = Path("artifacts/final/final_metrics.csv")
        metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        metrics_text = ""
        if not metrics_df.empty:
            for model_name in ["elastic_midas_chronos", "last_value"]:
                row = metrics_df[metrics_df["model"] == model_name]
                if not row.empty:
                    r = row.iloc[0]
                    metrics_text += (
                        f"  {model_name}: RMSE={r['rmse']:.2f}, "
                        f"MAE={r['mae']:.2f}, R²={r['r2']:.2f}, "
                        f"方向准确率={r['dir_acc']:.1%}\n"
                    )

        # --- 月度指标 ---
        ml_path = Path("data/monthly_local_features_real.csv")
        ind_lines = []
        if ml_path.exists():
            ml = pd.read_csv(ml_path)
            ml["date"] = pd.to_datetime(ml["date"])
            ml_recent = ml[ml["date"] >= "2025-01-01"]
            for ind in sorted(ml_recent["indicator_name"].unique()):
                sub = ml_recent[ml_recent["indicator_name"] == ind].sort_values("date")
                if len(sub) > 0:
                    last = sub.iloc[-1]
                    ind_lines.append(
                        f"  {ind}: {last['date'].strftime('%Y-%m')} "
                        f"值 {last['indicator_value']:.1f}%"
                    )

        return {
            "as_of_quarter": as_of_quarter,
            "actual_latest": actual_latest,
            "pred_quarter": pred_quarter,
            "pred_value": pred_value,
            "pred_model": pred_model,
            "ci_lower": ci.get("lower", "N/A"),
            "ci_upper": ci.get("upper", "N/A"),
            "recent_data": "\n".join(recent_lines),
            "metrics": metrics_text,
            "indicators": "\n".join(ind_lines[:12]),
        }

    def generate(self, inputs: Dict[str, Any]) -> str:
        """生成简报并做数字校验。"""
        as_of = inputs["as_of_quarter"]

        system = """你是省级宏观经济分析助手，输出正式书面简报。

规则（严格遵守）：
1. 只能使用用户消息中提供的数字，禁止引入任何外部数据。
2. 禁止提及任何具体政策文件、会议名称、领导讲话。
3. 对预测值必须注明"模型预测，存在不确定性"。
4. 输出结构固定为四段：
   (1) 本期概况 —— 基于数据截至季度的实际值；
       可附"模型此前对本期的预测为X%，误差Y个百分点"作为回顾
   (2) 主要指标动态
   (3) 下期预测与依据 —— 必须是下一个季度，注明是模型预测
   (4) 风险提示
5. 总长度控制在 400-600 字。
6. 使用正式、客观的书面中文。
"""

        user = f"""数据截至 {as_of} 季度。

请根据以下数据生成四川省经济简报。注意：
- {as_of} 是数据中最后一个有真实值的季度，已经发生完毕。
- 第三段的下期预测必须是 {inputs['pred_quarter']}，不是 {as_of}。

【{as_of} 实际值】
GDP累计同比增速：{inputs['actual_latest']:.1f}%

【近四个季度GDP实际值】
{inputs['recent_data']}

【下期预测（{inputs['pred_quarter']}）】
预测值：GDP累计同比增速 {inputs['pred_value']:.1f}%
模型：{inputs['pred_model']}
置信区间：[{inputs['ci_lower']}, {inputs['ci_upper']}]

【回测评估】
{inputs['metrics']}

【近期月度指标】
{inputs['indicators']}
"""

        answer = self.llm.chat(system=system, user=user, temperature=0.3, max_tokens=2000)

        # 数字校验
        if not self._validate_numbers(answer, user):
            self.logger.warning("数字校验失败，重试一次")
            answer = self.llm.chat(system=system, user=user, temperature=0.2, max_tokens=2000)

        return answer

    def _validate_numbers(self, briefing: str, input_text: str) -> bool:
        """校验简报中的数字是否存在于输入数据中。"""
        briefing_nums = set()
        for m in re.finditer(r'(\d+\.?\d*)\s*%?', briefing):
            v = float(m.group(1))
            if v > 0.5:
                briefing_nums.add(round(v, 1))
        input_nums = set()
        for m in re.finditer(r'(\d+\.?\d*)', input_text):
            v = float(m.group(1))
            if v > 0.5:
                input_nums.add(round(v, 1))
        extra = briefing_nums - input_nums
        if extra:
            self.logger.warning("数字校验: 以下数字不在输入中: %s", extra)
            return False
        return True
