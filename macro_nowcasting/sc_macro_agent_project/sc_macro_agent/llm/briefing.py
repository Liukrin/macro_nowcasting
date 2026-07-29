"""经济简报生成器——修复时间边界：回顾+展望模式。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .client import LLMClient
from ..config import AppConfig
from ..prediction_engine import PredictionEngine
from ..logging_utils import get_logger


class BriefingGenerator:
    """生成四川省宏观经济简报。

    采用"回顾+展望"模式：
      - 第一段（本期概况）：as_of_quarter 的实际值 + 模型回顾
      - 第三段（下期预测）：as_of_quarter 的下一个季度，来自 predict_next()
    """

    def __init__(self, config: AppConfig) -> None:
        self.logger = get_logger("sc_macro_agent.briefing")
        self.llm = LLMClient.get_instance()
        self.config = config

    def build_inputs(self, engine: "PredictionEngine") -> Dict[str, Any]:
        """装配简报所需的结构化数据。返回 dict。"""
        artifacts_dir = self.config.data.resolve_artifact_dir(create=False)
        data_dir = self.config.data.resolve_dir()

        # --- 数据截至季度 ---
        bp_path = artifacts_dir / "final" / "backtest_predictions.csv"
        if not bp_path.exists():
            self.logger.warning("Missing artifact: %s", bp_path)
            bp_df = pd.DataFrame()
        else:
            bp_df = pd.read_csv(bp_path)
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
        metrics_path = artifacts_dir / "final" / "final_metrics.csv"
        if not metrics_path.exists():
            self.logger.warning("Missing artifact: %s", metrics_path)
            metrics_df = pd.DataFrame()
        else:
            metrics_df = pd.read_csv(metrics_path)
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
        ml_path = data_dir / "monthly_local_features_real.csv"
        ind_lines = []
        if not ml_path.exists():
            self.logger.warning("Missing data file: %s", ml_path)
        else:
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
        from ..prompts.registry import render

        prompt = render("briefing",
            as_of=inputs["as_of_quarter"],
            actual_latest=inputs["actual_latest"],
            pred_quarter=inputs["pred_quarter"],
            pred_value=inputs["pred_value"],
            pred_model=inputs["pred_model"],
            ci_lower=inputs["ci_lower"],
            ci_upper=inputs["ci_upper"],
            recent_data=inputs["recent_data"],
            metrics=inputs["metrics"],
            indicators=inputs["indicators"],
        )

        answer = self.llm.chat(
            system=prompt["system"], user=prompt["user"],
            temperature=prompt["temperature"], max_tokens=prompt["max_tokens"],
            prompt_id=prompt["id"], prompt_version=prompt["version"],
            caller="briefing",
        )

        # 数字校验
        if not self._validate_numbers(answer, prompt["user"]):
            self.logger.warning("数字校验失败，重试一次")
            answer = self.llm.chat(
                system=prompt["system"], user=prompt["user"],
                temperature=0.2, max_tokens=prompt["max_tokens"],
                prompt_id=prompt["id"], prompt_version=prompt["version"],
                caller="briefing",
            )

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
