"""ModelAgent: 调用已冻结的训练与预测流程，不调 LLM。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from ..prediction_engine import PredictionEngine


class ModelAgent:
    def run(self, engine: PredictionEngine) -> Dict[str, Any]:
        # Train
        engine.run_agent(goal="audit_build_train", save_artifacts=False)

        # Predict next quarter
        pred = engine.predict_next()

        # Backtest metrics from frozen results
        bp = Path("artifacts/final/backtest_predictions.csv")
        metrics = Path("artifacts/final/final_metrics.csv")
        import pandas as pd
        backtest_rmse = 0.0
        vs_baseline = 1.0
        dir_acc = 0.0
        if metrics.exists():
            df = pd.read_csv(metrics)
            emc = df[df["model"] == "elastic_midas_chronos"]
            lv = df[df["model"] == "last_value"]
            if not emc.empty and not lv.empty:
                backtest_rmse = float(emc.iloc[0]["rmse"])
                lv_rmse = float(lv.iloc[0]["rmse"])
                vs_baseline = backtest_rmse / lv_rmse if lv_rmse > 0 else 1.0
                dir_acc = float(emc.iloc[0]["dir_acc"])

        # Caveats from known_limitations.md
        caveats = _load_caveats()

        return {
            "prediction_quarter": pred.get("prediction_quarter", "N/A"),
            "prediction_value": pred.get("prediction_value", 0.0),
            "model_name": pred.get("model_name", "unknown"),
            "confidence_interval": pred.get("confidence_interval", {}),
            "backtest_rmse": backtest_rmse,
            "vs_baseline_ratio": round(vs_baseline, 3),
            "direction_accuracy": round(dir_acc, 3),
            "caveats": caveats,
        }


def _load_caveats() -> list:
    path = Path("artifacts/final/known_limitations.md")
    if not path.exists():
        return ["无已知局限文档"]
    text = path.read_text(encoding="utf-8")
    # Extract numbered limitation headers
    import re
    caveats = []
    for m in re.finditer(r'##\s*\d+\.\s*(.+)', text):
        caveats.append(m.group(1).strip())
    return caveats[:6]
