"""
回测模块。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import BacktestConfig, ModelConfig
from ..logging_utils import get_logger
from .model_selection import ModelFactory
from ..utils import metrics_dict, pretty_quarter
from ..exceptions import BacktestError


@dataclass
class BacktestWindowResult:
    train_end: str
    test_quarter: str
    model_name: str
    actual: float
    prediction: float
    linear_component: Optional[float] = None
    nonlinear_component: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExpandingWindowBacktester:
    def __init__(self, bt_config: BacktestConfig, model_config: ModelConfig) -> None:
        self.bt_config = bt_config
        self.model_config = model_config
        self.logger = get_logger("sc_macro_agent.backtesting")
        self.factory = ModelFactory(model_config)

    def _quarters(self, panel: pd.DataFrame) -> List[pd.Timestamp]:
        return sorted(pd.to_datetime(panel["quarter_end"]).dropna().unique().tolist())

    def run(
        self,
        panel: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target_value",
        selected_model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if panel.empty:
            raise BacktestError("回测面板为空")
        quarters = self._quarters(panel)
        if len(quarters) < self.bt_config.initial_train_quarters + 1:
            raise BacktestError("季度样本太少，无法做 expanding-window 回测")

        windows: List[BacktestWindowResult] = []
        max_windows = self.bt_config.max_test_windows
        start_idx = self.bt_config.initial_train_quarters

        # 先收集所有可用的测试季度索引，再取尾部 max_windows 个，
        # 确保回测覆盖到最近季度而非最早季度。
        all_test_indices = list(range(start_idx, len(quarters), self.bt_config.step))
        if len(all_test_indices) > max_windows:
            all_test_indices = all_test_indices[-max_windows:]

        for i in all_test_indices:
            test_quarter = quarters[i]
            train_quarters = quarters[:i]
            train_df = panel[panel["quarter_end"].isin(train_quarters)].copy()
            test_df = panel[panel["quarter_end"] == test_quarter].copy()
            if train_df.empty or test_df.empty:
                continue

            X_train = train_df[feature_cols]
            y_train = train_df[target_col]
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]

            model_name = selected_model_name or self.model_config.primary_model
            if model_name == "auto":
                model_name = self.model_config.candidate_models[-1] if self.model_config.candidate_models else "hybrid_residual"
            model = self.factory.create(model_name)
            model.fit(X_train, y_train)

            if hasattr(model, "predict_components"):
                comp = model.predict_components(X_test)
                pred = comp["final_prediction"]
                linear = float(comp["linear_prediction"][0]) if len(comp["linear_prediction"]) else None
                nonlinear = float(comp["nonlinear_correction"][0]) if len(comp["nonlinear_correction"]) else None
            else:
                pred = model.predict(X_test)
                linear = None
                nonlinear = None

            windows.append(
                BacktestWindowResult(
                    train_end=pretty_quarter(train_quarters[-1]),
                    test_quarter=pretty_quarter(test_quarter),
                    model_name=model.model_name,
                    actual=float(y_test.iloc[0]),
                    prediction=float(pred[0]),
                    linear_component=linear,
                    nonlinear_component=nonlinear,
                )
            )

        if len(windows) < self.bt_config.min_required_test_windows:
            raise BacktestError(
                f"有效回测窗口太少: {len(windows)} < {self.bt_config.min_required_test_windows}"
            )

        window_df = pd.DataFrame([w.as_dict() for w in windows])
        metrics = metrics_dict(window_df["actual"], window_df["prediction"])
        # 方向准确率：预测的同比增速变化方向（相比上期）与实际一致的比例
        actual_diff = window_df["actual"].diff().iloc[1:]
        pred_diff = window_df["prediction"].diff().iloc[1:]
        direction_correct = int(((actual_diff > 0) == (pred_diff > 0)).sum())
        direction_total = int(len(actual_diff))
        metrics["direction_accuracy"] = direction_correct / max(direction_total, 1)
        metrics["direction_pairs"] = direction_total
        return {
            "n_windows": int(len(windows)),
            "metrics": metrics,
            "window_results": window_df.to_dict("records"),
        }
