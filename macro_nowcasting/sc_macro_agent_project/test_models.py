"""
轻量 smoke tests。
"""
from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from sc_macro_agent import AppConfig, PredictionEngine
from sc_macro_agent.models.backtesting import ExpandingWindowBacktester
from sc_macro_agent.exceptions import BacktestError


class TestPredictionEngine(unittest.TestCase):

    def setUp(self):
        self.config = AppConfig()
        project_dir = Path(__file__).resolve().parent
        self.config.data.data_dir = str(project_dir / "data")
        self.config.data.artifact_dir = str(project_dir / "artifacts")
        self.config.data.dataset_mode = "real"

    def test_audit(self):
        engine = PredictionEngine(self.config)
        result = engine.audit_data(save_artifacts=False)
        self.assertIn("status", result)
        self.assertIn("checks", result)

    def test_feature_build(self):
        engine = PredictionEngine(self.config)
        engine.initialize()
        result = engine.build_features()
        self.assertGreaterEqual(result["n_rows"], 1)
        self.assertGreaterEqual(result["n_features"], 1)

    def test_train_predict(self):
        engine = PredictionEngine(self.config)
        engine.run_agent(goal="audit_build_train_predict", save_artifacts=False)
        pred = engine.predict_next()
        self.assertIn("prediction_value", pred)
        self.assertIn("model_name", pred)

    def test_backtest_base_series_aligned(self):
        """正常路径：base_series 与 panel 索引一致时回测正常完成。"""
        engine = PredictionEngine(self.config)
        engine.initialize()
        engine.build_features()
        panel = engine.feature_artifacts.training_panel.copy()
        panel_t, base_series = engine._apply_target_transform(panel)

        backtester = ExpandingWindowBacktester(self.config.backtest, self.config.model)
        result = backtester.run(
            panel=panel_t,
            feature_cols=engine.feature_artifacts.feature_columns,
            target_col=engine.feature_artifacts.target_column,
            selected_model_name="elastic_midas",
            base_series=base_series,
        )
        self.assertGreater(result["n_windows"], 0)
        self.assertIn("rmse", result["metrics"])

    def test_backtest_base_series_misaligned_raises(self):
        """异常路径：reset_index 破坏对齐后应抛出 BacktestError。"""
        engine = PredictionEngine(self.config)
        engine.initialize()
        engine.build_features()
        panel = engine.feature_artifacts.training_panel.copy()
        panel_t, base_series = engine._apply_target_transform(panel)

        # 人为破坏索引对齐：对 panel 执行 reset_index
        panel_broken = panel_t.reset_index(drop=True)

        backtester = ExpandingWindowBacktester(self.config.backtest, self.config.model)
        with self.assertRaises(BacktestError):
            backtester.run(
                panel=panel_broken,
                feature_cols=engine.feature_artifacts.feature_columns,
                target_col=engine.feature_artifacts.target_column,
                selected_model_name="elastic_midas",
                base_series=base_series,
            )


if __name__ == "__main__":
    unittest.main()
