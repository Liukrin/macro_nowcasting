"""
轻量 smoke tests。
"""
from __future__ import annotations

import unittest
from pathlib import Path

from sc_macro_agent import AppConfig, PredictionEngine


class TestPredictionEngine(unittest.TestCase):
    from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
