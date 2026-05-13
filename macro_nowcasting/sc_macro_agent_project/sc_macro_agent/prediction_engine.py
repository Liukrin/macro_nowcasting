"""
统一预测引擎

这是整个项目的核心 orchestration 层：
- DataManager 负责读数据
- DataQualityAuditor 负责审计
- DFMModel 负责因子提取
- FeatureEngineer 负责季度面板构建
- ModelSelector / Backtester 负责训练和回测
- ReportBuilder 负责产物输出
- ForecastAgent 负责“任务步骤留痕”
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .agent import ForecastAgent
from .config import AppConfig
from .data.data_manager import DataManager, DataBundle
from .data.data_quality import DataQualityAuditor
from .features.feature_engineering import FeatureEngineer
from .logging_utils import get_logger
from .models.model_selection import ModelSelector, ModelFactory
from .models.backtesting import ExpandingWindowBacktester
from .models.dfm_model import DFMModel
from .api.reporting import ReportBuilder
from .utils import metrics_dict, pretty_quarter, save_json, save_text
from .exceptions import BacktestError, ModelTrainingError, FeatureBuildError, DataNotReadyError


class PredictionEngine:
    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig.from_env()
        self.logger = get_logger("sc_macro_agent.prediction_engine")
        self.data_manager = DataManager(self.config.data)
        self.auditor = DataQualityAuditor()
        self.feature_engineer = FeatureEngineer(self.config.features)
        self.dfm_model = DFMModel(
            n_factors=self.config.model.dfm_n_factors,
            standardize=self.config.features.standardize_before_pca,
        )
        self.selector = ModelSelector(self.config.model)
        self.backtester = ExpandingWindowBacktester(self.config.backtest, self.config.model)
        self.report_builder = ReportBuilder()
        self.agent = ForecastAgent(self.config.agent)

        self.bundle: Optional[DataBundle] = None
        self.audit_result: Optional[Dict[str, Any]] = None
        self.feature_artifacts = None
        self.selected_model = None
        self.selected_model_name: Optional[str] = None
        self.leaderboard: List[Dict[str, Any]] = []
        self.backtest_result: Optional[Dict[str, Any]] = None
        self.latest_prediction: Optional[Dict[str, Any]] = None
        self.last_run_at: Optional[str] = None
        self.initialized = False
        self.warnings: List[str] = []

    # ------------------------------------------------------------------
    # 基础流程
    # ------------------------------------------------------------------
    def initialize(self, force_refresh: bool = False) -> None:
        if self.initialized and not force_refresh:
            return
        self.data_manager.initialize()
        self.bundle = self.data_manager.get_bundle()
        self.initialized = True
        self.last_run_at = datetime.now().isoformat(timespec="seconds")

    def audit_data(self, save_artifacts: bool = True) -> Dict[str, Any]:
        self.initialize()
        assert self.bundle is not None
        audit_step = self.agent.start_step("audit_data")
        result = self.auditor.run_full_audit(
            quarterly_target_df=self.bundle.quarterly_target,
            monthly_local_df=self.bundle.monthly_local,
            monthly_national_df=self.bundle.monthly_national,
            quarterly_panel_df=self.bundle.quarterly_panel,
            metadata_df=self.bundle.metadata,
        )
        self.audit_result = result

        artifacts: Dict[str, str] = {}
        if save_artifacts:
            out_dir = self.config.data.resolve_artifact_dir() / "audit"
            out_dir.mkdir(parents=True, exist_ok=True)
            report = self.auditor.build_text_report(result, self.bundle.dataset_mode)
            artifacts["audit_json"] = str(save_json(out_dir / "audit_result.json", result))
            artifacts["audit_report"] = str(save_text(out_dir / "audit_report.txt", report))

        audit_step.close("completed", {"status": result["status"], "artifacts": artifacts})
        result["artifacts"] = artifacts
        result["dataset_mode"] = self.bundle.dataset_mode
        return result

    def build_features(self) -> Dict[str, Any]:
        self.initialize()
        assert self.bundle is not None
        step = self.agent.start_step("build_features")

        dfm_artifacts = self.dfm_model.fit_transform(
            self.bundle.monthly_local,
            self.bundle.monthly_national,
        )

        feature_artifacts = self.feature_engineer.build_training_panel(
            quarterly_target_df=self.bundle.quarterly_target,
            monthly_local_df=self.bundle.monthly_local,
            monthly_national_df=self.bundle.monthly_national,
            quarterly_panel_df=self.bundle.quarterly_panel,
            metadata_df=self.bundle.metadata,
            quarterly_factor_frame=dfm_artifacts.quarterly_factor_frame,
        )
        feature_artifacts.monthly_factor_frame = dfm_artifacts.monthly_factor_frame
        feature_artifacts.quarterly_factor_frame = dfm_artifacts.quarterly_factor_frame
        self.feature_artifacts = feature_artifacts

        info = {
            "n_rows": int(len(feature_artifacts.training_panel)),
            "n_features": int(len(feature_artifacts.feature_columns)),
            "notes": feature_artifacts.notes,
            "feature_family_summary": feature_artifacts.feature_registry.summary_by_family(),
        }
        step.close("completed", info)
        return info

    def _train_valid_split(self, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        panel = panel.sort_values("quarter_end").reset_index(drop=True)
        quarters = sorted(panel["quarter_end"].unique().tolist())
        n = len(quarters)
        if n < 3:
            return panel.iloc[:-1].copy(), panel.iloc[-1:].copy()

        # Use ~20% of quarters for validation, min 2, max 12
        valid_size = max(2, min(12, n // 5))
        valid_quarters = set(quarters[-valid_size:])
        train_df = panel[~panel["quarter_end"].isin(valid_quarters)].copy()
        valid_df = panel[panel["quarter_end"].isin(valid_quarters)].copy()
        if train_df.empty:
            split = max(1, len(panel) - valid_size)
            train_df = panel.iloc[:split].copy()
            valid_df = panel.iloc[split:].copy()
        return train_df, valid_df

    def train(self, force_rebuild: bool = False) -> Dict[str, Any]:
        self.initialize()
        if self.feature_artifacts is None or force_rebuild:
            self.build_features()
        assert self.feature_artifacts is not None

        step = self.agent.start_step("train")
        panel = self.feature_artifacts.training_panel.copy()
        feature_cols = self.feature_artifacts.feature_columns
        target_col = self.feature_artifacts.target_column

        train_df, valid_df = self._train_valid_split(panel)
        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_valid = valid_df[feature_cols]
        y_valid = valid_df[target_col]

        model, leaderboard = self.selector.select_best(X_train, y_train, X_valid, y_valid)
        self.selected_model = model
        self.selected_model_name = model.model_name
        self.leaderboard = leaderboard

        pred = model.predict(X_valid)
        metrics = metrics_dict(y_valid.tolist(), pred.tolist() if hasattr(pred, "tolist") else list(pred))
        payload = {
            "status": "trained",
            "selected_model": model.model_name,
            "n_rows": int(len(panel)),
            "n_features": int(len(feature_cols)),
            "metrics": metrics,
            "warnings": self.warnings,
            "leaderboard": leaderboard,
        }
        step.close("completed", payload)
        return payload

    def backtest(self) -> Dict[str, Any]:
        self.initialize()
        if self.feature_artifacts is None:
            self.build_features()
        if self.selected_model_name is None:
            self.train()
        assert self.feature_artifacts is not None

        step = self.agent.start_step("backtest")
        panel = self.feature_artifacts.training_panel.copy()
        feature_cols = self.feature_artifacts.feature_columns

        result = self.backtester.run(
            panel=panel,
            feature_cols=feature_cols,
            target_col=self.feature_artifacts.target_column,
            selected_model_name=self.selected_model_name,
        )
        self.backtest_result = result
        step.close("completed", result)
        return result

    def predict_next(self, use_blend: bool = True) -> Dict[str, Any]:
        self.initialize()
        if self.feature_artifacts is None:
            self.build_features()
        if self.selected_model is None:
            self.train()

        assert self.feature_artifacts is not None
        assert self.selected_model is not None
        panel = self.feature_artifacts.training_panel.copy()
        feature_cols = self.feature_artifacts.feature_columns
        latest_row = panel.sort_values("quarter_end").tail(1).copy()
        current_quarter = pd.to_datetime(latest_row["quarter_end"].iloc[0])
        next_quarter = (current_quarter + pd.offsets.QuarterEnd()).to_pydatetime()

        prediction_value = float(self.selected_model.predict(latest_row[feature_cols])[0])
        result = {
            "target_indicator": self.config.features.target_indicator,
            "prediction_quarter": pretty_quarter(next_quarter),
            "based_on_latest_quarter": pretty_quarter(current_quarter),
            "model_name": self.selected_model.model_name,
            "prediction_value": prediction_value,
            "benchmark_value": float(latest_row["target_value"].iloc[0]),
            "confidence_interval": self._build_confidence_interval(prediction_value),
            "top_features": self.selected_model.get_feature_importance(10),
            "notes": [
                "prediction_generated_from_latest_available_quarter_features",
                "if_real_data_is_short_treat_as_demo_nowcast_not_production_forecast",
            ],
        }
        if hasattr(self.selected_model, "predict_components"):
            comp = self.selected_model.predict_components(latest_row[feature_cols])
            result["components"] = {
                "linear_prediction": float(comp["linear_prediction"][0]),
                "nonlinear_correction": float(comp["nonlinear_correction"][0]),
                "final_prediction": float(comp["final_prediction"][0]),
            }
        else:
            result["components"] = {"prediction": prediction_value}

        self.latest_prediction = result
        return result

    def _build_confidence_interval(self, point_pred: float) -> Dict[str, float]:
        if self.backtest_result and self.backtest_result.get("metrics"):
            rmse = float(self.backtest_result["metrics"].get("rmse", 0.8))
        elif self.selected_model is not None and getattr(self.selected_model, "train_residuals_", None) is not None:
            residuals = pd.Series(self.selected_model.train_residuals_)
            rmse = float((residuals.pow(2).mean()) ** 0.5)
        else:
            rmse = 1.0
        spread = max(0.3, 1.28 * rmse)
        return {"lower": point_pred - spread, "upper": point_pred + spread}

    # ------------------------------------------------------------------
    # 汇总与导出
    # ------------------------------------------------------------------
    def get_factor_summary(self) -> Dict[str, Any]:
        return self.dfm_model.summary()

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": "ready" if self.initialized else "not_ready",
            "initialized": self.initialized,
            "dataset_mode": self.bundle.dataset_mode if self.bundle else None,
            "last_run_at": self.last_run_at,
            "warnings": self.warnings,
            "selected_model": self.selected_model_name,
            "n_rows": int(len(self.feature_artifacts.training_panel)) if self.feature_artifacts else 0,
            "n_features": int(len(self.feature_artifacts.feature_columns)) if self.feature_artifacts else 0,
        }

    def export_artifacts(self) -> Dict[str, str]:
        self.initialize()
        out_dir = self.config.data.resolve_artifact_dir() / datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)

        artifacts: Dict[str, str] = {}
        if self.audit_result is not None:
            artifacts.update(self.report_builder.export_bundle(out_dir / "audit_bundle", {
                "project_name": self.config.project_name,
                "dataset_mode": self.bundle.dataset_mode if self.bundle else None,
                "target_indicator": self.config.features.target_indicator,
                "selected_model": self.selected_model_name,
                "n_rows": int(len(self.feature_artifacts.training_panel)) if self.feature_artifacts else 0,
                "n_features": int(len(self.feature_artifacts.feature_columns)) if self.feature_artifacts else 0,
                "metrics": self.backtest_result["metrics"] if self.backtest_result else {},
                "prediction": self.latest_prediction or {},
                "factor_summary": self.get_factor_summary(),
                "top_features": self.selected_model.get_feature_importance(10) if self.selected_model else [],
            }))

        if self.feature_artifacts is not None:
            panel_path = out_dir / "training_panel.csv"
            self.feature_artifacts.training_panel.to_csv(panel_path, index=False, encoding="utf-8-sig")
            artifacts["training_panel"] = str(panel_path)

            feat_meta = self.feature_artifacts.feature_registry.to_frame()
            meta_path = out_dir / "feature_registry.csv"
            feat_meta.to_csv(meta_path, index=False, encoding="utf-8-sig")
            artifacts["feature_registry"] = str(meta_path)

        if self.backtest_result is not None:
            bt_path = out_dir / "backtest_results.json"
            save_json(bt_path, self.backtest_result)
            artifacts["backtest_results"] = str(bt_path)

        if self.latest_prediction is not None:
            pred_path = out_dir / "latest_prediction.json"
            save_json(pred_path, self.latest_prediction)
            artifacts["latest_prediction"] = str(pred_path)

        summary_path = out_dir / "run_summary.json"
        save_json(summary_path, self.summarize())
        artifacts["run_summary"] = str(summary_path)
        return artifacts

    def summarize(self) -> Dict[str, Any]:
        return {
            "project_name": self.config.project_name,
            "version": self.config.version,
            "dataset_mode": self.bundle.dataset_mode if self.bundle else None,
            "target_indicator": self.config.features.target_indicator,
            "selected_model": self.selected_model_name,
            "n_rows": int(len(self.feature_artifacts.training_panel)) if self.feature_artifacts else 0,
            "n_features": int(len(self.feature_artifacts.feature_columns)) if self.feature_artifacts else 0,
            "metrics": self.backtest_result["metrics"] if self.backtest_result else {},
            "prediction": self.latest_prediction or {},
            "factor_summary": self.get_factor_summary(),
            "top_features": self.selected_model.get_feature_importance(10) if self.selected_model else [],
            "leaderboard": self.leaderboard,
            "agent": self.agent.summary(),
        }

    def run_agent(self, goal: str = "audit_build_train_backtest_report", save_artifacts: bool = True, force_refresh: bool = False) -> Dict[str, Any]:
        if force_refresh:
            self.initialized = False
            self.bundle = None
            self.feature_artifacts = None
            self.selected_model = None
            self.selected_model_name = None
            self.leaderboard = []
            self.backtest_result = None
            self.latest_prediction = None

        self.initialize(force_refresh=force_refresh)

        if "audit" in goal:
            self.audit_data(save_artifacts=save_artifacts)
        if "build" in goal:
            self.build_features()
        if "train" in goal:
            self.train()
        if "backtest" in goal:
            try:
                self.backtest()
            except BacktestError as exc:
                self.warnings.append(str(exc))
                self.agent.record_warning(str(exc))
        if "report" in goal or "predict" in goal:
            self.predict_next()

        artifacts = self.export_artifacts() if save_artifacts else {}
        summary = self.summarize()
        return {
            "goal": goal,
            "status": "completed",
            "steps": self.agent.summary()["steps"],
            "summary": summary,
            "artifacts": artifacts,
            "warnings": self.agent.summary()["warnings"],
        }
