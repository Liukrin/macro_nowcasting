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

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .agent import ForecastAgent
from .config import AppConfig
from .data.data_manager import DataManager, DataBundle
from .data.data_quality import DataQualityAuditor
from .features.feature_engineering import FeatureEngineer
from .logging_utils import get_logger
from .models.model_selection import ModelSelector
from .models.backtesting import ExpandingWindowBacktester
from .models.dfm_model import DFMModel
from .api.reporting import ReportBuilder
from .utils import metrics_dict, pretty_quarter, save_json, save_text
from .exceptions import BacktestError


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
        # Lazy-load chronos corrector on first predict
        self._chronos_corrector: Any = None
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
        self._chronos_corrector: Any = None
        self._chronos_state = "not_loaded"  # not_loaded | ready | failed

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

        # DFM/PCA: 当使用白名单特征选择时跳过。
        # 白名单不会选入 dfm_factor_* 列，PCA 在全样本上 fit 构成信息泄漏。
        # 因此 policy 模式下直接断开 PCA 管道，避免将来误用。
        if self.config.features.use_policy_selection:
            dfm_artifacts = None
            quarterly_factor_frame = None
        else:
            dfm_artifacts = self.dfm_model.fit_transform(
                self.bundle.monthly_local,
                self.bundle.monthly_national,
            )
            quarterly_factor_frame = dfm_artifacts.quarterly_factor_frame if dfm_artifacts else None

        feature_artifacts = self.feature_engineer.build_training_panel(
            quarterly_target_df=self.bundle.quarterly_target,
            monthly_local_df=self.bundle.monthly_local,
            monthly_national_df=self.bundle.monthly_national,
            quarterly_panel_df=self.bundle.quarterly_panel,
            metadata_df=self.bundle.metadata,
            quarterly_factor_frame=quarterly_factor_frame,
        )
        if dfm_artifacts is not None:
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

    def _apply_target_transform(self, panel: pd.DataFrame) -> tuple[pd.DataFrame, Optional[pd.Series]]:
        """将 level 目标面板转为训练/回测使用的目标空间。

        统一 train() 与 backtest() 的目标变换，避免两者目标空间不一致。

        Returns:
            (panel, base_series)：
            - delta 变换：target_value 变为 Δy_t = y_t - y_{t-1}，丢弃无差分的第一行；
              base_series 记录每行对应的 y_{t-1}，用于预测后加回 level。
            - level 变换：面板原样返回，base_series 为 None。
        """
        assert self.feature_artifacts is not None
        target_col = self.feature_artifacts.target_column
        panel = panel.sort_values("quarter_end").reset_index(drop=True).copy()
        if self.config.features.target_transform == "delta":
            y_raw = panel[target_col].astype(float).copy()
            base_series = y_raw.shift(1).iloc[1:]
            delta_y = y_raw.diff().iloc[1:]
            panel = panel.iloc[1:].copy()
            panel[target_col] = delta_y.values
            base_series.index = panel.index
            return panel, base_series
        return panel, None

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

        # 统一目标变换：delta / level，与 backtest() 保持一致
        # Delta parameterization: 将 level 目标转为差分序列用于训练
        panel, _ = self._apply_target_transform(panel)

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

        # 统一目标变换：模型在 delta 空间训练/预测，窗口结束后用 y_{t-1} 加回 level，
        # 使 metrics 的 RMSE 单位仍为"百分点"
        panel, base_series = self._apply_target_transform(panel)

        result = self.backtester.run(
            panel=panel,
            feature_cols=feature_cols,
            target_col=self.feature_artifacts.target_column,
            selected_model_name=self.selected_model_name,
            base_series=base_series,
        )
        result["target_transform"] = self.config.features.target_transform
        self.backtest_result = result
        step.close("completed", result)
        return result

    def predict_next(self) -> Dict[str, Any]:
        self.initialize()
        if self.feature_artifacts is None:
            self.build_features()
        if self.selected_model is None:
            self.train()

        assert self.feature_artifacts is not None
        assert self.selected_model is not None
        panel = self.feature_artifacts.training_panel.copy()
        feature_cols = self.feature_artifacts.feature_columns
        panel = panel.sort_values("quarter_end").reset_index(drop=True)
        latest_row = panel.tail(1).copy()
        nowcast_quarter = pd.to_datetime(latest_row["quarter_end"].iloc[0])

        # Raw model prediction (delta space if target_transform='delta')
        raw_pred = float(self.selected_model.predict(latest_row[feature_cols])[0])

        # Delta add-back: y_hat = y_{t-1} + delta_hat
        # 训练面板中 X_t 与 y_t 同属第 t 季度（nowcast 映射，无 h=1 错位），
        # 因此差分基准应为 y_{t-1}（panel 倒数第二行的 target_value），而非 y_t。
        prediction_value = raw_pred
        delta_add_back_applied = False
        if self.config.features.target_transform == "delta":
            if len(panel) >= 2:
                y_t_minus_1 = float(panel.iloc[-2]["target_value"])
                prediction_value = y_t_minus_1 + raw_pred
                delta_add_back_applied = True
            else:
                prediction_value = raw_pred

        # Chronos residual correction (lazy load on first call)
        chronos_correction = 0.0
        if self._chronos_state == "not_loaded":
            try:
                from .chronos_adapter import ChronosResidualCorrector
                self._chronos_corrector = ChronosResidualCorrector("amazon/chronos-bolt-tiny")
                self._chronos_state = "ready" if not self._chronos_corrector.failed else "failed"
            except Exception:
                self._chronos_state = "failed"
        if self._chronos_state == "ready" and self._chronos_corrector is not None:
            try:
                # Use training residuals as context
                if hasattr(self.selected_model, "train_residuals_") and self.selected_model.train_residuals_ is not None:
                    residuals = self.selected_model.train_residuals_
                    e_hat, _ = self._chronos_corrector.correct(residuals)
                    chronos_correction = float(e_hat)
                    prediction_value += chronos_correction
            except Exception:
                pass

        actual_value = float(latest_row["target_value"].iloc[0])
        notes = [
            "prediction_generated_from_latest_available_quarter_features",
            "if_real_data_is_short_treat_as_demo_nowcast_not_production_forecast",
            "本预测为对最新已发布季度的 nowcast 复现，用于验证模型在真实月度信息集下的还原能力",
        ]
        if self.config.features.target_transform == "delta" and not delta_add_back_applied:
            notes.append("样本不足，未进行 delta 差分加回，输出为模型原始预测")
        if not self.backtest_result:
            notes.append("未运行回测，不提供置信区间")

        result = {
            "target_indicator": self.config.features.target_indicator,
            "nowcast_quarter": pretty_quarter(nowcast_quarter),
            "prediction_quarter": pretty_quarter(nowcast_quarter),  # 别名，兼容旧字段
            "based_on_latest_quarter": pretty_quarter(nowcast_quarter),
            "model_name": self.selected_model.model_name,
            "prediction_value": prediction_value,
            "actual_value": actual_value,
            "nowcast_error": prediction_value - actual_value,
            "benchmark_value": actual_value,
            "confidence_interval": self._build_confidence_interval(prediction_value),
            "top_features": self.selected_model.get_feature_importance(10),
            "chronos_correction": chronos_correction,
            "chronos_state": self._chronos_state,
            "target_transform": self.config.features.target_transform,
            "notes": notes,
        }
        if hasattr(self.selected_model, "predict_components"):
            comp = self.selected_model.predict_components(latest_row[feature_cols])
            result["components"] = {
                "linear_prediction": float(comp["linear_prediction"][0]),
                "nonlinear_correction": float(comp["nonlinear_correction"][0]),
                "final_prediction": float(comp["final_prediction"][0]),
            }
        else:
            result["components"] = {"prediction": raw_pred}

        self.latest_prediction = result
        return result

    def _build_confidence_interval(self, point_pred: float) -> Optional[Dict[str, Any]]:
        # 1.64 * backtest_RMSE ≈ 90% empirical interval
        # 未运行回测时不提供置信区间（返回 None），不再使用硬编码的 RMSE 魔数
        if not self.backtest_result or not self.backtest_result.get("metrics"):
            return None
        backtest_rmse = float(self.backtest_result["metrics"].get("rmse"))
        if backtest_rmse <= 0:
            return None
        spread = max(0.3, 1.64 * backtest_rmse)
        return {
            "lower": round(point_pred - spread, 2),
            "upper": round(point_pred + spread, 2),
            "method": "point ± 1.64 × backtest_RMSE (approx 90% empirical CI)",
            "backtest_rmse": backtest_rmse,
        }

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
