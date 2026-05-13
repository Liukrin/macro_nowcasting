"""
模型选择器

做法比较朴素但实用：
- 接收候选模型列表
- 在给定训练/验证切分上逐个 fit + evaluate
- 返回最佳模型和完整 leaderboard
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..config import ModelConfig
from ..logging_utils import get_logger
from ..utils import metrics_dict
from .baselines import LastValueModel, MeanRecentModel
from .midas_model import RidgeMIDASModel, ElasticMIDASModel
from .hybrid_model import HybridResidualModel
from ..exceptions import ModelTrainingError


@dataclass
class LeaderboardEntry:
    model_name: str
    rmse: float
    mae: float
    mape: float
    smape: float
    r2: float
    n_train: int
    n_valid: int
    notes: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelFactory:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def create(self, name: str):
        if name == "last_value":
            return LastValueModel()
        if name == "mean_recent":
            return MeanRecentModel(window=4)
        if name == "ridge_midas":
            return RidgeMIDASModel(alphas=self.config.ridge_alphas)
        if name == "elastic_midas":
            return ElasticMIDASModel(
                alphas=self.config.elastic_alphas,
                l1_ratios=self.config.elastic_l1_ratios,
                random_state=self.config.random_state,
            )
        if name == "hybrid_residual":
            return HybridResidualModel(
                ridge_alphas=self.config.ridge_alphas,
                random_state=self.config.random_state,
                n_estimators=self.config.residual_n_estimators,
                max_depth=self.config.residual_max_depth,
                min_samples_leaf=self.config.residual_min_samples_leaf,
                min_train_rows_for_tree=self.config.min_train_rows_for_tree,
            )
        raise ModelTrainingError(f"未知模型: {name}")

    def create_candidates(self):
        return [self.create(name) for name in self.config.candidate_models]


class ModelSelector:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.logger = get_logger("sc_macro_agent.model_selection")
        self.factory = ModelFactory(config)

    def _fit_and_score(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> LeaderboardEntry:
        notes: List[str] = []
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        m = metrics_dict(y_valid, pred)
        if hasattr(model, "use_tree_") and not getattr(model, "use_tree_", True):
            notes.append("tree_disabled_due_to_small_sample")
        return LeaderboardEntry(
            model_name=model.model_name,
            rmse=m["rmse"],
            mae=m["mae"],
            mape=m["mape"],
            smape=m["smape"],
            r2=m["r2"],
            n_train=int(len(X_train)),
            n_valid=int(len(X_valid)),
            notes=notes,
        )

    def select_best(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        candidates = self.factory.create_candidates()
        leaderboard: List[LeaderboardEntry] = []

        for model in candidates:
            try:
                entry = self._fit_and_score(model, X_train, y_train, X_valid, y_valid)
                leaderboard.append(entry)
            except Exception as exc:
                self.logger.warning("模型评估失败 | model=%s | err=%s", model.model_name, exc)

        if not leaderboard:
            raise ModelTrainingError("所有候选模型均训练失败")

        leaderboard.sort(key=lambda x: (x.rmse, x.mae))
        best_name = leaderboard[0].model_name
        best_model = self.factory.create(best_name)
        best_model.fit(pd.concat([X_train, X_valid]), pd.concat([y_train, y_valid]))
        return best_model, [item.as_dict() for item in leaderboard]
