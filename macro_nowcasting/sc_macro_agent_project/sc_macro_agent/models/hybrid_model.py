"""
混合模型：线性 MIDAS + 残差树模型

小样本下如果树模型不稳，会自动退化为线性模型输出
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .base import BaseForecastModel, ModelResult
from .midas_model import RidgeMIDASModel


class HybridResidualModel(BaseForecastModel):
    model_name = "hybrid_residual"

    def __init__(
        self,
        ridge_alphas: Optional[List[float]] = None,
        random_state: int = 42,
        n_estimators: int = 200,
        max_depth: int = 3,
        min_samples_leaf: int = 2,
        min_train_rows_for_tree: int = 12,
    ) -> None:
        super().__init__()
        self.linear_model = RidgeMIDASModel(alphas=ridge_alphas)
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_train_rows_for_tree = min_train_rows_for_tree
        self.tree = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        self.use_tree_: bool = False
        self.residual_feature_importance_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HybridResidualModel":
        self.feature_names = list(X.columns)
        self.linear_model.fit(X, y)
        linear_pred = self.linear_model.predict(X)
        residuals = y.to_numpy(dtype=float) - linear_pred

        if len(X) >= self.min_train_rows_for_tree:
            self.tree.fit(X, residuals)
            self.use_tree_ = True
            self.residual_feature_importance_ = pd.Series(
                self.tree.feature_importances_,
                index=self.feature_names,
            )
            nonlinear = self.tree.predict(X)
        else:
            self.use_tree_ = False
            self.residual_feature_importance_ = pd.Series(
                np.zeros(len(self.feature_names)),
                index=self.feature_names,
            )
            nonlinear = np.zeros(len(X))

        final_pred = linear_pred + nonlinear
        self.train_predictions_ = final_pred
        self.train_residuals_ = y.to_numpy(dtype=float) - final_pred
        self.is_fitted = True
        return self

    def predict_components(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        linear_pred = self.linear_model.predict(X)
        nonlinear_pred = self.tree.predict(X) if self.use_tree_ else np.zeros(len(X))
        final_pred = linear_pred + nonlinear_pred
        return {
            "linear_prediction": np.asarray(linear_pred, dtype=float),
            "nonlinear_correction": np.asarray(nonlinear_pred, dtype=float),
            "final_prediction": np.asarray(final_pred, dtype=float),
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_components(X)["final_prediction"]

    def predict_result(self, X: pd.DataFrame) -> ModelResult:
        comp = self.predict_components(X)
        return ModelResult(
            model_name=self.model_name,
            predictions=comp["final_prediction"],
            components=comp,
            feature_importance=self.get_feature_importance(),
            metadata=self.get_summary(),
        )

    def get_feature_importance(self, top_n: int = 20) -> List[Dict[str, Any]]:
        linear = pd.Series(dtype=float)
        if self.linear_model.coef_ is not None:
            linear = self.linear_model.coef_.abs()

        nonlinear = self.residual_feature_importance_
        if nonlinear is None:
            nonlinear = pd.Series(np.zeros(len(self.feature_names)), index=self.feature_names)

        if linear.empty:
            linear = pd.Series(np.zeros(len(self.feature_names)), index=self.feature_names)

        merged = pd.concat(
            [
                linear.rename("linear_abs_coef"),
                nonlinear.rename("residual_tree_importance"),
            ],
            axis=1,
        ).fillna(0.0)
        merged["combined_score"] = merged["linear_abs_coef"] + merged["residual_tree_importance"]
        merged = merged.sort_values("combined_score", ascending=False).head(top_n)

        return merged.reset_index().rename(columns={"index": "feature"}).to_dict("records")

    def get_summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "is_fitted": self.is_fitted,
            "use_tree": self.use_tree_,
            "linear_summary": self.linear_model.get_summary(),
            "top_features": self.get_feature_importance(10),
        }
