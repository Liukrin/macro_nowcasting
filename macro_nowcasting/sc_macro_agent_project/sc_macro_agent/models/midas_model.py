"""
MIDAS-style linear model collection.

Quarterly mixed-frequency feature regression.
Friendly for mid-term reporting and engineering deployment.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler

from .base import BaseForecastModel

# ElasticNetCV with small samples may not fully converge; results are still usable
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class _LinearMIDASBase(BaseForecastModel):
    estimator_name = "linear_base"

    def __init__(self) -> None:
        super().__init__()
        self.scaler = StandardScaler()
        self.model = None
        self.coef_: Optional[pd.Series] = None
        self.intercept_: float = 0.0

    @property
    def model_name(self) -> str:
        return self.estimator_name

    def _fit_model(self, Xs: np.ndarray, y: pd.Series) -> None:
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_LinearMIDASBase":
        self.feature_names = list(X.columns)
        Xs = self.scaler.fit_transform(X.to_numpy(dtype=float))
        self._fit_model(Xs, y)
        pred = self.model.predict(Xs)
        self.train_predictions_ = pred
        self.train_residuals_ = y.to_numpy(dtype=float) - pred
        self.intercept_ = float(getattr(self.model, "intercept_", 0.0))
        coef = getattr(self.model, "coef_", np.zeros(len(self.feature_names)))
        self.coef_ = pd.Series(np.asarray(coef, dtype=float), index=self.feature_names)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.scaler.transform(X.to_numpy(dtype=float))
        return np.asarray(self.model.predict(Xs), dtype=float)

    def get_feature_importance(self, top_n: int = 20) -> List[Dict[str, Any]]:
        if self.coef_ is None:
            return []
        top = self.coef_.abs().sort_values(ascending=False).head(top_n)
        total = float(top.sum()) or 1.0
        return [
            {
                "feature": name,
                "coefficient": float(self.coef_.loc[name]),
                "abs_coefficient": float(val),
                "normalized_importance": float(val / total),
            }
            for name, val in top.items()
        ]

    def get_summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "is_fitted": self.is_fitted,
            "intercept": self.intercept_,
            "feature_count": len(self.feature_names),
            "top_features": self.get_feature_importance(10),
        }


class RidgeMIDASModel(_LinearMIDASBase):
    estimator_name = "ridge_midas"

    def __init__(self, alphas: Optional[List[float]] = None) -> None:
        super().__init__()
        self.alphas = alphas or [0.01, 0.1, 1.0, 10.0, 100.0]

    def _fit_model(self, Xs: np.ndarray, y: pd.Series) -> None:
        self.model = RidgeCV(alphas=self.alphas)
        self.model.fit(Xs, y)


class ElasticMIDASModel(_LinearMIDASBase):
    estimator_name = "elastic_midas"

    def __init__(self, alphas: Optional[List[float]] = None, l1_ratios: Optional[List[float]] = None, random_state: int = 42) -> None:
        super().__init__()
        self.alphas = alphas or [0.001, 0.01, 0.1, 1.0]
        self.l1_ratios = l1_ratios or [0.1, 0.3, 0.5, 0.7, 0.9]
        self.random_state = random_state

    def _fit_model(self, Xs: np.ndarray, y: pd.Series) -> None:
        self.model = ElasticNetCV(
            alphas=self.alphas,
            l1_ratio=self.l1_ratios,
            cv=min(5, max(3, len(y) - 1)),
            random_state=self.random_state,
            max_iter=10000,
            tol=1e-4,
        )
        self.model.fit(Xs, y)
