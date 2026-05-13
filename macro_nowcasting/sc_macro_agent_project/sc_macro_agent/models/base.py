"""
模型基类。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class ModelResult:
    model_name: str
    predictions: np.ndarray
    components: Dict[str, np.ndarray]
    feature_importance: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["predictions"] = self.predictions.tolist()
        payload["components"] = {k: v.tolist() for k, v in self.components.items()}
        return payload


class BaseForecastModel:
    model_name: str = "base"

    def __init__(self) -> None:
        self.is_fitted = False
        self.feature_names: List[str] = []
        self.train_predictions_: Optional[np.ndarray] = None
        self.train_residuals_: Optional[np.ndarray] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseForecastModel":
        raise NotImplementedError

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_result(self, X: pd.DataFrame) -> ModelResult:
        pred = self.predict(X)
        return ModelResult(
            model_name=self.model_name,
            predictions=np.asarray(pred, dtype=float),
            components={"prediction": np.asarray(pred, dtype=float)},
            feature_importance=self.get_feature_importance(),
            metadata=self.get_summary(),
        )

    def get_feature_importance(self, top_n: int = 20) -> List[Dict[str, Any]]:
        return []

    def get_summary(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "is_fitted": self.is_fitted}
