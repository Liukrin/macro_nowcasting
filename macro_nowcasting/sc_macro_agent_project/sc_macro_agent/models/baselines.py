"""
一组小样本友好 baseline。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .base import BaseForecastModel


class LastValueModel(BaseForecastModel):
    model_name = "last_value"

    def __init__(self) -> None:
        super().__init__()
        self.last_value_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LastValueModel":
        ser = pd.Series(y).dropna()
        self.last_value_ = float(ser.iloc[-1]) if not ser.empty else 0.0
        self.train_predictions_ = np.repeat(self.last_value_, len(y))
        self.train_residuals_ = y.to_numpy() - self.train_predictions_
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.repeat(self.last_value_, len(X))

    def get_summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "is_fitted": self.is_fitted,
            "last_value": self.last_value_,
        }


class MeanRecentModel(BaseForecastModel):
    model_name = "mean_recent"

    def __init__(self, window: int = 4) -> None:
        super().__init__()
        self.window = window
        self.mean_value_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanRecentModel":
        ser = pd.Series(y).dropna()
        tail = ser.tail(self.window)
        self.mean_value_ = float(tail.mean()) if not tail.empty else 0.0
        self.train_predictions_ = np.repeat(self.mean_value_, len(y))
        self.train_residuals_ = y.to_numpy() - self.train_predictions_
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.repeat(self.mean_value_, len(X))

    def get_summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "is_fitted": self.is_fitted,
            "window": self.window,
            "mean_value": self.mean_value_,
        }
