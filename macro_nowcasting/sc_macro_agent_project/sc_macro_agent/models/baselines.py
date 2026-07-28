"""
一组小样本友好 baseline。
"""
from __future__ import annotations

from typing import Any, Dict
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


class SeasonalNaiveModel(BaseForecastModel):
    """预测值 = y_{t-4}（去年同季）。
    这是季节性数据的标准基准，对 YTD 累计同比尤其合适。"""

    model_name = "seasonal_naive"

    def __init__(self) -> None:
        super().__init__()
        self.last_seasonal_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaiveModel":
        ser = pd.Series(y).dropna()
        # y_{t-4}: 需要至少 5 个观测才能取到去年同季
        if len(ser) >= 5:
            self.last_seasonal_ = float(ser.iloc[-5])
        elif len(ser) >= 1:
            self.last_seasonal_ = float(ser.iloc[-1])
        else:
            self.last_seasonal_ = 0.0
        self.train_predictions_ = np.repeat(self.last_seasonal_, len(y))
        self.train_residuals_ = y.to_numpy() - self.train_predictions_
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.repeat(self.last_seasonal_, len(X))

    def get_summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "is_fitted": self.is_fitted,
            "last_seasonal": self.last_seasonal_,
        }


class DriftModel(BaseForecastModel):
    """预测值 = y_{t-4} + (y_{t-1} - y_{t-5})。
    去年同季 + 近期动量（相邻两季之差的延续）。"""

    model_name = "drift"

    def __init__(self) -> None:
        super().__init__()
        self.prediction_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "DriftModel":
        ser = pd.Series(y).dropna()
        if len(ser) >= 6:
            seasonal = float(ser.iloc[-5])  # y_{t-4}：索引 -5 = t-4（含当前）
            # y_{t-1} - y_{t-5} 是最近一季的环比动量
            # 索引：-1 = y_t, -2 = y_{t-1}, ..., -5 = y_{t-4}, -6 = y_{t-5}
            recent_diff = float(ser.iloc[-2]) - float(ser.iloc[-6])
            self.prediction_ = seasonal + recent_diff
        elif len(ser) >= 1:
            self.prediction_ = float(ser.iloc[-1])
        else:
            self.prediction_ = 0.0
        self.train_predictions_ = np.repeat(self.prediction_, len(y))
        self.train_residuals_ = y.to_numpy() - self.train_predictions_
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.repeat(self.prediction_, len(X))

    def get_summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "is_fitted": self.is_fitted,
            "prediction": self.prediction_,
        }
