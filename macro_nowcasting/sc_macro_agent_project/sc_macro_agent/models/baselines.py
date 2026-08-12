"""
一组小样本友好 baseline。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from .base import BaseForecastModel
from ..exceptions import ModelTrainingError


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


class ARIMABaselineModel(BaseForecastModel):
    """单变量 ARIMA 基准模型。

    完全忽略外生变量 X，仅使用 y 序列。
    statsmodels 在 fit() 内部 lazy import，构造期不触发依赖检查，
    确保 create_candidates() 在 select_best 的 try 之外调用时不会因
    缺少 statsmodels 而崩溃。
    """

    model_name = "arima"

    def __init__(self) -> None:
        super().__init__()
        self._fitted_result: Any = None
        self._order: Tuple[int, int, int] = (0, 0, 0)
        self._aic: Optional[float] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ARIMABaselineModel":
        ser = pd.Series(y).dropna().astype(float)
        if len(ser) < 5:
            # 样本太小，直接用均值预测
            self._order = (0, 0, 0)
            self._aic = None
            mean_val = float(ser.mean())
            self.train_predictions_ = np.repeat(mean_val, len(y))
            self.train_residuals_ = y.to_numpy() - self.train_predictions_
            self.is_fitted = True
            return self

        # Lazy import：statsmodels 不是硬依赖，缺失时由上层 try/except 捕获降级
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError as exc:
            raise ModelTrainingError(
                "statsmodels 未安装，无法使用 ARIMA 基准模型。"
                " 请执行: pip install statsmodels>=0.14"
            ) from exc

        # 在小网格上按 AIC 选优
        best_order: Tuple[int, int, int] = (1, 0, 0)
        best_aic: Optional[float] = None
        best_result = None

        for p in (0, 1, 2):
            for d in (0, 1):
                for q in (0, 1):
                    try:
                        model = ARIMA(ser.values, order=(p, d, q))
                        result = model.fit()
                        aic_val = float(result.aic)
                        if best_aic is None or aic_val < best_aic:
                            best_aic = aic_val
                            best_order = (p, d, q)
                            best_result = result
                    except Exception:
                        continue

        self._order = best_order
        self._aic = best_aic

        # 若网格搜索全部失败，回退到 ARIMA(1,0,0)
        if best_result is None:
            try:
                model = ARIMA(ser.values, order=(1, 0, 0))
                best_result = model.fit()
                self._aic = float(best_result.aic)
            except Exception as exc:
                raise ModelTrainingError(
                    f"ARIMA 基准模型拟合失败: {exc}"
                ) from exc

        self._fitted_result = best_result

        # 计算训练集上的预测和残差（hybrid 模型依赖这两个属性）
        try:
            fitted_values = best_result.fittedvalues
            # fittedvalues 的第一个元素可能为 NaN（取决于差分阶数）
            self.train_predictions_ = np.asarray(
                pd.Series(fitted_values).bfill().values,
                dtype=float,
            )
            self.train_residuals_ = ser.values - self.train_predictions_
        except Exception:
            # 极端情况：直接回退为均值
            mean_val = float(ser.mean())
            self.train_predictions_ = np.repeat(mean_val, len(y))
            self.train_residuals_ = y.to_numpy() - self.train_predictions_

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._fitted_result is None:
            # 模型使用了均值回退，用均值预测
            if self.train_predictions_ is not None:
                mean_val = float(np.mean(self.train_predictions_))
            else:
                mean_val = 0.0
            return np.repeat(mean_val, len(X)).astype(float)
        forecast = self._fitted_result.forecast(len(X))
        return np.asarray(forecast, dtype=float)

    def get_summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "is_fitted": self.is_fitted,
            "order": list(self._order),
            "aic": self._aic,
        }
