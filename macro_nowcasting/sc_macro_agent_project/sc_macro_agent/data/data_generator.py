"""
合成数据生成器

目的不是追求经济学真实性到论文级,
而是给工程链路提供更长、更稳定、可回测的开发数据源
当真实数据历史太短时，可以切到 demo / hybrid 模式
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class GeneratorConfig:
    start_date: str = "2020-01-01"
    end_date: str = "2026-03-31"
    random_state: int = 42
    province: str = "四川省"

    def monthly_dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start_date, self.end_date, freq="M")

    def quarterly_dates(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start_date, self.end_date, freq="Q")


class MacroDataGenerator:
    def __init__(self, config: Optional[GeneratorConfig] = None) -> None:
        self.config = config or GeneratorConfig()
        self.rng = np.random.default_rng(self.config.random_state)

    def _base_cycles(self, n: int) -> Dict[str, np.ndarray]:
        t = np.arange(n)
        demand = 0.8 * np.sin(2 * np.pi * t / 12) + 0.15 * np.sin(2 * np.pi * t / 36)
        production = 0.7 * np.sin(2 * np.pi * t / 10 + 0.4)
        price = 0.5 * np.sin(2 * np.pi * t / 14 + 1.1)
        finance = 0.6 * np.sin(2 * np.pi * t / 16 + 2.1)
        trend = np.linspace(0, 1.2, n)
        return {
            "demand": demand + trend,
            "production": production + 0.6 * trend,
            "price": price + 0.2 * trend,
            "finance": finance + 0.15 * trend,
        }

    def generate_monthly_national(self) -> pd.DataFrame:
        dates = self.config.monthly_dates()
        n = len(dates)
        cycles = self._base_cycles(n)
        noise = lambda scale: self.rng.normal(0, scale, n)

        values = {
            "PMI_PMI": 50 + 1.6 * cycles["production"] + 1.1 * cycles["demand"] + noise(0.35),
            "PMI_生产": 50 + 1.8 * cycles["production"] + noise(0.35),
            "PMI_新订单": 50 + 1.6 * cycles["demand"] + noise(0.35),
            "规模以上工业增加值_同比增速": 5 + 1.8 * cycles["production"] + noise(0.5),
            "固定资产投资（不含农户）_同比增速": 4.5 + 1.5 * cycles["finance"] + 0.8 * cycles["production"] + noise(0.6),
            "社会消费品零售总额_同比增速": 4.2 + 1.7 * cycles["demand"] + noise(0.5),
            "CPI_同比": 1.5 + 0.8 * cycles["price"] + noise(0.2),
            "PPI_同比": 0.8 + 1.3 * cycles["price"] + noise(0.25),
            "M2_同比增速": 8.0 + 1.0 * cycles["finance"] + noise(0.25),
            "全社会用电量_同比增速": 5.5 + 1.6 * cycles["production"] + noise(0.5),
            "出口_同比增速": 3.0 + 1.0 * cycles["demand"] + 0.8 * cycles["production"] + noise(0.8),
        }

        rows: List[Dict[str, Any]] = []
        for date in dates:
            idx = int(np.where(dates == date)[0][0])
            for indicator_name, arr in values.items():
                rows.append({
                    "date": date,
                    "region": "全国",
                    "indicator_name": indicator_name,
                    "indicator_value": float(arr[idx]),
                    "frequency": "monthly",
                    "unit": "%",
                    "source_name": "SyntheticGenerator",
                    "source_url": "-",
                    "note": "synthetic national monthly feature",
                })
        return pd.DataFrame(rows)

    def generate_monthly_local(self) -> pd.DataFrame:
        dates = self.config.monthly_dates()
        n = len(dates)
        cycles = self._base_cycles(n)
        noise = lambda scale: self.rng.normal(0, scale, n)

        values = {
            "规模以上工业增加值_同比增速": 5.2 + 1.9 * cycles["production"] + noise(0.45),
            "固定资产投资（不含农户）_同比增速": 5.0 + 1.8 * cycles["finance"] + 0.5 * cycles["production"] + noise(0.55),
            "社会消费品零售总额_同比增速": 4.5 + 1.9 * cycles["demand"] + noise(0.45),
            "CPI_同比": 1.3 + 0.7 * cycles["price"] + noise(0.2),
            "工业投资_同比增速": 4.8 + 1.6 * cycles["finance"] + 1.1 * cycles["production"] + noise(0.5),
            "房地产开发投资_同比增速": 2.0 + 0.8 * cycles["finance"] - 0.5 * cycles["demand"] + noise(0.7),
        }

        rows: List[Dict[str, Any]] = []
        for date in dates:
            idx = int(np.where(dates == date)[0][0])
            for indicator_name, arr in values.items():
                rows.append({
                    "date": date,
                    "region": self.config.province,
                    "indicator_name": indicator_name,
                    "indicator_value": float(arr[idx]),
                    "frequency": "monthly",
                    "unit": "%",
                    "source_name": "SyntheticGenerator",
                    "source_url": "-",
                    "note": "synthetic local monthly feature",
                })
        return pd.DataFrame(rows)

    def generate_quarterly_target(self) -> pd.DataFrame:
        q_dates = self.config.quarterly_dates()
        monthly_national = self.generate_monthly_national()
        monthly_local = self.generate_monthly_local()

        nat = monthly_national.pivot_table(index="date", columns="indicator_name", values="indicator_value", aggfunc="last")
        loc = monthly_local.pivot_table(index="date", columns="indicator_name", values="indicator_value", aggfunc="last")

        nat["quarter_end"] = nat.index.to_period("Q").to_timestamp("Q")
        loc["quarter_end"] = loc.index.to_period("Q").to_timestamp("Q")

        nat_q = nat.groupby("quarter_end").mean(numeric_only=True)
        loc_q = loc.groupby("quarter_end").mean(numeric_only=True)

        common_q = nat_q.index.intersection(loc_q.index)
        nat_q = nat_q.loc[common_q]
        loc_q = loc_q.loc[common_q]

        gdp = (
            0.32 * loc_q["规模以上工业增加值_同比增速"]
            + 0.28 * loc_q["固定资产投资（不含农户）_同比增速"]
            + 0.18 * nat_q["PMI_PMI"]
            + 0.16 * loc_q["社会消费品零售总额_同比增速"]
            + 0.06 * nat_q["M2_同比增速"]
        )
        gdp = (gdp - gdp.mean()) / max(gdp.std(), 1e-6) * 1.2 + 6.0

        rows: List[Dict[str, Any]] = []
        for q_end, gdp_val in gdp.items():
            rows.append({
                "date": q_end,
                "region": self.config.province,
                "indicator_name": "GDP_同比增速",
                "indicator_value": float(gdp_val),
                "frequency": "quarterly",
                "unit": "%",
                "source_name": "SyntheticGenerator",
                "source_url": "-",
                "note": "synthetic quarterly GDP yoy",
            })
            rows.append({
                "date": q_end,
                "region": self.config.province,
                "indicator_name": "第二产业增加值_同比增速",
                "indicator_value": float(gdp_val + self.rng.normal(0.2, 0.25)),
                "frequency": "quarterly",
                "unit": "%",
                "source_name": "SyntheticGenerator",
                "source_url": "-",
                "note": "synthetic sector target",
            })
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    def generate_quarterly_panel(self) -> pd.DataFrame:
        monthly_local = self.generate_monthly_local()
        monthly_national = self.generate_monthly_national()
        df = pd.concat([monthly_local, monthly_national], ignore_index=True)
        df["quarter_end"] = pd.to_datetime(df["date"]).dt.to_period("Q").dt.to_timestamp("Q")
        rows: List[Dict[str, Any]] = []

        for (region, indicator_name, q_end), grp in df.groupby(["region", "indicator_name", "quarter_end"]):
            grp = grp.sort_values("date")
            vals = grp["indicator_value"].astype(float)
            mapping = {
                "mean": float(vals.mean()),
                "last": float(vals.iloc[-1]),
                "std": float(vals.std()) if len(vals) > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
                "trend": float(np.polyfit(np.arange(len(vals)), vals.to_numpy(), 1)[0]) if len(vals) > 1 else 0.0,
            }
            for suffix, v in mapping.items():
                rows.append({
                    "date": q_end,
                    "region": region,
                    "indicator_name": f"{indicator_name}_{suffix}",
                    "indicator_value": float(v),
                    "frequency": "quarterly",
                    "unit": "%",
                    "source_name": "SyntheticGenerator",
                    "source_url": "-",
                    "note": "synthetic quarterly aggregated feature",
                })
        return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    def generate_metadata(self) -> pd.DataFrame:
        rows = [
            ["PMI_PMI", "PMI", "PMI", "monthly", False, True, False, True, "high", "synthetic"],
            ["PMI_生产", "PMI_Production", "PMI", "monthly", False, True, False, True, "high", "synthetic"],
            ["PMI_新订单", "PMI_NewOrders", "PMI", "monthly", False, True, False, True, "high", "synthetic"],
            ["规模以上工业增加值_同比增速", "IndustrialOutput", "工业", "monthly", False, True, False, True, "high", "synthetic"],
            ["固定资产投资（不含农户）_同比增速", "FAI", "投资", "monthly", False, True, False, True, "high", "synthetic"],
            ["社会消费品零售总额_同比增速", "RetailSales", "消费", "monthly", False, True, False, True, "high", "synthetic"],
            ["GDP_同比增速", "GDP_YoY", "目标", "quarterly", True, True, False, False, "high", "synthetic"],
        ]
        return pd.DataFrame(rows, columns=[
            "original_name", "standard_name", "category", "frequency",
            "is_target", "is_yoy", "is_cumulative", "leading_indicator",
            "data_quality", "notes"
        ])

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        return {
            "quarterly_target": self.generate_quarterly_target(),
            "monthly_local": self.generate_monthly_local(),
            "monthly_national": self.generate_monthly_national(),
            "quarterly_panel": self.generate_quarterly_panel(),
            "metadata": self.generate_metadata(),
        }
