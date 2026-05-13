"""
特征注册表

主要做两件事：
1. 给特征打标签，方便回头解释它来自哪里
2. 让报告层可以根据特征来源做聚合展示
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any


@dataclass
class FeatureMeta:
    name: str
    source_table: str
    source_indicator: Optional[str]
    region: Optional[str]
    family: str
    transform: str
    frequency: str
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeatureRegistry:
    def __init__(self) -> None:
        self._store: Dict[str, FeatureMeta] = {}

    def register(
        self,
        name: str,
        source_table: str,
        source_indicator: Optional[str],
        region: Optional[str],
        family: str,
        transform: str,
        frequency: str,
        note: str = "",
    ) -> None:
        self._store[name] = FeatureMeta(
            name=name,
            source_table=source_table,
            source_indicator=source_indicator,
            region=region,
            family=family,
            transform=transform,
            frequency=frequency,
            note=note,
        )

    def get(self, name: str) -> Optional[FeatureMeta]:
        return self._store.get(name)

    def to_frame(self):
        import pandas as pd
        if not self._store:
            return pd.DataFrame(columns=[
                "name", "source_table", "source_indicator", "region",
                "family", "transform", "frequency", "note"
            ])
        return pd.DataFrame([v.as_dict() for v in self._store.values()])

    def summary_by_family(self) -> Dict[str, int]:
        counter: Dict[str, int] = {}
        for meta in self._store.values():
            counter[meta.family] = counter.get(meta.family, 0) + 1
        return dict(sorted(counter.items(), key=lambda kv: kv[0]))

    def summary_by_region(self) -> Dict[str, int]:
        counter: Dict[str, int] = {}
        for meta in self._store.values():
            region = meta.region or "unknown"
            counter[region] = counter.get(region, 0) + 1
        return dict(sorted(counter.items(), key=lambda kv: kv[0]))

    def names(self) -> List[str]:
        return list(self._store.keys())


class ExplanationBuilder:
    def annotate_features(self, feature_importance: List[Dict[str, Any]], registry: FeatureRegistry) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for item in feature_importance:
            feature = item.get("feature")
            meta = registry.get(feature) if feature else None
            row = dict(item)
            if meta:
                row.update({
                    "family": meta.family,
                    "region": meta.region,
                    "source_table": meta.source_table,
                    "source_indicator": meta.source_indicator,
                    "transform": meta.transform,
                })
            rows.append(row)
        return rows

    def family_contribution_proxy(self, annotated_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not annotated_rows:
            return []
        df = pd.DataFrame(annotated_rows)
        score_col = None
        for candidate in ["combined_score", "abs_coefficient", "normalized_importance", "residual_tree_importance"]:
            if candidate in df.columns:
                score_col = candidate
                break
        if score_col is None:
            return []
        grp = df.groupby("family", dropna=False)[score_col].sum().sort_values(ascending=False).reset_index()
        return grp.rename(columns={score_col: "proxy_contribution"}).to_dict("records")
