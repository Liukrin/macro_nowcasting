"""
轻量 RAG 服务：TF-IDF 检索 + LLM 问答。
语料规模小（~200条），TF-IDF 足够有效，无需 GPU 嵌入模型。
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .llm.client import LLMClient
from .logging_utils import get_logger

# ================================================================
# 中文数字规范化（状态机，正确处理"十一""二十"等）
# ================================================================
_CN_DIGITS = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
_CN_TEN = '十'

# 内置"已知局限"占位文本：artifacts/final/known_limitations.md 缺失时使用
_BUILTIN_LIMITATIONS_TEXT = (
    "# 已知局限\n"
    "## 1. 单季同比口径不可用\n四川GDP累计值序列存在vintage断点，单季同比(qoq_yoy)口径已弃用。\n"
    "## 2. feature_vintage未实现\n当前使用季度完整数据，严格来说为同期回归(contemporaneous)而非严格nowcasting。\n"
    "## 3. target_lag_1未进入特征集\n16特征cap下target_lag_1被挤掉，模型缺少y_{t-1}锚点。\n"
    "## 4. chronos可选依赖\n云端不安装torch时Chronos残差修正自动跳过，不影响主预测。\n"
    "## 5. 样本量限制\n回测窗口数有限，统计检验功效有限。\n"
    "## 6. 数据覆盖\n国家月度数据截至2025-09，四川月度数据截至2025-12，2025Q4 GDP为估算值。\n"
    "## 7. 共享可变状态\nst.cache_resource跨会话共享engine，单人演示可接受，生产需每会话独立实例或加锁。\n"
)


def _normalize_chinese_numbers(text: str) -> str:
    """将中文数字统一为阿拉伯数字。正确处理"十一"→11、"二十"→20、"二十三"→23。"""
    result: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in _CN_DIGITS:
            digit = _CN_DIGITS[ch]
            # 向前看：是否后接"十"
            if i + 1 < len(text) and text[i + 1] == _CN_TEN:
                # [一-九]十[一-九]? 或 [一-九]十
                tens = digit * 10
                if i + 2 < len(text) and text[i + 2] in _CN_DIGITS:
                    unit = _CN_DIGITS[text[i + 2]]
                    result.append(str(tens + unit))
                    i += 3
                else:
                    result.append(str(tens))
                    i += 2
            else:
                result.append(str(digit))
                i += 1
        elif ch == _CN_TEN:
            # 十[一-九]? 或 单独的十
            if i + 1 < len(text) and text[i + 1] in _CN_DIGITS:
                unit = _CN_DIGITS[text[i + 1]]
                result.append(str(10 + unit))
                i += 2
            else:
                result.append('10')
                i += 1
        else:
            result.append(ch)
            i += 1
    return ''.join(result)


# ================================================================
# 查询解析：正则抽取结构化过滤条件
# ================================================================
_YEAR_RE = re.compile(r'(\d{4})\s*年?')


def _extract_year(query: str) -> int | None:
    m = _YEAR_RE.search(query)
    if not m:
        return None
    return int(m.group(1))


def _extract_quarter(query: str) -> int | None:
    """匹配: 第1季度, 第一季度, Q1, 一季度, 三季"""
    m = re.search(r'(?:第\s*)?Q\s*([1-4])', query)
    if m:
        return int(m.group(1))
    m = re.search(r'第\s*([一二三四1-4])\s*季度?', query)
    if m:
        raw = m.group(1)
        cn_map = {'一': 1, '二': 2, '三': 3, '四': 4}
        return cn_map.get(raw, int(raw) if raw.isdigit() else None)
    m = re.search(r'([1-4])\s*季度', query)
    if m:
        return int(m.group(1))
    m = re.search(r'([一二三四])\s*季度?', query)
    if m:
        return _CN_DIGITS.get(m.group(1))
    return None


def _extract_month(query: str) -> int | None:
    m = re.search(r'(\d{1,2})\s*月(?:份)?', _normalize_chinese_numbers(query))
    if m:
        mo = int(m.group(1))
        if 1 <= mo <= 12:
            return mo
    return None


def _extract_query_filters(query: str, entity_set: set[str]) -> dict:
    """从自然语言问题中抽取结构化过滤条件。

    query_keywords: 只含 indicator 名（如"GDP累计同比增速"），不含年份季度数字。
    年份/季度/月份通过正则抽取，单独存放。

    Returns:
        {year: int|None, quarter: int|None, month: int|None,
         indicators: [str], has_time_constraint: bool}
    """
    year = _extract_year(query)
    quarter = _extract_quarter(query)
    month = _extract_month(query)
    has_time = (year is not None) or (quarter is not None) or (month is not None)

    # 关键词：最长匹配优先（indicator 名 + model 名）
    indicators: list[str] = []
    remaining = query
    for kw in sorted(entity_set, key=len, reverse=True):
        if kw and kw in remaining:
            indicators.append(kw)
            remaining = remaining.replace(kw, ' ', 1)

    return {
        "year": year,
        "quarter": quarter,
        "month": month,
        "indicators": indicators,
        "has_time_constraint": has_time,
    }


class RAGService:
    """检索增强生成服务。"""

    def __init__(self, config: "AppConfig", engine: Optional["PredictionEngine"] = None) -> None:
        self.logger = get_logger("sc_macro_agent.rag")
        self.config = config
        self.data_dir = config.data.resolve_dir()
        self.artifacts_dir = config.data.resolve_artifact_dir(create=False) / "final"
        self.llm = LLMClient.get_instance()
        # 复用外部传入的 engine（app 传 load_engine()），不再自行触发流水线
        self.engine = engine
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_vectors: Optional[np.ndarray] = None
        self.documents: List[Dict[str, Any]] = []
        self._indicator_set: set[str] = set()
        self._entity_set: set[str] = set()  # indicator + model 名称，用于关键词匹配
        self._build_corpus()
        self._build_index()

    # ================================================================
    # B.1: 语料构建
    # ================================================================
    def _build_corpus(self) -> None:
        """构建两类语料：指标卡片 + 项目文档。"""
        docs: List[Dict[str, Any]] = []

        # --- 类型一：结构化指标卡片 ---
        docs.extend(self._build_indicator_cards())

        # --- 类型二：项目自述文档 ---
        docs.extend(self._build_project_docs())

        self.documents = docs
        self.logger.info("Corpus built: %d documents", len(docs))

    def _build_indicator_cards(self) -> List[Dict[str, Any]]:
        """从 quarterly_target 和 monthly_local 生成指标卡片。
        按文件来源判定频率标签，月度文件一律用月份标签（不会把 3/6/9/12 月标成季度）。
        """
        cards = []
        for fname, region_label, freq_label in [
            ("quarterly_target_real.csv", "四川省", "quarterly"),
            ("monthly_local_features_real.csv", "四川省", "monthly"),
        ]:
            fp = self.data_dir / fname
            if not fp.exists():
                self.logger.warning("Missing data file: %s", fp)
                continue
            df = pd.read_csv(fp)
            df["date"] = pd.to_datetime(df["date"])
            for ind_name in sorted(df["indicator_name"].unique()):
                sub = df[df["indicator_name"] == ind_name].sort_values("date")
                for _, row in sub.iterrows():
                    dt = row["date"]
                    val = row["indicator_value"]
                    if pd.isna(val):
                        continue
                    # 季度文件用季度标签，月度文件始终用月份标签
                    if freq_label == "quarterly":
                        date_str = f"{dt.year}年第{(dt.month - 1) // 3 + 1}季度"
                        quarter_val = (dt.month - 1) // 3 + 1
                    else:
                        date_str = f"{dt.year}年{dt.month}月"
                        quarter_val = (dt.month - 1) // 3 + 1
                    text = f"{date_str}，{region_label}{ind_name}为{val:.1f}%。数据来源：国家统计局/四川省统计局。"
                    cards.append({
                        "text": text,
                        "metadata": {
                            "type": "indicator_card",
                            "region": region_label,
                            "date": dt.strftime("%Y-%m-%d"),
                            "year": dt.year,
                            "month": dt.month,
                            "quarter": quarter_val,
                            "indicator": ind_name,
                            "value": float(val),
                            "source_file": fname,
                        }
                    })
                    self._indicator_set.add(ind_name)
                    self._entity_set.add(ind_name)
        return cards

    def _build_project_docs(self) -> List[Dict[str, Any]]:
        """从 artifacts/final 文件 + engine 实时状态构建项目文档片段。

        文件存在时优先读文件；engine 提供实时回测指标/审计/前瞻预测。
        两条路都要能产出文档，云端缺 artifacts/ 时依然有语料。
        """
        docs: List[Dict[str, Any]] = []

        # --- 已知局限：文件存在读文件，否则用内置中文占位文本 ---
        kl_path = self.artifacts_dir / "known_limitations.md"
        if kl_path.exists():
            kl_text = kl_path.read_text(encoding="utf-8")
        else:
            kl_text = _BUILTIN_LIMITATIONS_TEXT
        docs.extend(self._split_markdown_sections(kl_text, "known_limitations.md"))

        # --- data_lineage：仅文件（无则跳过） ---
        dl_path = self.artifacts_dir / "data_lineage.md"
        if dl_path.exists():
            docs.extend(self._split_markdown_sections(dl_path.read_text(encoding="utf-8"), "data_lineage.md"))

        # --- final_metrics / backtest_predictions：文件存在则读 ---
        fp = self.artifacts_dir / "final_metrics.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            for _, row in df.iterrows():
                text = (f"模型 {row['model']} 的评估指标：RMSE={row['rmse']:.4f}, "
                        f"MAE={row['mae']:.4f}, R²={row['r2']:.4f}, 方向准确率={row['dir_acc']:.4f}。")
                docs.append({
                    "text": text,
                    "metadata": {"type": "model_metric", "model": row["model"]},
                })
                self._entity_set.add(str(row["model"]))

        fp = self.artifacts_dir / "backtest_predictions.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            for _, row in df.iterrows():
                text = (f"{row['test_quarter']}季度，四川省GDP累计同比实际值为{row['actual']:.1f}%，"
                        f"elastic_midas_chronos模型预测值为{row['elastic_midas_chronos']:.1f}%，"
                        f"last_value基准预测值为{row['last_value']:.1f}%。")
                y = None
                qm = re.match(r'(\d{4})Q(\d)', str(row['test_quarter']))
                if qm:
                    y = int(qm.group(1))
                docs.append({
                    "text": text,
                    "metadata": {"type": "backtest_prediction", "quarter": str(row["test_quarter"]),
                                "year": y if y else None},
                })

        # --- 实时构建：回测指标 / 数据审计 ← engine ---
        if self.engine is not None:
            docs.extend(self._build_realtime_engine_docs())
        else:
            self.logger.warning("No engine provided to RAGService; realtime backtest/audit docs skipped")

        # --- 前瞻预测：复用传入 engine，不自行触发流水线 ---
        if self.engine is not None:
            try:
                pred = getattr(self.engine, "latest_prediction", None) or self.engine.predict_next()
                ci = pred.get("confidence_interval") or {}
                lower = ci.get("lower", "N/A")
                upper = ci.get("upper", "N/A")
                if lower == "N/A":
                    ci_text = "未运行回测，无置信区间"
                else:
                    ci_text = f"90%置信区间[{lower}, {upper}]"
                pred_q = pred.get("prediction_quarter", pred.get("nowcast_quarter", "N/A"))
                date_str = pd.Timestamp.now().strftime("%Y-%m-%d")
                text = (
                    f"以下为模型预测值，非官方统计数据：模型预测{pred_q}"
                    f"四川省GDP累计同比增速为{pred['prediction_value']:.2f}%，{ci_text}，"
                    f"预测生成于{date_str}。模型为{pred['model_name']}，使用delta差分参数化。"
                )
                docs.append({
                    "text": text,
                    "metadata": {"type": "forward_prediction", "quarter": pred_q,
                                "value": pred["prediction_value"], "date": date_str},
                })
            except Exception as exc:
                self.logger.warning("forward_prediction doc build failed: %s", exc)
        else:
            self.logger.warning("No engine provided to RAGService; forward_prediction doc skipped")

        return docs

    def _build_realtime_engine_docs(self) -> List[Dict[str, Any]]:
        """从 engine 实时状态构建回测指标与数据审计文档（不依赖 artifacts 文件）。"""
        docs: List[Dict[str, Any]] = []
        bt = getattr(self.engine, "backtest_result", None) or {}
        metrics = bt.get("metrics")
        if metrics:
            model_name = getattr(self.engine, "selected_model_name", None) or "selected_model"
            rmse = metrics.get("rmse", 0.0)
            mae = metrics.get("mae", 0.0)
            r2 = metrics.get("r2", 0.0)
            dir_acc = metrics.get("direction_accuracy", 0.0)
            text = (f"模型 {model_name} 的回测评估指标：RMSE={rmse:.4f}, MAE={mae:.4f}, "
                    f"R²={r2:.4f}, 方向准确率={dir_acc:.4f}。")
            docs.append({"text": text, "metadata": {"type": "model_metric", "model": model_name}})
            self._entity_set.add(model_name)
        audit = getattr(self.engine, "audit_result", None)
        if audit:
            summary_txt = audit.get("summary")
            if isinstance(summary_txt, dict):
                summary_txt = str(summary_txt)
            if summary_txt:
                docs.append({"text": f"数据审计结论：{str(summary_txt)[:2000]}",
                             "metadata": {"type": "project_doc", "source_file": "audit_result"}})
        return docs

    @staticmethod
    def _split_markdown_sections(text: str, source_file: str) -> List[Dict[str, Any]]:
        """把 markdown 按 ## / # 标题切成片段，返回 project_doc 文档。"""
        docs: List[Dict[str, Any]] = []
        text = text.replace('\r\n', '\n')
        raw_sections = re.split(r"\n(?=##?\s)", text)
        sections: list[str] = []
        i = 0
        while i < len(raw_sections):
            sec = raw_sections[i].strip()
            # 仅标题的短片段并入下一段，避免独占 TF-IDF 打分
            is_title = ('\n' not in sec and len(sec) < 20)
            if is_title and i + 1 < len(raw_sections):
                next_sec = raw_sections[i + 1].strip()
                sec = sec + '\n' + next_sec
                i += 1
            if len(sec) >= 3:
                sections.append(sec)
            i += 1
        for sec in sections:
            docs.append({"text": sec[:2000], "metadata": {"type": "project_doc", "source_file": source_file}})
        return docs

    # ================================================================
    # B.2: TF-IDF 索引
    # ================================================================
    def _build_index(self) -> None:
        """构建 TF-IDF 索引。"""
        if not self.documents:
            self.logger.warning("No documents to index")
            return
        texts = [_normalize_chinese_numbers(d["text"]) for d in self.documents]
        self.vectorizer = TfidfVectorizer(max_features=3000, analyzer='char_wb', ngram_range=(2, 4))
        self.doc_vectors = self.vectorizer.fit_transform(texts)
        self.logger.info("Index built: %d docs, %d features", len(texts), self.doc_vectors.shape[1])

    # ================================================================
    # B.2.5: 硬过滤
    # ================================================================
    def _apply_metadata_filter(self, filters: dict, candidate_indices: list[int]) -> list[int]:
        """根据时间/指标约束缩小候选集。返回过滤后的索引列表。"""
        year = filters.get("year")
        quarter = filters.get("quarter")
        month = filters.get("month")
        indicators = filters.get("indicators", [])

        filtered: list[int] = []
        for idx in candidate_indices:
            meta = self.documents[idx].get("metadata", {})

            # 时间约束
            if year is not None and meta.get("year") != year:
                continue
            if quarter is not None and meta.get("quarter") != quarter:
                # 季度不匹配直接跳过（季度文件）
                # 但月度文件没有 quarter 字段 → 不因 quarter 过滤月度记录
                if meta.get("source_file", "").startswith("quarterly"):
                    if meta.get("quarter") != quarter:
                        continue
            if month is not None and meta.get("month") != month:
                if meta.get("month") is not None:
                    continue

            # 指标约束
            if indicators:
                doc_indicator = meta.get("indicator", "")
                if doc_indicator and not any(ind in doc_indicator for ind in indicators):
                    continue

            filtered.append(idx)

        return filtered

    # ================================================================
    # B.2.6: 混合打分 + min_score
    # ================================================================
    def search(self, query: str, top_k: int = 5, min_score: float = 0.15) -> List[Tuple[str, float, Dict[str, Any]]]:
        """检索 top_k 最相关片段。返回 [(text, score, metadata)]。

        - 查询解析抽取结构化过滤条件 → 硬过滤候选集
        - 有时间约束且过滤后为空 → 直接返回空（明确证据：资料中无此时间的数据）
        - 无时间约束且过滤后为空 → 退回全量 TF-IDF 检索
        - 混合打分：TF-IDF 0.7 + 关键词命中 0.3；抽不到关键词时退化为纯 TF-IDF
        - min_score 阈值：低于阈值的丢弃
        """
        if self.vectorizer is None or self.doc_vectors is None:
            return []

        filters = _extract_query_filters(query, self._entity_set)
        total_docs = len(self.documents)

        # 候选集：默认全部文档
        full_indices = list(range(total_docs))
        has_any_filter = (filters["year"] is not None or filters["quarter"] is not None
                          or filters["month"] is not None or bool(filters["indicators"]))

        # indicator-seeking query with no entity match and no time constraint → empty.
        # (e.g. "人口出生率是多少" — the data simply doesn't exist in our corpus.)
        # Exclude queries about model predictions (contain "模型"/"预测") from this
        # short-circuit, since forward-prediction docs don't map 1:1 to indicator names.
        if (not filters["has_time_constraint"] and not filters["indicators"]
                and re.search(r'(是多少|什么水平)', query)
                and not re.search(r'(模型|预测)', query)):
            return []

        if has_any_filter:
            candidate_indices = self._apply_metadata_filter(filters, full_indices)

            # 有时间约束且过滤后为空 → 直接返回空列表
            # （这是"资料中确实没有"的证据，不退回全量检索）
            if not candidate_indices:
                if filters["has_time_constraint"]:
                    self.logger.debug("Metadata filter (%s) returned empty — no data for this time range", filters)
                    return []
                # 无时间约束、仅有 indicator 过滤后仍为空 → 退回全量
                self.logger.debug("Indicator-only filter returned empty, falling back to full corpus")
                candidate_indices = full_indices
        else:
            # 什么也没抽到 → 全量检索
            candidate_indices = full_indices

        if not candidate_indices:
            return []

        # TF-IDF 在候选集内计算余弦相似度
        q_norm = _normalize_chinese_numbers(query)
        q_vec = self.vectorizer.transform([q_norm])
        candidate_vectors = self.doc_vectors[candidate_indices]
        scores = cosine_similarity(q_vec, candidate_vectors)[0]

        # 关键词命中加分（仅当抽到关键词时）
        q_keywords = filters["indicators"]
        use_hybrid = bool(q_keywords)
        combined_scores: list[float] = []
        for i, idx in enumerate(candidate_indices):
            tfidf_s = float(scores[i])
            if use_hybrid:
                doc_text = _normalize_chinese_numbers(self.documents[idx]["text"])
                hit_count = sum(1 for kw in q_keywords if kw in doc_text)
                kw_ratio = hit_count / max(len(q_keywords), 1)
                combined_scores.append(0.7 * tfidf_s + 0.3 * kw_ratio)
            else:
                # 无关键词 → 纯 TF-IDF（不退化为 0.7×），保持召回宽松
                combined_scores.append(tfidf_s)

        # 排序 + min_score 过滤
        ranked = sorted(zip(candidate_indices, combined_scores), key=lambda x: x[1], reverse=True)
        results: list[Tuple[str, float, Dict[str, Any]]] = []
        for idx, score in ranked:
            if score < min_score:
                continue
            results.append((self.documents[idx]["text"], score, self.documents[idx]["metadata"]))
            if len(results) >= top_k:
                break

        return results

    # ================================================================
    # B.3: 问答链路
    # ================================================================
    def ask(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """RAG 问答：检索 + LLM 生成。"""
        sources = self.search(question, top_k)
        if not sources:
            return {"answer": "资料中没有相关信息。", "sources": []}

        # 拼装 context
        context_parts = []
        for i, (text, score, meta) in enumerate(sources):
            context_parts.append(f"[{i+1}] {text}")
        context = "\n".join(context_parts)

        from .prompts.registry import render
        prompt = render("rag_qa", context=context, question=question)

        answer = self.llm.chat(
            prompt["system"], prompt["user"],
            temperature=prompt["temperature"], max_tokens=prompt["max_tokens"],
            prompt_id=prompt["id"], prompt_version=prompt["version"],
            caller="rag",
        )
        return {
            "answer": answer,
            "sources": [{"text": t, "score": s, "metadata": m} for t, s, m in sources],
        }


# ================================================================
# 便捷函数
# ================================================================
_rag_instance: Optional[RAGService] = None


def get_rag(config: Optional["AppConfig"] = None, engine: Optional["PredictionEngine"] = None) -> RAGService:
    global _rag_instance
    if _rag_instance is None:
        if config is None:
            from .config import AppConfig as _AppConfig
            config = _AppConfig()
        _rag_instance = RAGService(config, engine)
    return _rag_instance
