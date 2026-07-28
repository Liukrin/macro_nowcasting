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


def _normalize_chinese_numbers(text: str) -> str:
    """将中文数字统一为阿拉伯数字，避免'第三季度'与'第3季度'失配。"""
    cn_nums = {'一':'1','二':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9','零':'0','十':'10'}
    result = text
    for cn, ar in cn_nums.items():
        result = result.replace(cn, ar)
    return result


class RAGService:
    """检索增强生成服务。"""

    def __init__(self, data_dir: str = "data", artifacts_dir: str = "artifacts/final") -> None:
        self.logger = get_logger("sc_macro_agent.rag")
        self.data_dir = Path(data_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.llm = LLMClient()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_vectors: Optional[np.ndarray] = None
        self.documents: List[Dict[str, Any]] = []
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
        """从 quarterly_target 和 monthly_local 生成指标卡片。"""
        cards = []
        for fname, region_label in [("quarterly_target_real.csv", "四川省"), ("monthly_local_features_real.csv", "四川省")]:
            fp = self.data_dir / fname
            if not fp.exists():
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
                    quarter_str = f"{dt.year}年第{(dt.month-1)//3+1}季度" if dt.month in (3, 6, 9, 12) else f"{dt.year}年{dt.month}月"
                    text = f"{quarter_str}，{region_label}{ind_name}为{val:.1f}%。数据来源：国家统计局/四川省统计局。"
                    cards.append({
                        "text": text,
                        "metadata": {
                            "type": "indicator_card",
                            "region": region_label,
                            "date": dt.strftime("%Y-%m-%d"),
                            "indicator": ind_name,
                            "value": float(val),
                            "source_file": fname,
                        }
                    })
        return cards

    def _build_project_docs(self) -> List[Dict[str, Any]]:
        """从 artifacts/final/ 构建项目文档片段。"""
        docs = []
        for fname in ["data_lineage.md", "known_limitations.md"]:
            fp = self.artifacts_dir / fname
            if not fp.exists():
                continue
            text = fp.read_text(encoding="utf-8")
            # Split by sections
            sections = re.split(r"\n## ", text)
            for sec in sections:
                sec = sec.strip()
                if len(sec) < 20:
                    continue
                docs.append({
                    "text": sec[:2000],
                    "metadata": {"type": "project_doc", "source_file": fname},
                })

        # final_metrics
        fp = self.artifacts_dir / "final_metrics.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            for _, row in df.iterrows():
                text = f"模型 {row['model']} 的评估指标：RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, R²={row['r2']:.4f}, 方向准确率={row['dir_acc']:.4f}。"
                docs.append({
                    "text": text,
                    "metadata": {"type": "model_metric", "model": row["model"]},
                })

        # backtest predictions
        fp = self.artifacts_dir / "backtest_predictions.csv"
        if fp.exists():
            df = pd.read_csv(fp)
            for _, row in df.iterrows():
                text = (f"{row['test_quarter']}季度，四川省GDP累计同比实际值为{row['actual']:.1f}%，"
                        f"elastic_midas_chronos模型预测值为{row['elastic_midas_chronos']:.1f}%，"
                        f"last_value基准预测值为{row['last_value']:.1f}%。")
                docs.append({
                    "text": text,
                    "metadata": {"type": "backtest_prediction", "quarter": row["test_quarter"]},
                })

        # Latest forward prediction (for Q7: future forecast)
        try:
            from .prediction_engine import PredictionEngine
            from .config import AppConfig
            cfg = AppConfig(); cfg.data.data_dir = str(self.data_dir)
            engine = PredictionEngine(cfg)
            engine.run_agent(goal="audit_build_train", save_artifacts=False)
            pred = engine.predict_next()
            date_str = pd.Timestamp.now().strftime("%Y-%m-%d")
            text = (
                f"以下为模型预测值，非官方统计数据：模型预测{pred['prediction_quarter']}"
                f"四川省GDP累计同比增速为{pred['prediction_value']:.2f}%，"
                f"90%置信区间[{pred['confidence_interval']['lower']}, {pred['confidence_interval']['upper']}]，"
                f"预测生成于{date_str}。"
                f"模型为{pred['model_name']}，使用delta差分参数化。"
            )
            docs.append({
                "text": text,
                "metadata": {"type": "forward_prediction", "quarter": pred["prediction_quarter"],
                            "value": pred["prediction_value"], "date": date_str},
            })
        except Exception:
            pass  # silently skip if engine not available

        return docs

    # ================================================================
    # B.2: TF-IDF 检索
    # ================================================================
    def _build_index(self) -> None:
        """构建 TF-IDF 索引。"""
        if not self.documents:
            self.logger.warning("No documents to index")
            return
        # Normalize Chinese numbers to digits for consistent matching
        texts = [_normalize_chinese_numbers(d["text"]) for d in self.documents]
        self.vectorizer = TfidfVectorizer(max_features=3000, analyzer='char_wb', ngram_range=(2, 4))
        self.doc_vectors = self.vectorizer.fit_transform(texts)
        self.logger.info("Index built: %d docs, %d features", len(texts), self.doc_vectors.shape[1])

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """检索 top_k 最相关片段。返回 [(text, score, metadata)]。"""
        if self.vectorizer is None or self.doc_vectors is None:
            return []
        query = _normalize_chinese_numbers(query)
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.doc_vectors)[0]
        top_idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            if scores[idx] > 0.0:
                results.append((self.documents[idx]["text"], float(scores[idx]), self.documents[idx]["metadata"]))
        return results

    # ================================================================
    # B.3: 问答链路
    # ================================================================
    def ask(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """RAG 问答：检索 + LLM 生成。"""
        sources = self.search(question, top_k)
        if not sources:
            return {"answer": "资料中没有相关信息。", "sources": []}

        # 拼装 prompt
        context_parts = []
        for i, (text, score, meta) in enumerate(sources):
            context_parts.append(f"[{i+1}] {text}")
        context = "\n".join(context_parts)

        system = (
            "你是一个经济数据分析助手。你只能使用下面提供的检索资料来回答问题。\n"
            "规则：\n"
            "1. 如果检索资料中包含答案，直接引用并回答。\n"
            "2. 如果检索资料不足以回答问题，必须明确说'资料中没有相关信息'，不得猜测或编造。\n"
            "3. 禁止编造任何数字、政策文件名、会议名称、或未在资料中出现的信息。\n"
            "4. 引用时标注来源编号 [1] [2] 等。"
        )
        user = f"检索资料：\n{context}\n\n问题：{question}"

        answer = self.llm.chat(system, user)
        return {
            "answer": answer,
            "sources": [{"text": t, "score": s, "metadata": m} for t, s, m in sources],
        }


# ================================================================
# 便捷函数
# ================================================================
_rag_instance: Optional[RAGService] = None


def get_rag() -> RAGService:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGService()
    return _rag_instance
