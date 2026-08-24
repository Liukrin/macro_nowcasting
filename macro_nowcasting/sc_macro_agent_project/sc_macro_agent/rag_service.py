"""
轻量 RAG 服务：分池 TF-IDF 检索 + LLM 问答。
indicator_card 与 project_doc 独立向量空间，避免卡片模板词主导 IDF。
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

# ================================================================
# 版本标识（每次修改 ask() 签名必须升级）
# ================================================================
RAG_SERVICE_VERSION = "2.0.0"

# ================================================================
# 内置占位文本：云端部署无 artifacts/final/ 目录时的兜底
# ================================================================
# ⚠ 修改 docs/known_limitations.md 时必须同步更新此常量
# ⚠ 修改 docs/data_lineage.md      时必须同步更新此常量

# 兜底副本：docs/known_limitations.md 读取失败时使用，正常运行走 docs/ 下的源文件
_BUILTIN_LIMITATIONS_TEXT = (
    "# 已知局限\n"
    "\n"
    "以下为项目数据与模型的已知局限。\n"
    "\n"
    "## 1. 单季同比口径不可用\n"
    "四川GDP累计值序列在2017/2018间存在第四次全国经济普查造成的vintage断点。\n"
    "累计值差分得到的单季名义值跨断点后出现严重口径跳变(2017=13.16%, 2018=16.01%,\n"
    "而官方实际增速8.1%/8.0%)。因此单季同比(qoq_yoy)口径已弃用。\n"
    "\n"
    "## 2. feature_vintage 已实现（默认 two_month）\n"
    "已实现 full_quarter / two_month / one_month 三模式月度截断，默认 two_month，\n"
    "因为预测当季时通常仅前两月数据已发布。full_quarter 保留当季全部 3 个月，\n"
    "作为泄漏上界参照（含尚未发布的第 3 个月）；one_month 为最保守下限。\n"
    "指标级 ragged edge（按各指标真实发布时滞截断）为后续方向。\n"
    "\n"
    "## 3. target_lag_1未进入特征集\n"
    "16特征cap下target_lag_1被四川/全国/PMI特征挤掉。\n"
    "模型无法显式使用y_{t-1},缺少锚点。\n"
    "\n"
    "## 4. chronos-2-small API不兼容\n"
    "Chronos2Pipeline要求3D输入(n_series, n_variates, history_length),\n"
    "而bolt系列接受2D(batch, length)。未适配。\n"
    "\n"
    "## 5. 样本量限制\n"
    "回测窗口数24-32,统计检验(DM)功效有限。小样本下即使真实预测能力存在,\n"
    "也可能无法通过显著性检验。\n"
    "\n"
    "## 6. 数据覆盖\n"
    "- 国家月度数据截止2025-09(固定资产投资/工业增加值等)\n"
    "- PMI截止2025-12\n"
    "- 四川月度数据截止2025-12\n"
    "- 2025Q4 GDP为估算值\n"
    "\n"
    "## 7. 指标口径混杂\n"
    "四川月度指标均为累计同比(ytd_yoy),国家月度含累计同比和当月同比(mom_yoy)两种。\n"
    "混频建模中将两者纳入同一特征矩阵,口径差异可能引入噪声。\n"
    "\n"
    "## 8. 方向准确率口径\n"
    "last_value 预测的变化量恒为 0（所有窗口使用同一个 y_{t-1}），\n"
    "其方向准确率等于\"实际同比下降的窗口占比\"，不作预测能力解读。\n"
    "所有模型的方向准确率已改为与 0.5（随机猜测）比较并做二项检验。\n"
    "小样本下（23 对方向比较）无一模型显著优于随机。\n"
    "\n"
    "## 9. 常数偏差修正对 delta 残差无效的更正\n"
    "原报告称\"delta 残差均值为 0 是差分序列的自然性质\"——此表述有误。\n"
    "正确原因：带截距项的线性模型（ElasticNet 含截距），\n"
    "其样本内拟合残差均值必然约等于 0。截距吸收了系统性偏差，\n"
    "因此用样本内残差均值做常数修正等同于加 0，不起作用。\n"
    "\n"
    "## 10. 共享可变状态（st.cache_resource）\n"
    "load_view_data 与 load_engine 使用 @st.cache_resource，\n"
    "返回的是跨会话共享的同一个对象引用，其 dict 与其中的 engine 会被所有会话共用。\n"
    "- 本项目为单人演示场景，共享引擎可接受；\n"
    "- 生产环境需改为每会话独立实例或加锁。\n"
    "\n"
    "## 11. TF-IDF 词汇匹配局限\n"
    "采用 char_wb n-gram (2,4) 的 TF-IDF 为纯词汇匹配，无法理解语义等价。\n"
    "典型例子：\"四川的生产情况\"无法关联到\"规模以上工业增加值_累计同比\"，\n"
    "因为语料卡片文本不含\"生产\"一词。\n"
    "语义检索（BGE等嵌入模型）可解此类语义鸿沟，\n"
    "但本地嵌入权重不完整且会增加部署复杂度，暂未部署。\n"
    "\n"
    "## 12. 跨区域 diversity re-ranking 缺失\n"
    "search() 的合并排序仅按归一化 TF-IDF 分降序，不做跨区域多样性重排。\n"
    "当查询同时提到四川和全国时（如\"四川和全国比工业增速差多少\"），\n"
    "top-5 结果可能全部偏向一个区域，无法保证两个区域的结果都出现。\n"
    "引入 MMR（Maximal Marginal Relevance）或按区域分层抽样可解。但当前检索\n"
    "量级较小（top_k=5），实际影响有限，暂未实现。\n"
)

# 兜底副本：docs/data_lineage.md 读取失败时使用，正常运行走 docs/ 下的源文件
_BUILTIN_LINEAGE_TEXT = (
    "# 数据溯源台账\n"
    "\n"
    "## 数据源\n"
    "- 四川省数据202512.xlsx: 季度GDP (col 0-8) + 月度经济指标 (col 12-17)\n"
    "- 国家数据202512.xlsx: 月度经济指标 (Sheet '月度', col 0-4)\n"
    "- pmi_data.csv: PMI 14个分项 (2015-01 ~ 2025-12)\n"
    "\n"
    "## 日期解析修复\n"
    "- Excel序列号 (45566/45839/45901等) 被正确解析为2024-10~2025-09\n"
    "- YYYYMM格式整数 (201002等) 优先于Excel序列号检查\n"
    "- 15行全NaN空行已清除\n"
    "\n"
    "## 口径区分\n"
    "- 固定资产投资额累计增长(%) → 累计同比 (ytd_yoy)\n"
    "- 工业增加值同比增长(%) → 当月同比 (mom_yoy)\n"
    "- 房地产投资/社消零售累计值 → 水平序列 (cum_level, 排除特征池)\n"
    "- GDP_累计同比 → 指数转增速 (117.7→17.7%)\n"
    "\n"
    "## ETL脚本\n"
    "- rebuild_etl.py: 完整重建流程\n"
)


def _normalize_chinese_numbers(text: str) -> str:
    """将中文数字统一为阿拉伯数字。正确处理"十一"→11、"二十"→20、"二十三"→23。"""
    result: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in _CN_DIGITS:
            digit = _CN_DIGITS[ch]
            if i + 1 < len(text) and text[i + 1] == _CN_TEN:
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

_REGION_PATTERNS: dict[str, re.Pattern] = {
    "四川省": re.compile(r"(四川|川内|全省|成都|蜀)"),
    "全国":   re.compile(r"(全国|国家层面|国内整体|中国整体)"),
}


def _extract_year(query: str) -> int | None:
    m = _YEAR_RE.search(query)
    if not m:
        return None
    return int(m.group(1))


def _extract_quarter(query: str) -> int | None:
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


def _extract_region(query: str) -> str | None:
    hits = [label for label, pat in _REGION_PATTERNS.items() if pat.search(query)]
    if len(hits) == 1:
        return hits[0]
    return None


# ================================================================
# 同义词归一表
# ================================================================
_INDICATOR_ALIASES: dict[str, str] = {
    "社零": "社会消费品零售总额",
    "消费": "社会消费品零售总额",
    "工增": "工业增加值",
    "规上工业": "工业增加值",
    "工业增速": "工业增加值",
    "地产投资": "房地产开发投资",
    "房地产投资": "房地产开发投资",
    "经济增速": "GDP_同比增速",
    "GDP增速": "GDP_同比增速",
    "GDP增长": "GDP_同比增速",
    "固投": "固定资产投资",
}


def normalize_query_aliases(text: str) -> str:
    """整串替换用户俗称→语料标准术语。用于 _extract_query_filters 和 query_indicator。"""
    result = text
    for alias, canonical in _INDICATOR_ALIASES.items():
        if alias in result:
            result = result.replace(alias, canonical, 1)
    return result


def get_indicator_aliases() -> dict[str, str]:
    return dict(_INDICATOR_ALIASES)


# ================================================================
# 元问题正则
# ================================================================
_META_PATTERNS = re.compile(
    r'(你是什么|你是谁|介绍一下|你能回答|你能做什么|你会什么|你会回答|'
    r'这个系统|什么功能|能干嘛|能做什么|帮助|使用说明)'
)


def _is_meta_question(question: str) -> bool:
    return bool(_META_PATTERNS.search(question))


# ================================================================
# 概念/方法论问题前置路由 —— 挡在工具循环之外
# ================================================================

# 条件 A —— 概念提问句式
_METHODOLOGY_SENTENCE_PATTERNS = re.compile(
    r'是什么|什么是|原理|怎么工作|如何工作|为什么用|'
    r'有什么区别|区别是什么|介绍一下|解释一下|含义|概念'
)

# 条件 A —— 方法论术语
_METHODOLOGY_TERMS = re.compile(
    r'MIDAS|DFM|TSLM|Chronos|PCA|卡尔曼|Kalman|'
    r'混频|nowcast|现时预测|动态因子|岭回归|Ridge|'
    r'ElasticNet|GBRT|回测|expanding|RAG|Agent|智能体'
)

# 条件 B —— 数值查询信号（命中任一即排除）
_NUMERICAL_QUERY_YEAR = re.compile(r'\b(19|20)\d{2}\b')
_NUMERICAL_QUERY_PERIOD = re.compile(
    r'季度|Q1|Q2|Q3|Q4|一季度|二季度|三季度|四季度|月份|月'
)
_NUMERICAL_QUERY_VALUE = re.compile(
    r'是多少|多少|数值|几|增速是|值为'
)


def _is_methodology_question(question: str) -> bool:
    """高置信度判定概念/方法论问题，命中则跳过工具循环。

    条件 A：同时命中概念提问句式 + 方法论术语。
    条件 B：不含任何数值查询信号（年份/季度/月份/取值词）。
    两个条件缺一不可 —— 误判代价不对称，宁可漏过不可错杀。
    """
    # 条件 A
    has_sentence = bool(_METHODOLOGY_SENTENCE_PATTERNS.search(question))
    has_term = bool(_METHODOLOGY_TERMS.search(question))
    if not (has_sentence and has_term):
        return False

    # 条件 B：任何数值信号命中即排除
    if _NUMERICAL_QUERY_YEAR.search(question):
        return False
    if _NUMERICAL_QUERY_PERIOD.search(question):
        return False
    if _NUMERICAL_QUERY_VALUE.search(question):
        return False

    return True


# ================================================================
# PMI 分项前缀映射
# ================================================================
_PMI_SUBINDEX_PREFIX_MAP: dict[str, str] = {
    "生产": "PMI_生产",
    "新订单": "PMI_新订单",
    "原材料库存": "PMI_原材料库存",
    "从业人员": "PMI_从业人员",
    "供应商配送时间": "PMI_供应商配送时间",
    "新出口订单": "PMI_新出口订单",
    "进口": "PMI_进口",
    "采购量": "PMI_采购量",
    "主要原材料购进价格": "PMI_主要原材料购进价格",
    "出厂价格": "PMI_出厂价格",
    "产成品库存": "PMI_产成品库存",
    "在手订单": "PMI_在手订单",
    "生产经营活动预期": "PMI_生产经营活动预期",
}


# 四川默认加分
_LOCAL_REGION_BOOST = 0.05

# 当选模型文档加分：所有 model_metric 文档模板高度相似，char_wb n-gram
# 聚集成团（scores 0.18–0.19）。0.02 的微微倾斜足以让"系统最终选用"的文档胜出。
_SELECTED_MODEL_BOOST = 0.02

# 分池阈值
_CARD_MIN_SCORE = 0.15
_DOC_MIN_SCORE = 0.08

# 元问题 capability 文档加分
_META_CAPABILITY_BOOST = 0.30


def _extract_query_filters(query: str, entity_set: set[str]) -> dict:
    """从自然语言问题中抽取结构化过滤条件。
    先做别名归一化再匹配实体，TF-IDF 打分仍用原始 query 文本。
    """
    year = _extract_year(query)
    quarter = _extract_quarter(query)
    month = _extract_month(query)
    region = _extract_region(query)
    has_time = (year is not None) or (quarter is not None) or (month is not None)

    # 别名归一化后的字符串用于实体匹配
    normalized = normalize_query_aliases(query)
    indicators: list[str] = []
    remaining = normalized
    for kw in sorted(entity_set, key=len, reverse=True):
        if kw and kw in remaining:
            indicators.append(kw)
            remaining = remaining.replace(kw, ' ', 1)

    return {
        "year": year,
        "quarter": quarter,
        "month": month,
        "region": region,
        "indicators": indicators,
        "has_time_constraint": has_time,
    }


class RAGService:
    """检索增强生成服务——分池 TF-IDF 索引。"""

    def __init__(self, config: "AppConfig", engine: Optional["PredictionEngine"] = None) -> None:
        self.logger = get_logger("sc_macro_agent.rag")
        self.config = config
        self.data_dir = config.data.resolve_dir()
        self.artifacts_dir = config.data.resolve_artifact_dir(create=False) / "final"
        self.llm = LLMClient.get_instance()
        self.engine = engine
        self.documents: List[Dict[str, Any]] = []
        self._indicator_set: set[str] = set()
        self._entity_set: set[str] = set()
        self._caliber_notes: dict[str, str] = {}

        # 分池索引
        self.vec_card: Optional[TfidfVectorizer] = None
        self.mat_card: Optional[np.ndarray] = None
        self.idx_card: list[int] = []           # pool pos → documents index

        self.vec_doc: Optional[TfidfVectorizer] = None
        self.mat_doc: Optional[np.ndarray] = None
        self.idx_doc: list[int] = []             # pool pos → documents index

        self._build_caliber_lookup()
        self._build_corpus()
        self._build_index()

        # 缓存注入真实指标名后的工具 schema（语料在构造期已冻结）
        from .tools import build_tool_schemas
        self._tool_schemas = build_tool_schemas(self)

    # ================================================================
    # 口径标注查找
    # ================================================================
    def _build_caliber_lookup(self) -> None:
        fp = self.data_dir / "metadata_real.csv"
        if not fp.exists():
            self.logger.warning("metadata_real.csv not found, caliber_note will be empty")
            return
        try:
            md = pd.read_csv(fp)
            for _, row in md.iterrows():
                name = row.get("standard_name", "")
                if not name or pd.isna(name):
                    continue
                is_yoy = bool(row.get("is_yoy", False))
                is_cumulative = bool(row.get("is_cumulative", False))
                parts: list[str] = []
                if is_yoy:
                    parts.append("同比")
                if is_cumulative:
                    parts.append("累计（年初至今）")
                if not parts:
                    parts.append("水平值")
                self._caliber_notes[str(name)] = "，".join(parts)
        except Exception as exc:
            self.logger.warning("Failed to read metadata_real.csv: %s", exc)

    # ================================================================
    # B.1: 语料构建
    # ================================================================
    def _build_corpus(self) -> None:
        docs: List[Dict[str, Any]] = []
        docs.extend(self._build_indicator_cards())
        docs.extend(self._build_project_docs())
        self.documents = docs
        # coverage 文档在 self.documents 就绪后生成（依赖 list_indicators()）
        coverage_text = self._build_coverage_doc()
        if coverage_text:
            self.documents.append({
                "text": coverage_text,
                "metadata": {"type": "coverage", "source": "system"},
            })
        n_card = sum(1 for d in docs if d.get("metadata", {}).get("type") == "indicator_card")
        n_doc = len(docs) - n_card
        self.logger.info("Corpus built: %d docs (card=%d, doc=%d)", len(docs), n_card, n_doc)

    def _build_indicator_cards(self) -> List[Dict[str, Any]]:
        cards = []
        for fname, region_label, freq_label, has_caliber in [
            ("quarterly_target_real.csv", "四川省", "quarterly", False),
            ("monthly_local_features_real.csv", "四川省", "monthly", False),
            ("monthly_national_features_real.csv", "全国", "monthly", True),
        ]:
            fp = self.data_dir / fname
            if not fp.exists():
                self.logger.warning("Missing data file: %s", fp)
                continue
            df = pd.read_csv(fp)
            df["date"] = pd.to_datetime(df["date"])
            for ind_name in sorted(df["indicator_name"].unique()):
                display_name = ind_name
                if fname == "monthly_national_features_real.csv":
                    display_name = _PMI_SUBINDEX_PREFIX_MAP.get(ind_name, ind_name)
                sub = df[df["indicator_name"] == ind_name].sort_values("date")
                for _, row in sub.iterrows():
                    dt = row["date"]
                    val = row["indicator_value"]
                    if pd.isna(val):
                        continue
                    if freq_label == "quarterly":
                        date_str = f"{dt.year}年第{(dt.month - 1) // 3 + 1}季度"
                        quarter_val = (dt.month - 1) // 3 + 1
                    else:
                        date_str = f"{dt.year}年{dt.month}月"
                        quarter_val = (dt.month - 1) // 3 + 1
                    caliber = None
                    if has_caliber and "caliber" in df.columns:
                        caliber = row.get("caliber")
                    caliber_suffix = f"（口径：{caliber}）" if caliber and pd.notna(caliber) else ""
                    text = (
                        f"{date_str}，{region_label}{display_name}为{val:.1f}%{caliber_suffix}。"
                        f"数据来源：国家统计局/四川省统计局。"
                    )
                    cards.append({
                        "text": text,
                        "metadata": {
                            "type": "indicator_card",
                            "region": region_label,
                            "date": dt.strftime("%Y-%m-%d"),
                            "year": dt.year,
                            "month": dt.month,
                            "quarter": quarter_val,
                            "indicator": display_name,
                            "value": float(val),
                            "source_file": fname,
                            "caliber": caliber if caliber and pd.notna(caliber) else None,
                        }
                    })
                    self._indicator_set.add(display_name)
                    self._entity_set.add(display_name)
        # 别名 canonical 值也加入 _entity_set，使关键词抽取能命中
        # 指标名含后缀（如"_累计同比"）但别名只映射到词干。
        # 例："工业增速"→"规模以上工业增加值"，entity 为"规模以上工业增加值_累计同比"，
        # 加入词干后 _extract_query_filters 可抽到关键词，metadata filter 用 str in 匹配。
        for canonical in set(_INDICATOR_ALIASES.values()):
            self._entity_set.add(canonical)
        return cards

    def _build_project_docs(self) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []

        # 统一路径优先级：docs/（源码版本控制）→ artifacts/final/（运行产物）→ 内置常量兜底
        kl_path = Path(__file__).parent.parent / "docs" / "known_limitations.md"
        if not kl_path.exists():
            kl_path = self.artifacts_dir / "known_limitations.md"
        if kl_path.exists():
            from datetime import datetime as _dt
            mtime = _dt.fromtimestamp(kl_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            self.logger.info("known_limitations.md 命中源文件: %s", kl_path)
            kl_text = f"（本文档最后修改时间：{mtime}）\n\n{kl_path.read_text(encoding='utf-8')}"
        else:
            self.logger.warning("未找到源文件，已退到内置副本，内容可能过时 (known_limitations.md)")
            kl_text = _BUILTIN_LIMITATIONS_TEXT
        docs.extend(self._split_markdown_sections(kl_text, "known_limitations.md"))

        dl_path = Path(__file__).parent.parent / "docs" / "data_lineage.md"
        if not dl_path.exists():
            dl_path = self.artifacts_dir / "data_lineage.md"
        if dl_path.exists():
            from datetime import datetime as _dt
            mtime = _dt.fromtimestamp(dl_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            self.logger.info("data_lineage.md 命中源文件: %s", dl_path)
            dl_text = f"（本文档最后修改时间：{mtime}）\n\n{dl_path.read_text(encoding='utf-8')}"
        else:
            self.logger.warning("未找到源文件，已退到内置副本，内容可能过时 (data_lineage.md)")
            dl_text = _BUILTIN_LINEAGE_TEXT
        docs.extend(self._split_markdown_sections(dl_text, "data_lineage.md"))

        mh_path = Path(__file__).parent.parent / "docs" / "methodology.md"
        if not mh_path.exists():
            mh_path = self.artifacts_dir / "methodology.md"
        if mh_path.exists():
            self.logger.info("methodology.md 命中源文件: %s", mh_path)
            mh_text = mh_path.read_text(encoding="utf-8")
        else:
            self.logger.warning("未找到 methodology.md 源文件，跳过注入")
            mh_text = ""
        if mh_text.strip():
            docs.extend(self._split_markdown_sections(mh_text, "methodology.md"))

        if self.engine is not None:
            docs.extend(self._build_model_metric_docs())
            docs.extend(self._build_backtest_window_docs())
            docs.extend(self._build_audit_docs())
        else:
            self.logger.warning("No engine provided to RAGService; model metrics / backtest / audit docs skipped")

        if self.engine is not None:
            try:
                pred = getattr(self.engine, "latest_prediction", None) or self.engine.predict_next()
                ci = pred.get("confidence_interval") or {}
                lower = ci.get("lower", "N/A")
                upper = ci.get("upper", "N/A")
                ci_text = "未运行回测，无置信区间" if lower == "N/A" else f"90%置信区间[{lower}, {upper}]"
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

        # --- capability 文档（A1）---
        docs.append({
            "text": (
                "本系统是四川省GDP混频预测（nowcasting）数据助手，支持以下查询：\n"
                "1. 查询四川省和全国的经济指标数值，包括GDP增速、工业增加值、社会消费品零售总额、"
                "固定资产投资、房地产开发投资、PMI及其分项等，覆盖季度和月度频率；\n"
                "2. 查询模型排行榜，比较不同模型的回测性能（RMSE、方向准确率等）；\n"
                "3. 获取当前季度的GDP nowcast预测值与置信区间；\n"
                "4. 查阅项目的已知局限、数据覆盖范围和方法论说明。\n"
                "\n"
                "本系统不覆盖：股票行情、企业财报、四川以外的省份数据、"
                "2010年以前数据、以及未在指标清单中列出的任何指标。"
            ),
            "metadata": {"type": "capability", "source": "system"},
        })

        # coverage 文档在 _build_corpus() 中 self.documents 就绪后追加，避免循环依赖

        return docs

    def _build_coverage_doc(self) -> str:
        """从 list_indicators() 真实输出自动生成数据覆盖说明。

        按区域分组，列出指标名、频率、时间范围，末尾附别名表。
        """
        info = self.list_indicators()
        indicators = info.get("indicators", [])
        if not indicators:
            return ""

        # 按区域分组
        by_region: dict[str, list[dict]] = {}
        for ind in indicators:
            region = ind.get("region", "未知")
            by_region.setdefault(region, []).append(ind)

        total = len(indicators)
        parts: list[str] = [f"本系统语料覆盖 {total} 个指标。"]

        for region in ("四川省", "全国"):
            group = by_region.get(region, [])
            if not group:
                continue
            # 统计频率范围
            quarterly = [i for i in group if i.get("freq") == "quarterly"]
            monthly = [i for i in group if i.get("freq") == "monthly"]
            names = [i["name"] for i in group]
            part = f"{region}（{len(group)} 个）：{'、'.join(names)}"
            if quarterly:
                q_starts = [i["start"] for i in quarterly if i.get("start")]
                q_ends = [i["end"] for i in quarterly if i.get("end")]
                q_range = f"{min(q_starts)}–{max(q_ends)}" if q_starts else "N/A"
                part += f"，季度 {q_range}"
            if monthly:
                m_starts = [i["start"] for i in monthly if i.get("start")]
                m_ends = [i["end"] for i in monthly if i.get("end")]
                m_range = f"{min(m_starts)}–{max(m_ends)}" if m_starts else "N/A"
                part += f"，月度 {m_range}"
            parts.append(part)

        # 别名表
        aliases = info.get("aliases", {})
        if aliases:
            alias_pairs = [f"{a}={c}" for a, c in aliases.items()]
            parts.append(f"常用别名：{'，'.join(alias_pairs)}。")

        return "。".join(parts) if len(parts) > 1 else parts[0]

    def _build_model_metric_docs(self) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        leaderboard = getattr(self.engine, "leaderboard", None) or []
        selected_name = getattr(self.engine, "selected_model_name", None)
        if not leaderboard:
            return docs

        parsed: list[dict] = []
        for entry in leaderboard:
            if isinstance(entry, dict):
                name = entry.get("model_name", "")
                rmse = float(entry.get("rmse", 0.0))
                mae = float(entry.get("mae", 0.0))
                r2 = float(entry.get("r2", 0.0))
                dir_acc = float(entry.get("direction_accuracy", 0.0))
            elif hasattr(entry, "model_name"):
                name = entry.model_name
                rmse = float(getattr(entry, "rmse", 0.0))
                mae = float(getattr(entry, "mae", 0.0))
                r2 = float(getattr(entry, "r2", 0.0))
                dir_acc = float(getattr(entry, "direction_accuracy", 0.0))
            else:
                continue
            parsed.append({"name": name, "rmse": rmse, "mae": mae, "r2": r2, "dir_acc": dir_acc})
        parsed.sort(key=lambda x: x["rmse"])
        best_rmse = parsed[0]["rmse"] if parsed else 0.0
        total = len(parsed)

        for rank, entry in enumerate(parsed, 1):
            name = entry["name"]
            is_selected = "是" if selected_name and name == selected_name else "否"
            gap = f"本模型为最优" if rank == 1 else f"较最优模型 RMSE 高 {entry['rmse'] - best_rmse:.4f}"
            text = (
                f"排名：第 {rank} 名（共 {total} 个候选）| 系统选用：{is_selected}\n"
                f"模型 {name} 的评估指标（验证集差分口径）："
                f"RMSE={entry['rmse']:.4f}, MAE={entry['mae']:.4f}, "
                f"R²={entry['r2']:.4f}, 方向准确率={entry['dir_acc']:.4f}。"
                f"（{gap}）"
            )
            docs.append({
                "text": text,
                "metadata": {"type": "model_metric", "model": name, "rank": rank, "selected": is_selected == "是"},
            })
            self._entity_set.add(str(name))

        bt = getattr(self.engine, "backtest_result", None) or {}
        bt_metrics = bt.get("metrics")
        lines = ["模型对比汇总（按验证集 RMSE 升序，验证集差分口径）", ""]
        for rank, entry in enumerate(parsed, 1):
            tag = " ← 系统选用" if (selected_name and entry["name"] == selected_name) else ""
            base_tag = "（基线）" if entry["name"] in ("last_value", "mean_recent") else ""
            lines.append(f"第{rank}名 {entry['name']:<18} RMSE {entry['rmse']:.4f}  MAE {entry['mae']:.4f}{tag}{base_tag}")
        lines.append("")
        lines.append("说明：此处 RMSE 为验证集差分口径。")
        if bt_metrics:
            backtest_rmse = bt_metrics.get("rmse")
            backtest_dir = bt_metrics.get("direction_accuracy")
            if backtest_rmse is not None:
                dir_str = f"，方向准确率 {backtest_dir:.1%}" if backtest_dir is not None else ""
                lines.append(
                    f"32 窗口 expanding-window 回测（level 口径、含 2020 年断点）的 RMSE 为 {backtest_rmse:.3f}{dir_str}，"
                    "两者口径不同，不可直接比较。"
                )
        docs.append({
            "text": "\n".join(lines),
            "metadata": {"type": "model_comparison", "n_models": total, "best_model": parsed[0]["name"]},
        })

        if selected_name:
            best = parsed[0]
            docs.append({
                "text": (
                    f"系统最终选用的模型是 {selected_name}。"
                    f"该模型在验证集差分口径下 RMSE={best['rmse']:.4f}，MAE={best['mae']:.4f}，"
                    f"排名第 1 名（共 {total} 个候选）。"
                    f"系统选用该模型作为预测引擎，用户界面展示的预测值与回测指标均来自该模型。"
                ),
                "metadata": {"type": "model_metric", "model": selected_name, "selected": True, "rank": 1},
            })

        self._entity_set.add("系统选用")
        self._entity_set.add("模型对比")
        self._entity_set.add("selected_model")
        return docs

    def _build_backtest_window_docs(self) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        bt = getattr(self.engine, "backtest_result", None) or {}
        windows = bt.get("window_results", [])
        if not isinstance(windows, list):
            return docs
        for w in windows:
            if not isinstance(w, dict):
                continue
            qtr = w.get("test_quarter", "")
            actual = w.get("actual")
            pred = w.get("prediction")
            model_name = w.get("model_name", getattr(self.engine, "selected_model_name", "?"))
            if actual is None or pred is None:
                continue
            text = (f"{qtr}季度，四川省GDP累计同比实际值为{float(actual):.1f}%，"
                    f"{model_name}模型预测值为{float(pred):.1f}%。"
                    f"（32窗口 expanding-window，level 口径）")
            y = None
            qm = re.match(r'(\d{4})Q(\d)', str(qtr))
            if qm:
                y = int(qm.group(1))
            docs.append({
                "text": text,
                "metadata": {"type": "backtest_prediction", "quarter": str(qtr), "year": y if y else None},
            })
        return docs

    def _build_audit_docs(self) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
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
        docs: List[Dict[str, Any]] = []
        text = text.replace('\r\n', '\n')
        raw_sections = re.split(r"\n(?=##?\s)", text)
        sections: list[str] = []
        i = 0
        while i < len(raw_sections):
            sec = raw_sections[i].strip()
            is_title = ('\n' not in sec and len(sec) < 20)
            if is_title and i + 1 < len(raw_sections):
                sec = sec + '\n' + raw_sections[i + 1].strip()
                i += 1
            if len(sec) >= 3:
                sections.append(sec)
            i += 1
        for sec in sections:
            docs.append({"text": sec[:2000], "metadata": {"type": "project_doc", "source_file": source_file}})
        return docs

    # ================================================================
    # B.2: 分池 TF-IDF 索引
    # ================================================================
    def _build_index(self) -> None:
        """构建两套独立 TF-IDF 索引。
        分池理由：indicator_card 占语料 93%（~2887 条），共用向量空间会让
        IDF 被卡片模板词主导，project_doc 的区分度被淹没（Q6 实测 top1 仅 0.0399）。
        """
        import time as _time
        if not self.documents:
            self.logger.warning("No documents to index")
            return

        t0 = _time.perf_counter()

        # --- POOL_CARD ---
        self.idx_card = [i for i, d in enumerate(self.documents)
                         if d.get("metadata", {}).get("type") == "indicator_card"]
        if self.idx_card:
            card_texts = [_normalize_chinese_numbers(self.documents[i]["text"]) for i in self.idx_card]
            self.vec_card = TfidfVectorizer(max_features=3000, analyzer='char_wb', ngram_range=(2, 4))
            self.mat_card = self.vec_card.fit_transform(card_texts)
        else:
            self.vec_card = TfidfVectorizer(max_features=3000, analyzer='char_wb', ngram_range=(2, 4))
            self.mat_card = None

        n_card_feat = self.mat_card.shape[1] if self.mat_card is not None else 0

        # --- POOL_DOC ---
        self.idx_doc = [i for i, d in enumerate(self.documents)
                        if d.get("metadata", {}).get("type") != "indicator_card"]
        if self.idx_doc:
            doc_texts = [_normalize_chinese_numbers(self.documents[i]["text"]) for i in self.idx_doc]
            self.vec_doc = TfidfVectorizer(max_features=1200, analyzer='char_wb', ngram_range=(2, 4))
            self.mat_doc = self.vec_doc.fit_transform(doc_texts)
        else:
            self.vec_doc = TfidfVectorizer(max_features=1200, analyzer='char_wb', ngram_range=(2, 4))
            self.mat_doc = None

        n_doc_feat = self.mat_doc.shape[1] if self.mat_doc is not None else 0
        elapsed = (_time.perf_counter() - t0) * 1000
        self.logger.info("Index built: card=%d docs/%d feat, doc=%d docs/%d feat (%.0f ms)",
                         len(self.idx_card), n_card_feat, len(self.idx_doc), n_doc_feat, elapsed)

    # ================================================================
    # B.2.5: 硬过滤（仅作用于 POOL_CARD）
    # ================================================================
    def _apply_metadata_filter(self, filters: dict, pool_indices: list[int],
                               fallback_region: bool = False) -> list[int]:
        """根据时间/指标/地区约束在池内缩小候选集。返回池内位置列表。"""
        year = filters.get("year")
        quarter = filters.get("quarter")
        month = filters.get("month")
        region = filters.get("region")
        indicators = filters.get("indicators", [])

        filtered: list[int] = []
        for pool_pos in pool_indices:
            doc_idx = self.idx_card[pool_pos]
            meta = self.documents[doc_idx].get("metadata", {})

            if year is not None and meta.get("year") != year:
                continue
            if quarter is not None and meta.get("quarter") != quarter:
                if meta.get("source_file", "").startswith("quarterly"):
                    if meta.get("quarter") != quarter:
                        continue
            if month is not None and meta.get("month") != month:
                if meta.get("month") is not None:
                    continue

            if indicators:
                doc_indicator = meta.get("indicator", "")
                if doc_indicator and not any(ind in doc_indicator for ind in indicators):
                    continue

            if region and region != "" and not fallback_region:
                doc_region = meta.get("region", "")
                if doc_region and doc_region != region:
                    continue

            filtered.append(pool_pos)

        return filtered

    # ================================================================
    # B.2.6: 分池检索
    # ================================================================
    def search(self, query: str, top_k: int = 5, min_score: float = 0.15) -> List[Tuple[str, float, Dict[str, Any]]]:
        """分池检索 + 归一化合并。

        - POOL_CARD（indicator_card）：硬过滤 → TF-IDF → Sichuan bonus → pool top_k
        - POOL_DOC（其余文档）：直接 TF-IDF → pool top_k
        - 归一化：score_norm = score / (pool_top1 + 1e-9)
        - 按归一化分降序合并取 top_k，返回**原始 score** + metadata["pool"]
        """
        results: list[Tuple[str, float, Dict[str, Any]]] = []
        filters = _extract_query_filters(query, self._entity_set)
        has_any_filter = (filters["year"] is not None or filters["quarter"] is not None
                          or filters["month"] is not None or filters["region"] is not None
                          or bool(filters["indicators"]))

        # 提前退出：indicator-seeking query with no match
        if (not filters["has_time_constraint"] and not filters["indicators"]
                and re.search(r'(是多少|什么水平)', query)
                and not re.search(r'(模型|预测)', query)):
            return []

        q_norm = _normalize_chinese_numbers(query)
        card_ranked: list[tuple[int, float]] = []  # (documents index, raw_score)
        doc_ranked: list[tuple[int, float]] = []

        # ================================================================
        # POOL_CARD
        # ================================================================
        if self.mat_card is not None and len(self.idx_card) > 0:
            all_card_pos = list(range(len(self.idx_card)))
            if has_any_filter:
                candidate_pos = self._apply_metadata_filter(filters, all_card_pos)
                if not candidate_pos and filters["region"]:
                    self.logger.debug("Region filter (%s) empty, relaxing region", filters["region"])
                    candidate_pos = self._apply_metadata_filter(filters, all_card_pos, fallback_region=True)
                if not candidate_pos:
                    if filters["has_time_constraint"]:
                        self.logger.debug("Card filter returned empty — no data for this time range")
                    else:
                        candidate_pos = all_card_pos
            else:
                candidate_pos = all_card_pos

            if candidate_pos:
                q_vec = self.vec_card.transform([q_norm])
                candidate_mat = self.mat_card[candidate_pos]
                scores = cosine_similarity(q_vec, candidate_mat)[0]

                q_keywords = filters["indicators"]
                use_hybrid = bool(q_keywords)
                for i, pool_pos in enumerate(candidate_pos):
                    tfidf_s = float(scores[i])
                    if use_hybrid:
                        doc_idx = self.idx_card[pool_pos]
                        doc_text = _normalize_chinese_numbers(self.documents[doc_idx]["text"])
                        hit_count = sum(1 for kw in q_keywords if kw in doc_text)
                        kw_ratio = hit_count / max(len(q_keywords), 1)
                        tfidf_s = 0.7 * tfidf_s + 0.3 * kw_ratio

                    # Sichuan bonus: 查询同时提到四川和全国时（如"四川和全国比"），
                    # 两边都不加分，让 TF-IDF 公平竞争
                    _both_regions = (
                        filters["region"] is None
                        and _REGION_PATTERNS["四川省"].search(query)
                        and _REGION_PATTERNS["全国"].search(query)
                    )
                    if not filters["region"] and has_any_filter and not _both_regions:
                        doc_idx = self.idx_card[pool_pos]
                        if self.documents[doc_idx].get("metadata", {}).get("region") == "四川省":
                            tfidf_s += _LOCAL_REGION_BOOST

                    if tfidf_s >= _CARD_MIN_SCORE:
                        card_ranked.append((self.idx_card[pool_pos], tfidf_s))

                card_ranked.sort(key=lambda x: x[1], reverse=True)

        # ================================================================
        # POOL_DOC
        # ================================================================
        if self.mat_doc is not None and len(self.idx_doc) > 0:
            q_vec = self.vec_doc.transform([q_norm])
            scores = cosine_similarity(q_vec, self.mat_doc)[0]
            for i, doc_idx in enumerate(self.idx_doc):
                s = float(scores[i])
                # 当选模型文档轻微倾斜：所有 model_metric 模板高度相似
                if self.documents[doc_idx].get("metadata", {}).get("selected") is True:
                    s += _SELECTED_MODEL_BOOST
                if s >= _DOC_MIN_SCORE:
                    doc_ranked.append((doc_idx, s))
            doc_ranked.sort(key=lambda x: x[1], reverse=True)

        # ================================================================
        # 归一化 + 合并
        # ================================================================
        card_top1 = card_ranked[0][1] if card_ranked else 1.0
        doc_top1 = doc_ranked[0][1] if doc_ranked else 1.0

        merged: list[tuple[int, float, str]] = []  # (doc_idx, raw_score, pool)

        for doc_idx, raw_score in card_ranked:
            norm = raw_score / (card_top1 + 1e-9)
            merged.append((doc_idx, raw_score, "card", norm))

        for doc_idx, raw_score in doc_ranked:
            norm = raw_score / (doc_top1 + 1e-9)
            merged.append((doc_idx, raw_score, "doc", norm))

        merged.sort(key=lambda x: x[3], reverse=True)

        for doc_idx, raw_score, pool, _norm in merged[:top_k]:
            meta = dict(self.documents[doc_idx].get("metadata", {}))
            meta["pool"] = pool
            results.append((self.documents[doc_idx]["text"], raw_score, meta))

        return results

    # ================================================================
    # B.3: 查询改写
    # ================================================================
    def _rewrite_query(self, question: str, history: list[dict] | None = None) -> tuple[str | None, dict | None]:
        """历史感知的查询改写。

        将 {history} 变量注入 query_rewrite 模板，使 LLM 能消解指代。
        history 为空时传递占位文本。

        Returns:
            (keywords, usage) — keywords 为改写后的检索词（失败时 None），
            usage 为本次 LLM 调用的 token 统计（未调 LLM 时 None）。
        """
        if self.llm.is_mock:
            return None, None
        try:
            from .prompts.registry import render
            history_str = self._format_history_for_rewrite(history)
            prompt = render("query_rewrite", question=question, history=history_str)
            resp = self.llm.chat_with_meta(
                prompt["system"], prompt["user"],
                temperature=prompt["temperature"], max_tokens=prompt["max_tokens"],
                prompt_id=prompt["id"], prompt_version=prompt["version"],
                caller="rag",
            )
            keywords = resp["response"].strip()
            usage = {
                "prompt_tokens": resp.get("prompt_tokens", 0) or 0,
                "completion_tokens": resp.get("completion_tokens", 0) or 0,
                "cache_hit_tokens": resp.get("cache_hit_tokens", 0) or 0,
            }
            if not keywords or keywords.startswith("[MOCK LLM]"):
                return None, usage
            return keywords, usage
        except Exception as exc:
            self.logger.warning("Query rewrite failed: %s", exc)
            return None, None

    # ================================================================
    # B.4: 问答链路（多轮对话 + function calling 编排 + 诚实兜底）
    # ================================================================
    def ask(self, question: str, history: list[dict] | None = None,
            top_k: int = 5) -> Dict[str, Any]:
        """多轮对话问答入口。

        流程：
        1. 工具调用轮（B2）：带 tools 调 LLM，dispatch 结果，最多 2 轮循环
        2. 工具成功 → route="tool"，直接返回
        3. 工具未调用 / 全部 found=false / 超时 → RAG 检索（B3）
        4. 检索为空 → 诚实兜底（B4）：强制注入 capability + coverage 文档
        """
        import json as _json
        import time as _time
        from .tools import dispatch as _dispatch
        from .llm.client import sanitize_assistant_message, make_deadline, check_deadline

        t_start = _time.perf_counter()
        rewrite_keywords: str | None = None
        rewrite_applied = False
        rewrite_reason: str | None = None
        tool_calls_log: list[dict] = []
        sources: list = []
        route: str = "rag"
        route_reason: str = ""
        answer: str = ""

        # ---- token / cost 追踪（与 app.py _trace_cost 同一公式）----
        _TRACE_PROMPT_PRICE_PER_1K = 0.001       # CNY / 1K input tokens (cache miss)
        _TRACE_COMPLETION_PRICE_PER_1K = 0.002    # CNY / 1K output tokens
        _TRACE_CACHE_HIT_RATIO = 1.0 / 50          # cache hit input ~1/50 cost

        usage: dict[str, int] = {
            "prompt_tokens": 0, "completion_tokens": 0,
            "cache_hit_tokens": 0, "n_llm_calls": 0,
        }

        def _acc_tokens(resp: dict) -> None:
            """从 chat_messages / chat_with_meta 返回的 dict 累加 token。"""
            usage["prompt_tokens"] += resp.get("prompt_tokens", 0) or 0
            usage["completion_tokens"] += resp.get("completion_tokens", 0) or 0
            usage["cache_hit_tokens"] += resp.get("cache_hit_tokens", 0) or 0
            usage["n_llm_calls"] += 1

        # ----------------------------------------------------------
        # B2: 工具调用轮
        # ----------------------------------------------------------
        try:
            from .prompts.registry import render
            qa_prompt = render("rag_qa", context="", question=question, retrieval_note="")
        except Exception:
            qa_prompt = {"system": "你是经济数据分析助手。优先使用工具查询数据。", "temperature": 0.3, "max_tokens": 1200, "version": "fallback"}

        # 构建消息列表
        system_msg = qa_prompt["system"]
        compacted_history = self._compact_history(history)
        messages: list[dict] = [{"role": "system", "content": system_msg}]
        messages.extend(compacted_history)
        messages.append({"role": "user", "content": question})

        deadline = make_deadline()
        max_tool_rounds = 2
        _TOOL_DECISION_MAX_TOKENS = 400
        any_tool_called = False
        any_found_or_available = False
        tool_decision = {"round0_finish_reason": "", "round0_completion_tokens": 0,
                         "round0_retried": False}

        # --- 概念/方法论问题前置路由：高置信度判定则跳过工具循环 ---
        skip_tool_loop = _is_methodology_question(question)
        if skip_tool_loop:
            tool_decision = {"round0_finish_reason": "skipped",
                             "round0_completion_tokens": 0,
                             "round0_retried": False}
            route_reason = "methodology_bypass"
            self.logger.info("概念问题前置路由：跳过工具循环 question=%.80s", question)

        for tool_round in range(max_tool_rounds):
            if skip_tool_loop:
                break
            if not check_deadline(deadline):
                self.logger.warning("Tool loop timeout after round %d", tool_round)
                break

            round_max_tokens = _TOOL_DECISION_MAX_TOKENS if tool_round == 0 else 1200
            resp = self.llm.chat_messages(
                messages, tools=self._tool_schemas,
                temperature=0.1, max_tokens=round_max_tokens,
                prompt_id="rag_qa", prompt_version=qa_prompt.get("version", "unknown"),
                caller="rag_tool",
            )
            _acc_tokens(resp)

            # --- 记录 tool_decision 诊断信息 ---
            if tool_round == 0:
                tool_decision["round0_completion_tokens"] = resp.get("completion_tokens", 0)

            # --- 截断护栏：round-0 被截断则用 1200 重发一次 ---
            if tool_round == 0 and resp.get("finish_reason") == "length":
                self.logger.warning(
                    "工具决策轮输出被截断（max_tokens=%d），用 1200 重发",
                    _TOOL_DECISION_MAX_TOKENS,
                )
                resp = self.llm.chat_messages(
                    messages, tools=self._tool_schemas,
                    temperature=0.1, max_tokens=1200,
                    prompt_id="rag_qa", prompt_version=qa_prompt.get("version", "unknown"),
                    caller="rag_tool",
                )
                _acc_tokens(resp)
                tool_decision["round0_retried"] = True
                tool_decision["round0_completion_tokens"] = resp.get("completion_tokens", 0)

            if tool_round == 0:
                tool_decision["round0_finish_reason"] = resp.get("finish_reason", "")

            if resp.get("tool_calls"):
                any_tool_called = True
                # 清洗 assistant 消息并追加
                clean_msg = sanitize_assistant_message(
                    resp.get("raw_message") or {
                        "role": "assistant",
                        "content": resp.get("content"),
                        "tool_calls": resp["tool_calls"],
                    }
                )
                messages.append(clean_msg)

                for tc in resp["tool_calls"]:
                    fn_name = tc["function"]["name"]
                    try:
                        fn_args = _json.loads(tc["function"]["arguments"])
                    except _json.JSONDecodeError:
                        self.logger.warning(
                            "工具参数解析失败 tool=%s raw=%s",
                            fn_name, tc["function"]["arguments"][:200],
                        )
                        fn_args = {"_parse_error": tc["function"]["arguments"]}
                        tool_result = {"error": "工具参数解析失败", "found": False}
                    else:
                        try:
                            tool_result = _dispatch(self, fn_name, fn_args)
                        except Exception as exc:
                            tool_result = {"error": str(exc)}
                    tc_entry = {
                        "name": fn_name,
                        "arguments": fn_args,
                        "result_summary": self._summarize_tool_result(tool_result),
                    }
                    tool_calls_log.append(tc_entry)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": _json.dumps(tool_result, ensure_ascii=False),
                    })
                    if tool_result.get("found") is True or tool_result.get("available") is True:
                        any_found_or_available = True
            else:
                # LLM 没有发起 tool call
                if tool_round == 0:
                    # 首轮仅用于工具决策，正文一律丢弃，转 RAG 路径
                    answer = ""
                    self.logger.debug("首轮未调用工具，丢弃决策轮正文，转 RAG 检索")
                else:
                    answer = resp.get("content", "")
                break

        # 工具调用后取最终答案
        if any_tool_called and not answer:
            if check_deadline(deadline):
                resp_final = self.llm.chat_messages(
                    messages, temperature=0.3, max_tokens=1200,
                    prompt_id="rag_qa", prompt_version=qa_prompt.get("version", "unknown"),
                    caller="rag_tool",
                )
                _acc_tokens(resp_final)
                answer = resp_final.get("content", "")
            else:
                answer = "（工具查询超时，请稍后重试）"

        # 判定 route
        if any_tool_called and any_found_or_available and answer.strip():
            route = "tool"
            route_reason = "tool_hit"
        elif any_tool_called:
            route_reason = "tool_miss_fallback"
        elif not skip_tool_loop:
            route_reason = "no_tool_call"
        # 若 skip_tool_loop 为 True，route_reason 已在前面设为 methodology_bypass

        # 工具路径 found=false 但有 candidates → 把 candidate 信息注入后续 context
        tool_candidates_context = ""
        if any_tool_called and not any_found_or_available:
            for tc in tool_calls_log:
                cands = tc.get("result_summary", {}).get("candidates", [])
                if cands:
                    cand_strs = [f"{c['name']}（{c.get('region', '')}）" for c in cands]
                    tool_candidates_context = (
                        "工具查询未精确匹配，但发现以下候选指标："
                        + "、".join(cand_strs)
                        + "。请向用户确认具体需要哪一个。\n"
                    )
                    break

        # ----------------------------------------------------------
        # B3 / B4: RAG 检索 + 诚实兜底
        # ----------------------------------------------------------
        if route != "tool":
            # 查询改写
            if self.llm.is_mock:
                rewrite_reason = "LLM 不可用"
            else:
                try:
                    kw, rw_usage = self._rewrite_query(question, history)
                    if rw_usage is not None:
                        _acc_tokens(rw_usage)
                    if kw:
                        rewrite_keywords = kw
                        rewrite_applied = True
                except Exception as exc:
                    rewrite_reason = f"改写异常: {exc}"

            search_query = rewrite_keywords if rewrite_applied else question
            sources = self.search(search_query, top_k)

            # 元问题加分：命中时 capability 文档 +0.3
            if _is_meta_question(question):
                boosted: list = []
                for text, score, meta in sources:
                    if meta.get("type") == "capability":
                        boosted.append((text, score + _META_CAPABILITY_BOOST, meta))
                    else:
                        boosted.append((text, score, meta))
                boosted.sort(key=lambda x: x[1], reverse=True)
                sources = boosted

            # 改写降级：改写结果为空时回退原始 query
            if not sources and rewrite_applied:
                self.logger.debug("Rewrite keywords '%s' returned empty, falling back to raw query",
                                  search_query)
                sources = self.search(question, top_k)

            if sources:
                # --- B3: RAG 检索命中 ---
                context_parts: list[str] = []
                for i, (text, score, meta) in enumerate(sources):
                    context_parts.append(f"[{i+1}] {text}")
                context = "\n".join(context_parts)

                retrieval_note = tool_candidates_context if tool_candidates_context else ""
                from .prompts.registry import render as _render
                rag_prompt = _render("rag_qa", context=context, question=question,
                                     retrieval_note=retrieval_note)
                rag_resp = self.llm.chat_with_meta(
                    rag_prompt["system"], rag_prompt["user"],
                    temperature=rag_prompt["temperature"],
                    max_tokens=rag_prompt["max_tokens"],
                    prompt_id=rag_prompt["id"], prompt_version=rag_prompt["version"],
                    caller="rag",
                )
                answer = rag_resp["response"]
                _acc_tokens(rag_resp)
                route = "rag"
            else:
                # --- B4: 诚实兜底 ---
                cap_docs = [d for d in self.documents
                           if d.get("metadata", {}).get("type") in ("capability", "coverage")]
                if cap_docs:
                    context = "\n\n".join(d["text"] for d in cap_docs)
                else:
                    context = "（系统能力说明暂不可用）"
                retrieval_note = (
                    "检索未命中任何数据文档。以下仅为系统能力与数据覆盖说明，"
                    "不含用户所问的具体数据。"
                )
                if tool_candidates_context:
                    retrieval_note = tool_candidates_context + retrieval_note

                from .prompts.registry import render as _render
                rag_prompt = _render("rag_qa", context=context, question=question,
                                     retrieval_note=retrieval_note)
                rag_resp = self.llm.chat_with_meta(
                    rag_prompt["system"], rag_prompt["user"],
                    temperature=rag_prompt["temperature"],
                    max_tokens=rag_prompt["max_tokens"],
                    prompt_id=rag_prompt["id"], prompt_version=rag_prompt["version"],
                    caller="rag",
                )
                answer = rag_resp["response"]
                _acc_tokens(rag_resp)
                route = "rag_no_hit"

        # 计算成本（与 app.py _trace_cost 同一公式）
        miss = usage["prompt_tokens"] - usage["cache_hit_tokens"]
        est_cost = (
            miss * _TRACE_PROMPT_PRICE_PER_1K / 1000
            + usage["cache_hit_tokens"] * _TRACE_PROMPT_PRICE_PER_1K * _TRACE_CACHE_HIT_RATIO / 1000
            + usage["completion_tokens"] * _TRACE_COMPLETION_PRICE_PER_1K / 1000
        )
        usage_out = {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "cache_hit_tokens": usage["cache_hit_tokens"],
            "total_tokens": usage["prompt_tokens"] + usage["completion_tokens"],
            "est_cost_cny": round(est_cost, 6),
            "n_llm_calls": usage["n_llm_calls"],
        }

        elapsed = _time.perf_counter() - t_start
        return {
            "answer": answer,
            "sources": [{"text": t, "score": s, "metadata": m} for t, s, m in sources],
            "rewrite_keywords": rewrite_keywords if rewrite_applied else None,
            "rewrite_applied": rewrite_applied,
            "rewrite_reason": rewrite_reason,
            "route": route,
            "tool_calls": tool_calls_log,
            "tool_decision": tool_decision,
            "tool_loop_skipped": skip_tool_loop,
            "route_reason": route_reason,
            "n_history_used": len(compacted_history),
            "elapsed_s": round(elapsed, 2),
            "usage": usage_out,
        }

    # ================================================================
    # Function Calling 工具方法
    # ================================================================

    @staticmethod
    def _norm(text: str) -> str:
        return text.replace("_", "").replace(" ", "").replace("（", "").replace("）", "")

    @staticmethod
    def _indicator_fuzzy_match(query: str, target: str) -> float:
        if not query or not target:
            return 0.0
        q = RAGService._norm(query)
        t = RAGService._norm(target)
        if not q or not t:
            return 0.0
        q_chars = set(q)
        t_chars = set(t)
        if not q_chars.issubset(t_chars):
            # 部分匹配降级：查询含口径修饰词（"累计"/"同比"）而标准名中没有时，
            # 不再直接判零，但得分被压在 [0.65, 0.69] 这条窄带内。
            #   - coverage < 0.7 直接判零：滤掉真正不相关的指标
            #   - 上限 0.69：严格低于子集匹配的最低分 0.70，
            #     保证"完全包含"永远优先于"部分覆盖"
            #   - 下限效果：0.88*coverage 需 ≥ 0.65 才能过 query_indicator 的命中阈值，
            #     即实际生效的 coverage 门槛是 0.739，比表面的 0.7 更严
            coverage = len(q_chars & t_chars) / len(q_chars)
            if coverage < 0.7:
                return 0.0
            return min(0.69, 0.88 * coverage)
        if q == t:
            base = 1.00
        elif q in t:
            base = 0.90
        else:
            qi = 0
            for ch in t:
                if qi < len(q) and ch == q[qi]:
                    qi += 1
            base = 0.80 if qi == len(q) else 0.70
        penalty = min(0.10, 0.10 * (len(t) - len(q)) / max(len(t), 1))
        return max(0.0, base - penalty)

    def query_indicator(self, indicator: str, year: int,
                        quarter: int | None = None,
                        month: int | None = None,
                        region: str = "四川省") -> dict:
        normalized = normalize_query_aliases(indicator)

        scored_matches: list[tuple[float, str, dict]] = []
        seen_names: set[tuple[str, str]] = set()

        for doc in self.documents:
            meta = doc.get("metadata", {})
            if meta.get("type") != "indicator_card":
                continue
            doc_region = meta.get("region", "")
            if region and region != "" and doc_region != region:
                continue
            if meta.get("year") != year:
                continue
            if quarter is not None and meta.get("quarter") != quarter:
                continue
            if month is not None and meta.get("month") != month:
                continue

            doc_indicator = meta.get("indicator", "")
            if not doc_indicator:
                continue

            key = (doc_indicator, doc_region)
            if key in seen_names:
                continue
            seen_names.add(key)

            score = self._indicator_fuzzy_match(normalized, doc_indicator)
            if score > 0:
                scored_matches.append((score, doc_indicator, {
                    "name": doc_indicator, "region": doc_region, "score": score,
                    "value": meta.get("value"), "period": meta.get("date", ""),
                }))

        scored_matches.sort(key=lambda x: x[0], reverse=True)

        if not scored_matches:
            return {"found": False, "value": None, "matched_indicator": "", "period": "",
                    "score": 0.0, "candidates": [], "caliber_note": ""}

        top1_score = scored_matches[0][0]
        top2_score = scored_matches[1][0] if len(scored_matches) >= 2 else 0.0
        gap = top1_score - top2_score
        positive_only = [s for s in scored_matches if s[0] > 0]
        n_positive = len(positive_only)

        if top1_score < 0.65:
            candidates = [{"name": info["name"], "region": info["region"],
                           "score": round(sc, 4), "value": info["value"]}
                          for sc, _, info in positive_only[:3]]
            return {"found": False, "value": None, "matched_indicator": "", "period": "",
                    "score": top1_score, "candidates": candidates, "caliber_note": ""}

        if n_positive == 1 or gap >= 0.08:
            info = scored_matches[0][2]
            matched_name = scored_matches[0][1]
            caliber_note = self._caliber_notes.get(matched_name, "")
            return {"found": True, "value": info["value"], "matched_indicator": matched_name,
                    "period": info["period"], "score": top1_score, "candidates": [],
                    "caliber_note": caliber_note}

        candidates = [{"name": info["name"], "region": info["region"],
                       "score": round(sc, 4), "value": info["value"]}
                      for sc, _, info in positive_only[:3]]
        return {"found": False, "value": None, "matched_indicator": "", "period": "",
                "score": top1_score, "candidates": candidates, "caliber_note": ""}

    def list_indicators(self) -> dict:
        indicators: dict[tuple[str, str], dict] = {}
        for doc in self.documents:
            meta = doc.get("metadata", {})
            if meta.get("type") != "indicator_card":
                continue
            name = meta.get("indicator", "")
            region = meta.get("region", "")
            date_str = meta.get("date", "")
            source = meta.get("source_file", "")
            freq = "quarterly" if "quarterly" in source else "monthly"
            key = (name, region)
            if key not in indicators:
                indicators[key] = {"name": name, "region": region, "freq": freq,
                                   "start": date_str, "end": date_str, "n_obs": 0}
            entry = indicators[key]
            entry["n_obs"] += 1
            if date_str < entry["start"]:
                entry["start"] = date_str
            if date_str > entry["end"]:
                entry["end"] = date_str
        return {
            "indicators": sorted(indicators.values(), key=lambda x: (x["name"], x["region"])),
            "aliases": get_indicator_aliases(),
        }

    def get_model_leaderboard(self) -> dict:
        if self.engine is None:
            return {"available": False, "reason": "预测引擎未初始化，无法获取模型排行榜"}
        leaderboard = getattr(self.engine, "leaderboard", None) or []
        if not leaderboard:
            return {"available": False, "reason": "预测引擎尚未运行完成"}
        selected = getattr(self.engine, "selected_model_name", None)
        models: list[dict] = []
        for rank, entry in enumerate(leaderboard, 1):
            if isinstance(entry, dict):
                name, rmse, mae = entry.get("model_name", ""), float(entry.get("rmse", 0)), float(entry.get("mae", 0))
                mape = float(entry.get("mape", 0)); smape = float(entry.get("smape", 0))
                r2 = float(entry.get("r2", 0)); n_train = int(entry.get("n_train", 0)); n_valid = int(entry.get("n_valid", 0))
            elif hasattr(entry, "model_name"):
                name = entry.model_name; rmse = float(getattr(entry, "rmse", 0)); mae = float(getattr(entry, "mae", 0))
                mape = float(getattr(entry, "mape", 0)); smape = float(getattr(entry, "smape", 0))
                r2 = float(getattr(entry, "r2", 0)); n_train = int(getattr(entry, "n_train", 0)); n_valid = int(getattr(entry, "n_valid", 0))
            else:
                continue
            models.append({
                "rank": rank, "name": str(name), "rmse": rmse, "mae": mae,
                "mape": mape, "smape": smape, "r2": r2, "n_train": n_train, "n_valid": n_valid,
                "is_selected": bool(selected and name == selected),
                "is_baseline": str(name) in ("last_value", "mean_recent"),
            })
        return {"available": True, "leaderboard": models,
                "caliber_note": ("此处 RMSE 为验证集差分口径，与 32 窗口 expanding-window 回测 level 口径不可直接比较。"
                                 "验证集差分 RMSE 用于模型选择，回测 level RMSE 用于绝对精度评估。")}

    def get_prediction(self) -> dict:
        if self.engine is None:
            return {"available": False, "reason": "预测引擎未初始化，无法获取预测结果"}
        try:
            pred = getattr(self.engine, "latest_prediction", None)
            if pred is None:
                return {"available": False, "reason": "预测引擎尚未运行完成"}
            ci = pred.get("confidence_interval")
            ci_available = ci is not None and isinstance(ci, dict)
            return {
                "available": True,
                "target_quarter": pred.get("nowcast_quarter", pred.get("prediction_quarter", "")),
                "prediction_value": pred.get("prediction_value"),
                "ci_available": ci_available,
                "ci_lower": ci.get("lower") if ci_available else None,
                "ci_upper": ci.get("upper") if ci_available else None,
                "ci_method": ci.get("method", "") if ci_available else "",
                "model_name": pred.get("model_name", ""),
                "notes": pred.get("notes", []),
            }
        except Exception as exc:
            return {"available": False, "reason": f"获取预测失败: {exc}"}

    # ================================================================
    # 历史精简与工具结果摘要（B5 / B2 辅助）
    # ================================================================

    @staticmethod
    def _compact_history(history: list[dict] | None) -> list[dict]:
        """精简对话历史以节省 token。

        - assistant 消息原样保留
        - user 消息只保留原始问题，剥离当轮拼过的 context 块
        - 超过 6 条时只取最近 6 条
        - 每条超过 500 字时截断并加 "…（已截断）"
        """
        if not history:
            return []
        compacted: list[dict] = []
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                # 剥离可能嵌入的 context 块
                content = RAGService._strip_context_from_user_message(content)
            if len(content) > 500:
                content = content[:500] + "…（已截断）"
            compacted.append({"role": role, "content": content})
        if len(compacted) > 6:
            compacted = compacted[-6:]
        return compacted

    @staticmethod
    def _strip_context_from_user_message(content: str) -> str:
        """从 user 消息中移除嵌入的检索资料块。

        匹配模式：以"检索资料："开头到"问题："之间的内容，
        以及以"【检索资料】"开头到"【/检索资料】"之间的内容。
        """
        import re as _re
        # 模式 1: "检索资料：...\n问题："
        content = _re.sub(r'检索资料[：:]\s*\n.*?(?=问题[：:])', '', content, flags=_re.DOTALL)
        # 模式 2: 编号的检索条目 [1] ... [2] ... (长 context 拼入)
        content = _re.sub(r'(?:^|\n)\[\d+\]\s[^\n]+(?=\n(?:\[\d+\]|问题[：:]|$))', '', content)
        content = content.strip()
        return content

    @staticmethod
    def _format_history_for_rewrite(history: list[dict] | None) -> str:
        """将历史转为查询改写提示词可用的字符串。

        无历史时返回占位文本；有历史时取最近 4 条，每条截断到 120 字。
        """
        if not history:
            return "（无历史，这是第一轮提问）"
        recent = history[-4:]
        lines: list[str] = []
        for msg in recent:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if len(content) > 120:
                content = content[:120] + "…"
            label = "用户" if role == "user" else "助手"
            lines.append(f"[{label}] {content}")
        return "\n".join(lines)

    @staticmethod
    def _summarize_tool_result(result: dict) -> dict:
        """从工具返回的 dict 提取摘要，供返回值的 tool_calls 字段使用。"""
        summary: dict = {}
        if "found" in result:
            summary["found"] = result["found"]
        if "available" in result:
            summary["available"] = result["available"]
        if "value" in result:
            summary["value"] = result["value"]
        if "matched_indicator" in result and result.get("matched_indicator"):
            summary["matched_indicator"] = result["matched_indicator"]
        if "candidates" in result and result["candidates"]:
            summary["candidates"] = [
                {"name": c["name"], "region": c.get("region", "")}
                for c in result["candidates"][:3]
            ]
        if "caliber_note" in result and result.get("caliber_note"):
            summary["caliber_note"] = result["caliber_note"]
        summary["found_or_available"] = (
            result.get("found") is True or result.get("available") is True
        )
        return summary


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
