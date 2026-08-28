from __future__ import annotations

import sys
import traceback

# --- Phase 1: bare-minimum imports for error display ---
try:
    import streamlit as st
except ImportError as e:
    # streamlit itself is missing — nothing we can do; let it crash with a clear message
    raise RuntimeError(f"streamlit 未安装，请检查 requirements.txt: {e}") from e

try:
    import streamlit.components.v1 as _st_components
except ImportError:
    _st_components = None

st.set_page_config(
    page_title="四川省 GDP 混频预测系统",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- 诊断计时器（仅本步诊断用；默认关闭，SC_MACRO_DEBUG_TIMING=true 时开启） ---
import os as _os
import time as _time
_TIMING_ON = _os.environ.get("SC_MACRO_DEBUG_TIMING", "").lower() == "true"
_T0 = _time.perf_counter()
def _tick(label: str) -> None:
    if not _TIMING_ON:
        return
    import sys
    print(f"[TIMING] {label}: {_time.perf_counter() - _T0:.2f}s", file=sys.stderr, flush=True)

# --- Phase 2: all other imports, with error display ---
try:
    from pathlib import Path
    from typing import Any, Dict, Iterable, Tuple

    import inspect
    import json

    from dotenv import load_dotenv
    load_dotenv()

    import os
    # Streamlit Secrets 桥接：本地走 .env，云端走 st.secrets（client.py 不依赖 streamlit）
    try:
        if "DEEPSEEK_API_KEY" in st.secrets and not os.environ.get("DEEPSEEK_API_KEY"):
            os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        pass

    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    from sc_macro_agent import AppConfig, PredictionEngine
    _IMPORT_OK = True
except Exception as _import_err:
    _IMPORT_OK = False
    st.error(f"## 导入失败: {_import_err}")
    st.code(traceback.format_exc())
    st.stop()

_tick("imports_done")

# ---- 版本标识（cloud 环境下 mtime 不可靠，用版本号）----
APP_VERSION = "chat-2.0.0"


def _build_stamp() -> str:
    """返回应用与 RAG 服务的版本标识。

    云端部署时 git checkout 会把所有文件 mtime 设成同一值，
    因此改用版本号 + git SHA（若可得）。
    """
    from sc_macro_agent import rag_service as _rs

    rag_ver = getattr(_rs, "RAG_SERVICE_VERSION", "?")
    stamp = f"app {APP_VERSION} · rag {rag_ver}"

    # git commit 短 hash（本地或 STREAMLIT_GIT_SHA 环境变量）
    try:
        sha = os.environ.get("STREAMLIT_GIT_SHA")
        if not sha:
            import subprocess
            r = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=2,
            )
            if r.returncode == 0:
                sha = r.stdout.strip()
        if sha:
            stamp += f" · {sha[:8]}"
    except Exception:
        pass
    return stamp


# ==================== 设计系统 ====================
COLORS = {
    "bg_primary": "#090c10",
    "bg_secondary": "#111318",
    "bg_tertiary": "#1a1d24",
    "border": "#2a2e37",
    "border_hover": "#3f4450",
    "text_primary": "#f0f2f5",
    "text_secondary": "#9ca3af",
    "text_muted": "#6b7280",
    "accent_cyan": "#22d3ee",
    "accent_cyan_dim": "#0891b2",
    "accent_purple": "#a78bfa",
    "accent_green": "#34d399",
    "accent_amber": "#fbbf24",
    "accent_red": "#f87171",
}

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background: {COLORS["bg_primary"]};
        color: {COLORS["text_primary"]};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }}
    #MainMenu {{visibility: hidden;}}
    [data-testid="stHeaderLogo"] {{display: none;}}
    /* 注意：不能整体隐藏 header！Streamlit 1.46+ 侧边栏折叠时的展开按钮
       （stExpandSidebarButton）渲染在 header 内，隐藏 header 会连带杀死展开入口 */
    footer {{visibility: hidden;}}
    .block-container {{
        max-width: 1400px;
        padding: 2rem 3rem 3rem 3rem;
    }}
    .minimal-header {{
        margin-bottom: 2.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid {COLORS["border"]};
    }}
    .minimal-header h1 {{
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: {COLORS["text_primary"]};
        margin: 0;
    }}
    .minimal-header .subtitle {{
        font-size: 0.9rem;
        color: {COLORS["text_secondary"]};
        margin-top: 0.5rem;
        font-weight: 400;
    }}
    .status-badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.35rem 0.9rem;
        background: {COLORS["bg_secondary"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 999px;
        font-size: 0.8rem;
        color: {COLORS["accent_cyan"]};
        font-weight: 600;
        margin-top: 1rem;
    }}
    .status-badge::before {{
        content: "";
        width: 6px;
        height: 6px;
        background: {COLORS["accent_cyan"]};
        border-radius: 50%;
        box-shadow: 0 0 8px {COLORS["accent_cyan"]};
    }}
    .kpi-card {{
        background: linear-gradient(180deg, {COLORS["bg_secondary"]} 0%, {COLORS["bg_primary"]} 100%);
        border: 1px solid {COLORS["border"]};
        border-radius: 16px;
        padding: 1.75rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    .kpi-card:hover {{
        border-color: {COLORS["accent_cyan_dim"]};
        box-shadow: 0 0 20px rgba(34, 211, 238, 0.15), 0 4px 6px rgba(0, 0, 0, 0.3);
        transform: translateY(-2px);
    }}
    .kpi-card::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    }}
    .kpi-card .label {{
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {COLORS["text_muted"]};
        margin-bottom: 0.75rem;
        font-weight: 600;
    }}
    .kpi-card .value {{
        font-size: 2.5rem;
        font-weight: 800;
        color: {COLORS["text_primary"]};
        line-height: 1;
        letter-spacing: -0.02em;
        font-feature-settings: "tnum";
    }}
    .kpi-card .meta {{
        margin-top: 0.75rem;
        font-size: 0.85rem;
        color: {COLORS["text_secondary"]};
        line-height: 1.5;
    }}
    .kpi-card.accent-left {{ border-left: 3px solid {COLORS["accent_cyan"]}; }}
    .kpi-card.accent-purple {{ border-left: 3px solid {COLORS["accent_purple"]}; }}
    .kpi-card.accent-green {{ border-left: 3px solid {COLORS["accent_green"]}; }}
    .section-header {{
        margin-bottom: 1.25rem;
        margin-top: 2rem;
    }}
    .section-label {{
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {COLORS["accent_cyan"]};
        font-weight: 700;
        margin-bottom: 0.25rem;
    }}
    .section-title {{
        font-size: 1.25rem;
        font-weight: 700;
        color: {COLORS["text_primary"]};
        letter-spacing: -0.01em;
    }}
    .section-desc {{
        font-size: 0.9rem;
        color: {COLORS["text_secondary"]};
        margin-top: 0.35rem;
        line-height: 1.5;
    }}
    .metric-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.8rem;
        background: {COLORS["bg_tertiary"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        font-size: 0.85rem;
        color: {COLORS["text_secondary"]};
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }}
    .metric-pill .dot {{
        width: 6px;
        height: 6px;
        border-radius: 50%;
    }}
    .metric-pill .dot.good {{ background: {COLORS["accent_green"]}; }}
    .metric-pill .dot.warn {{ background: {COLORS["accent_amber"]}; }}
    .subtle-divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, {COLORS["border"]}, transparent);
        margin: 2rem 0;
        border: none;
    }}
    [data-testid="stSidebar"] {{
        background: {COLORS["bg_secondary"]};
        border-right: 1px solid {COLORS["border"]};
    }}
    [data-testid="stSidebar"] .block-container {{
        padding: 2rem 1.5rem;
    }}
    /* 折叠态展开按钮强化（streamlit 1.61 testid：stExpandSidebarButton）：
       胶囊样式 + 发光描边 + 文字提示，确保深色主题下一眼可见 */
    [data-testid="stExpandSidebarButton"] {{
        visibility: visible !important;
        position: fixed;
        top: 1rem;
        left: 0.75rem;
        z-index: 1000;
        background: {COLORS["bg_secondary"]};
        border: 1px solid {COLORS["accent_cyan_dim"]};
        border-radius: 999px;
        padding: 0.4rem 1rem;
        box-shadow: 0 0 12px rgba(34, 211, 238, 0.25);
    }}
    [data-testid="stExpandSidebarButton"]:hover {{
        border-color: {COLORS["accent_cyan"]};
        box-shadow: 0 0 18px rgba(34, 211, 238, 0.45);
    }}
    [data-testid="stExpandSidebarButton"]::after {{
        content: "展开导航";
        font-size: 0.8rem;
        color: {COLORS["accent_cyan"]};
        letter-spacing: 0.05em;
        margin-left: 0.4rem;
    }}
    .stRadio > div {{
        background: {COLORS["bg_tertiary"]};
        border-radius: 12px;
        padding: 0.5rem;
    }}
    .stRadio label {{
        color: {COLORS["text_secondary"]} !important;
        font-size: 0.9rem;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        transition: all 0.2s;
    }}
    .stRadio label:hover {{
        background: {COLORS["bg_secondary"]};
        color: {COLORS["text_primary"]} !important;
    }}
    .stButton > button {{
        background: {COLORS["bg_tertiary"]};
        color: {COLORS["text_primary"]};
        border: 1px solid {COLORS["border"]};
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s;
    }}
    .stButton > button:hover {{
        background: {COLORS["accent_cyan_dim"]};
        border-color: {COLORS["accent_cyan"]};
        color: white;
        box-shadow: 0 0 20px rgba(34, 211, 238, 0.15);
    }}
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
        border-bottom: 1px solid {COLORS["border"]};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {COLORS["text_secondary"]};
        font-weight: 600;
        padding: 1rem 0;
        letter-spacing: -0.01em;
    }}
    .stTabs [aria-selected="true"] {{
        color: {COLORS["accent_cyan"]} !important;
        border-bottom: 2px solid {COLORS["accent_cyan"]} !important;
    }}
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: {COLORS["bg_primary"]};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {COLORS["border"]};
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {COLORS["border_hover"]};
    }}
    /* 导航 emoji 字体统一，避免部分字形回退到单色符号字体导致宽度不一致 */
    section[data-testid="stSidebar"] div[role="radiogroup"] label p,
    div[data-testid="stExpander"] div[data-baseweb="select"] div {{
        font-family: "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji",
                     "Segoe UI Symbol", -apple-system, "PingFang SC",
                     "Microsoft YaHei", sans-serif;
        font-variant-emoji: emoji;
    }}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 自愈脚本：Streamlit 会把侧边栏折叠状态持久化到 localStorage（key: stSidebarCollapsed-*），
# 一旦记住折叠，initial_sidebar_state="expanded" 也会被覆盖。
# 每个浏览器会话首次加载时清除该标记并自动展开侧边栏；
# 之后用户仍可手动折叠，不会被反复弹开。
if _st_components is not None:
    _st_components.html("""
    <script>
    (function() {
      var KEY = "scMacroSidebarAutoExpandDone";
      try {
        if (window.parent.sessionStorage.getItem(KEY) === "1") return;
        window.parent.sessionStorage.setItem(KEY, "1");
        Object.keys(window.parent.localStorage)
          .filter(function(k) { return k.indexOf("stSidebarCollapsed-") === 0; })
          .forEach(function(k) { window.parent.localStorage.removeItem(k); });
      } catch (e) {}
      function tryExpand(n) {
        var doc = window.parent.document;
        if (doc.querySelector('[data-testid="stSidebarCollapseButton"]')) return; // 已展开
        var btn = doc.querySelector('[data-testid="stExpandSidebarButton"]');
        if (btn) { btn.click(); return; }
        if ((n || 0) < 20) { setTimeout(function() { tryExpand((n || 0) + 1); }, 250); }
      }
      setTimeout(function() { tryExpand(0); }, 300);
    })();
    </script>
    """, height=0)


def fmt_number(value: Any, digits: int = 2, default: str = "-") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_pct_decimal(value: Any, digits: int = 1, default: str = "-") -> str:
    if value is None:
        return default
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


# 预测注记展示层映射：prediction_engine 产出的英文常量 → 中文（数据层契约不改，仅展示层翻译）
_NOTE_TRANSLATIONS = {
    "prediction_generated_from_latest_available_quarter_features": "预测基于最新可用季度的特征生成",
    "if_real_data_is_short_treat_as_demo_nowcast_not_production_forecast": "样本量有限，本结果应视为演示性 nowcast，非生产级预测",
}


def _format_prediction_note(note: str) -> str:
    """把英文注记常量映射为中文，未命中的原样显示。"""
    return _NOTE_TRANSLATIONS.get(note, note)


def _step_brief(step: dict) -> str:
    """Agent 步骤的一行摘要，用于折叠区表格。"""
    name = step.get("name", "?")
    meta = step.get("result") or step.get("review") or {}
    if "data" in name:
        return f"模式={meta.get('latest_quarter','?')}，{meta.get('usable_indicators',0)}指标，{'OK' if meta.get('data_ok') else '阻断'}"
    if "model" in name:
        return f"{meta.get('model_name','?')}，RMSE={meta.get('backtest_rmse','?')}，预测={meta.get('prediction_value','?'):.1f}"
    if "analyst" in name:
        return f"生成简报（含 metrics + indicators 上下文）"
    if "critic" in name:
        if meta.get("critic_error"):
            return "解析失败"
        if meta.get("passed"):
            return f"通过，{len(meta.get('issues',[]))} 个低优提示"
        return f"不通过，{len(meta.get('issues',[]))} 个问题"
    return "-"


def safe_df(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])
    return pd.DataFrame()


def mapping_df(mapping: Dict[str, Any]) -> pd.DataFrame:
    if not mapping:
        return pd.DataFrame(columns=["字段", "值"])
    # 值统一转字符串：混型列（str/int/list 混杂）会让 st.dataframe 的
    # pyarrow 序列化报 ArrowTypeError（Expected bytes, got a 'int' object）
    return pd.DataFrame([{"字段": k, "值": str(v)} for k, v in mapping.items()])


def render_section(label: str, title: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="section-header">
            <div class="section-label">{label}</div>
            <div class="section-title">{title}</div>
            <div class="section-desc">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_warning_pills(items: Iterable[str]) -> None:
    html = "".join(
        f"<span class='metric-pill'><span class='dot warn'></span>{str(x)}</span>" for x in items if str(x).strip())
    if html:
        st.markdown(html, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_engine() -> PredictionEngine:
    """纯计算函数：初始化引擎并跑完整流水线。不含任何 st.* UI 调用。
    UI 由 main() 负责渲染；缓存命中时函数体整体跳过，幂等。
    """
    import os, sys
    _tick("load_engine:enter")
    if _TIMING_ON:
        print(f"[ENV] SC_MACRO_LIGHT_MODE={os.environ.get('SC_MACRO_LIGHT_MODE')}", file=sys.stderr, flush=True)
    config = AppConfig.from_env()
    engine = PredictionEngine(config=config)
    _tick("load_engine:engine_constructed")
    light_mode = os.environ.get("SC_MACRO_LIGHT_MODE", "").lower() == "true"
    _tick("load_engine:before_run_agent")
    try:
        engine.initialize()
        if light_mode:
            engine.audit_data(save_artifacts=False)
            engine.build_features()
        else:
            engine.audit_data(save_artifacts=False)
            engine.build_features()
            engine.train()
            try:
                engine.backtest()
            except Exception as bt_exc:
                engine.warnings.append(str(bt_exc))
                engine.agent.record_warning(str(bt_exc))
            engine.predict_next()
    except Exception as e:
        print(f"[WARN] Pipeline init failed: {e}", file=sys.stderr)
        engine._init_error = str(e)
    _tick("load_engine:after_run_agent")
    _tick("load_engine:return")
    return engine


# 共享可变状态说明：
# st.cache_resource 返回跨会话共享的同一个对象引用，load_view_data 返回的 dict
# 及其中的 engine 会被所有会话共用。
#   - 本项目为单人演示场景，共享引擎可接受；
#   - 生产环境需改为每会话独立实例或加锁（该条已同步至 known_limitations.md）。
@st.cache_resource(show_spinner=False)
def load_view_data(_: int) -> Dict[str, Any]:
    _tick("load_view_data:enter")
    engine = load_engine()
    # 先取预测（predict 内部触发 train 时会赋值 selected_model），
    # 再 summarize，确保 summary 里 selected_model / leaderboard / top_features 有值
    try:
        prediction = getattr(engine, "latest_prediction", None) or engine.predict_next()
    except Exception:
        prediction = None
    _tick("load_view_data:after_predict_next")
    try:
        status = engine.get_status()
        _tick("load_view_data:after_get_status")
        summary = engine.summarize()
        _tick("load_view_data:after_summarize")
    except Exception:
        status = {"phase": "init_error", "completed": False}
        summary = {}
    audit_result = getattr(engine, "audit_result", None) or {}
    backtest = getattr(engine, "backtest_result", None) or {}
    factor_summary = engine.get_factor_summary() if hasattr(engine, "get_factor_summary") else {}
    _tick("load_view_data:after_factor_summary")
    try:
        snapshot = engine.data_manager.get_latest_snapshot()
    except Exception:
        snapshot = None
    try:
        availability = engine.data_manager.get_data_availability()
    except Exception:
        availability = {"items": []}
    _tick("load_view_data:after_data_availability")
    try:
        signal_overview = engine.data_manager.build_training_signal_overview()
    except Exception:
        signal_overview = {}
    _tick("load_view_data:after_signal_overview")
    leaderboard_df = safe_df(summary.get("leaderboard", []))
    top_features_df = safe_df(summary.get("top_features", []))
    agent_steps_df = safe_df(summary.get("agent", {}).get("steps", []))
    window_df = safe_df(backtest.get("window_results", []))
    checks_df = safe_df(audit_result.get("checks", []))
    items_df = safe_df(availability.get("items", []))
    summary_df = safe_df(audit_result.get("summary", []))

    try:
        registry = engine.feature_artifacts.feature_registry if engine.feature_artifacts is not None else None
    except Exception:
        registry = None
    family_df = pd.DataFrame(columns=["family", "count"])
    region_df = pd.DataFrame(columns=["region", "count"])
    if registry is not None:
        try:
            family_df = pd.DataFrame(list(registry.summary_by_family().items()), columns=["family", "count"])
            region_df = pd.DataFrame(list(registry.summary_by_region().items()), columns=["region", "count"])
        except Exception:
            pass

    _tick("load_view_data:return")
    return {
        "engine": engine,
        "status": status,
        "summary": summary,
        "prediction": prediction,
        "audit": audit_result,
        "backtest": backtest,
        "factor_summary": factor_summary,
        "snapshot": snapshot,
        "availability": availability,
        "signal_overview": signal_overview,
        "leaderboard_df": leaderboard_df,
        "top_features_df": top_features_df,
        "agent_steps_df": agent_steps_df,
        "window_df": window_df,
        "checks_df": checks_df,
        "items_df": items_df,
        "summary_df": summary_df,
        "family_df": family_df,
        "region_df": region_df,
    }


def feature_score_df(top_features_df: pd.DataFrame) -> pd.DataFrame:
    if top_features_df.empty:
        return pd.DataFrame()
    candidates = [
        "combined_score",
        "abs_coefficient",
        "normalized_importance",
        "residual_tree_importance",
        "coefficient",
        "importance",
        "score",
    ]
    score_col = next((c for c in candidates if c in top_features_df.columns), None)
    if score_col is None or "feature" not in top_features_df.columns:
        return pd.DataFrame()
    out = top_features_df[["feature", score_col]].copy()
    out = out.rename(columns={score_col: "score"}).sort_values("score", ascending=False)
    return out.head(12)


# ==================== 图表函数（修复重复参数问题） ====================
def apply_base_style(fig: go.Figure) -> go.Figure:
    """应用基础深色样式，不含特定布局参数"""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9ca3af", family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"),
        hoverlabel=dict(
            bgcolor="#111318",
            bordercolor="#2a2e37",
            font=dict(color="#f0f2f5"),
        ),
    )
    fig.update_xaxes(
        gridcolor="#2a2e37",
        linecolor="#2a2e37",
        zerolinecolor="#2a2e37",
        tickfont=dict(color="#6b7280"),
        title_font=dict(color="#9ca3af"),
    )
    fig.update_yaxes(
        gridcolor="#2a2e37",
        linecolor="#2a2e37",
        zerolinecolor="#2a2e37",
        tickfont=dict(color="#6b7280"),
        title_font=dict(color="#9ca3af"),
    )
    return fig


def create_leaderboard_chart(df: pd.DataFrame) -> go.Figure | None:
    if df.empty or "model_name" not in df.columns:
        return None
    score_col = next((c for c in ["rmse", "mae", "score", "mape"] if c in df.columns), None)
    if score_col is None:
        return None
    plot_df = df[["model_name", score_col]].copy().sort_values(score_col, ascending=True)

    fig = px.bar(
        plot_df,
        x=score_col,
        y="model_name",
        orientation="h",
        text_auto=True,
    )

    fig.update_traces(
        marker_color="#22d3ee",
        textposition="outside",
        textfont=dict(color="#f0f2f5", size=11),
        hovertemplate=f"<b>%{{y}}</b><br>{score_col.upper()}: %{{x:.4f}}<extra></extra>",
    )

    fig = apply_base_style(fig)
    fig.update_layout(
        height=300,
        margin=dict(l=120, r=40, t=10, b=10),
        xaxis_title=f"<b>{score_col.upper()}</b>",
        yaxis_title="",
        showlegend=False,
    )
    return fig


def create_backtest_line(window_df: pd.DataFrame) -> go.Figure | None:
    if window_df.empty or not {"actual", "prediction"}.issubset(window_df.columns):
        return None
    plot_df = window_df.copy()
    x_col = "test_quarter" if "test_quarter" in plot_df.columns else plot_df.index

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_df[x_col],
        y=plot_df["actual"],
        mode="lines+markers",
        name="真实值",
        line=dict(width=3, color="#f0f2f5"),
        marker=dict(size=6, color="#f0f2f5", line=dict(width=2, color="#111318")),
        hovertemplate="<b>真实值</b><br>%{y:.2f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=plot_df[x_col],
        y=plot_df["prediction"],
        mode="lines+markers",
        name="预测值",
        line=dict(width=3, color="#22d3ee", dash="dash"),
        marker=dict(size=6, color="#22d3ee", line=dict(width=2, color="#111318")),
        hovertemplate="<b>预测值</b><br>%{y:.2f}<extra></extra>",
    ))

    fig = apply_base_style(fig)
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        xaxis_title="<b>季度</b>",
        yaxis_title="<b>GDP 数值</b>",
        hovermode="x unified",
    )
    fig.update_xaxes(tickangle=-45, nticks=8)
    return fig


def create_error_bar(window_df: pd.DataFrame) -> go.Figure | None:
    if window_df.empty or not {"actual", "prediction"}.issubset(window_df.columns):
        return None
    plot_df = window_df.copy()
    plot_df["abs_error"] = (plot_df["actual"] - plot_df["prediction"]).abs()
    plot_df["error_pct"] = (plot_df["abs_error"] / plot_df["actual"] * 100).round(2)
    x_col = "test_quarter" if "test_quarter" in plot_df.columns else plot_df.index

    colors = []
    for val in plot_df["abs_error"]:
        if val < plot_df["abs_error"].median() * 0.5:
            colors.append("#34d399")
        elif val < plot_df["abs_error"].median():
            colors.append("#fbbf24")
        else:
            colors.append("#f87171")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot_df[x_col],
        y=plot_df["abs_error"],
        marker_color=colors,
        text=plot_df["error_pct"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside",
        textfont=dict(color="#9ca3af", size=10),
        hovertemplate="<b>%{x}</b><br>绝对误差: %{y:.3f}<br>相对误差: %{text}<extra></extra>",
    ))

    fig = apply_base_style(fig)
    fig.update_xaxes(tickangle=-45, nticks=8)
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis_title="<b>季度</b>",
        yaxis_title="<b>绝对误差</b>",
        showlegend=False,
        bargap=0.3,
    )
    return fig


def create_feature_chart(score_df: pd.DataFrame) -> go.Figure | None:
    if score_df.empty:
        return None
    plot_df = score_df.sort_values("score", ascending=True).tail(12)

    fig = px.bar(
        plot_df,
        x="score",
        y="feature",
        orientation="h",
    )

    fig.update_traces(
        marker=dict(
            color=plot_df["score"],
            colorscale=[[0, "#a78bfa"], [1, "#22d3ee"]],
            line=dict(width=0),
        ),
        texttemplate="%{x:.3f}",
        textposition="outside",
        textfont=dict(color="#f0f2f5", size=10),
        hovertemplate="<b>%{y}</b><br>重要性: %{x:.4f}<extra></extra>",
    )

    fig = apply_base_style(fig)
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis_title="<b>重要性得分</b>",
        yaxis_title="",
        showlegend=False,
    )
    return fig


def create_factor_variance_chart(factor_summary: Dict[str, Any]) -> go.Figure | None:
    ratios = factor_summary.get("explained_variance_ratio") or []
    if not ratios:
        return None
    df = pd.DataFrame({
        "factor": [f"Factor {i + 1}" for i in range(len(ratios))],
        "ratio": ratios,
    })
    df["cumulative"] = df["ratio"].cumsum()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df["factor"],
        y=df["ratio"],
        name="单因子解释力",
        marker=dict(color="#22d3ee", opacity=0.8, line=dict(width=0)),
        hovertemplate="<b>%{x}</b><br>解释方差: %{y:.1%}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df["factor"],
        y=df["cumulative"],
        name="累计解释力",
        mode="lines+markers",
        line=dict(color="#a78bfa", width=3),
        marker=dict(size=8, color="#a78bfa", line=dict(width=2, color="#111318")),
        hovertemplate="<b>%{x}</b><br>累计: %{y:.1%}<extra></extra>",
    ))

    fig = apply_base_style(fig)
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(
            title="<b>解释方差比例</b>",
            tickformat=".0%",
            range=[0, 1],
        ),
        xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        bargap=0.4,
    )
    return fig


def create_availability_chart(items_df: pd.DataFrame) -> go.Figure | None:
    if items_df.empty:
        return None
    ratio_col = next((c for c in ["coverage_ratio", "non_missing_ratio", "missing_ratio"] if c in items_df.columns),
                     None)
    name_col = next((c for c in ["table", "name", "dataset", "frequency"] if c in items_df.columns), None)
    if ratio_col is None or name_col is None:
        return None
    plot_df = items_df[[name_col, ratio_col]].copy()
    if ratio_col == "missing_ratio":
        plot_df[ratio_col] = 1 - plot_df[ratio_col].astype(float)

    colors = []
    for val in plot_df[ratio_col]:
        if val > 0.9:
            colors.append("#34d399")
        elif val > 0.7:
            colors.append("#fbbf24")
        else:
            colors.append("#f87171")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot_df[name_col],
        y=plot_df[ratio_col],
        marker_color=colors,
        text=plot_df[ratio_col].apply(lambda x: f"{x:.0%}"),
        textposition="outside",
        textfont=dict(color="#9ca3af"),
        hovertemplate="<b>%{x}</b><br>可用比例: %{y:.1%}<extra></extra>",
    ))

    fig = apply_base_style(fig)
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="",
        yaxis=dict(
            title="<b>数据可用性</b>",
            tickformat=".0%",
            range=[0, 1.1],
        ),
        showlegend=False,
        bargap=0.4,
    )
    return fig


def create_component_pie(components: Dict[str, Any]) -> go.Figure | None:
    if not components:
        return None
    values = []
    names = []
    for k, v in components.items():
        try:
            values.append(abs(float(v)))
            names.append(k)
        except (TypeError, ValueError):
            continue
    if not values:
        return None

    colors = ["#22d3ee", "#a78bfa", "#34d399", "#fbbf24", "#f87171"]

    fig = px.pie(
        values=values,
        names=names,
        hole=0.55,
        color_discrete_sequence=colors,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont=dict(color="#f0f2f5", size=11),
        hovertemplate="<b>%{label}</b><br>数值: %{value:.2f}<br>占比: %{percent}<extra></extra>",
        marker=dict(line=dict(color="#111318", width=2)),
    )

    fig = apply_base_style(fig)
    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=False,
        annotations=[dict(
            text="构成",
            x=0.5, y=0.5,
            font_size=16,
            font_color="#9ca3af",
            showarrow=False,
        )],
    )
    return fig


def create_registry_pie(df: pd.DataFrame, name_col: str) -> go.Figure | None:
    if df.empty:
        return None

    colors = ["#22d3ee", "#a78bfa", "#34d399", "#fbbf24", "#f87171", "#60a5fa"]

    fig = px.pie(
        df,
        values="count",
        names=name_col,
        hole=0.6,
        color_discrete_sequence=colors,
    )

    fig.update_traces(
        textposition="outside",
        textinfo="label+percent",
        textfont=dict(color="#9ca3af", size=11),
        hovertemplate="<b>%{label}</b><br>数量: %{value}<br>占比: %{percent}<extra></extra>",
        marker=dict(line=dict(color="#111318", width=2)),
    )

    fig = apply_base_style(fig)
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig


# ==================== 侧边栏控制 ====================
# ---- 导航双向同步（单一真相源 current_page）----
def _sync_page_from(widget_key: str) -> None:
    """导航 widget 变更时写回统一的路由状态。"""
    st.session_state.current_page = st.session_state[widget_key]


def _prime_nav_widget(widget_key: str) -> None:
    """把 widget 的 state 对齐到 current_page。

    Streamlit 在 key 已存在时忽略 index 参数，只能通过直接写 session_state 同步。
    必须在对应 widget 创建**之前**调用，之后调用会抛
    "cannot be modified after the widget is instantiated"。
    """
    cur = st.session_state.current_page
    if st.session_state.get(widget_key) != cur:
        st.session_state[widget_key] = cur


PAGE_NAMES = [
    "🏠 概览驾驶舱", "🔮 现时预测", "📈 历史回测", "🔍 因子分析",
    "🧪 数据质量", "🔧 Agent 工作流", "📝 AI 简报", "💬 数据问答", "📊 LLM 追踪",
]

def sidebar_controls(data: Dict[str, Any]) -> Tuple[str, bool]:
    st.sidebar.markdown(
        f"""
        <div style="margin-bottom: 2rem;">
            <div style="font-size: 1.1rem; font-weight: 800; color: {COLORS["text_primary"]}; margin-bottom: 0.5rem;">
                Sichuan GDP
            </div>
            <div style="font-size: 0.85rem; color: {COLORS["text_muted"]}; line-height: 1.6;">
                Mixed-Frequency Nowcasting System<br>
                <span style="color: {COLORS["accent_cyan"]};">Agent Pipeline v2.0</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 与 main() 中顶部导航共用 session_state.current_page 作为单一真相源
    st.session_state.setdefault("current_page", PAGE_NAMES[0])
    _prime_nav_widget("_nav_side")
    page = st.sidebar.radio(
        "导航",
        PAGE_NAMES,
        key="_nav_side",
        on_change=_sync_page_from,
        args=("_nav_side",),
        label_visibility="collapsed",
    )

    st.sidebar.markdown(
        f'<div style="height: 1px; background: linear-gradient(90deg, transparent, {COLORS["border"]}, transparent); margin: 1.5rem 0;"></div>',
        unsafe_allow_html=True)

    status = data["status"]
    summary = data["summary"]

    st.sidebar.markdown(
        f'<div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: {COLORS["text_muted"]}; margin-bottom: 1rem; font-weight: 700;">系统状态</div>',
        unsafe_allow_html=True)

    # 用 markdown 渲染系统状态，替代 st.metric（避免窄屏/长模型名截断）
    model_name = summary.get("selected_model") or "-"
    dataset_mode = status.get("dataset_mode") or "-"
    n_rows = status.get("n_rows") or 0
    n_features_active = status.get("n_features_active") or status.get("n_features") or 0
    n_features_total = status.get("n_features_total") or status.get("n_features") or 0
    st.sidebar.markdown(
        f'<div style="font-size:0.85rem;color:{COLORS["text_primary"]};margin-bottom:0.6rem;">'
        f'<b>模型</b>&nbsp;{model_name}&ensp;|&ensp;<b>模式</b>&nbsp;{dataset_mode}'
        f'</div>'
        f'<div style="font-size:0.85rem;color:{COLORS["text_primary"]};">'
        f'<b>样本</b>&nbsp;{n_rows} 行&ensp;·&ensp;<b>特征</b>&nbsp;{n_features_active}/{n_features_total} 维'
        f'</div>',
        unsafe_allow_html=True)

    # Prediction info
    prediction = data["prediction"]
    if prediction and prediction.get("prediction_value"):
        st.sidebar.markdown(
            f'<div style="height: 1px; background: linear-gradient(90deg, transparent, {COLORS["border"]}, transparent); margin: 1.5rem 0;"></div>',
            unsafe_allow_html=True)
        st.sidebar.markdown(
            f'<div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: {COLORS["text_muted"]}; margin-bottom: 0.5rem; font-weight: 700;">最新预测</div>',
            unsafe_allow_html=True)
        pred_val = prediction.get("prediction_value", 0)
        pred_q = prediction.get("prediction_quarter", "?")
        # 轻量模式下未跑回测，confidence_interval 可能为 None（key 存在但值为 None），需兜底
        ci = prediction.get("confidence_interval") or {}
        st.sidebar.markdown(
            f'<div style="font-size: 1.2rem; font-weight: 800; color: {COLORS["accent_cyan"]};">{pred_q}: {fmt_number(pred_val, 2)}%</div>',
            unsafe_allow_html=True)
        st.sidebar.caption(f"90% CI [{fmt_number(ci.get('lower'))}, {fmt_number(ci.get('upper'))}] | {prediction.get('target_transform','?')}")
        cs = prediction.get("chronos_state", "?")
        if cs == "ready":
            chronos_label = f"TSLM 残差修正：{prediction.get('chronos_correction',0):+.3f}"
        elif cs == "failed":
            reason = prediction.get("chronos_failure_reason", "")
            if "未安装" in str(reason) or "依赖" in str(reason):
                chronos_label = "TSLM 残差修正：部署环境未启用（无 torch）"
            else:
                chronos_label = f"TSLM 残差修正：{reason}"
        else:
            chronos_label = "TSLM 残差修正：未加载"
        st.sidebar.caption(chronos_label)

    st.sidebar.markdown(
        f'<div style="height: 1px; background: linear-gradient(90deg, transparent, {COLORS["border"]}, transparent); margin: 1.5rem 0;"></div>',
        unsafe_allow_html=True)

    refresh = st.sidebar.button("⟳ 刷新数据", use_container_width=True)

    if st.sidebar.button("⬇ 导出结果", use_container_width=True):
        exported = data["engine"].export_artifacts()
        st.sidebar.success("已导出 artifacts")
        for name, path in exported.items():
            st.sidebar.caption(f"{name}: {Path(path).name}")

    warnings = list(dict.fromkeys((status.get("warnings") or []) + (data["audit"].get("warnings") or [])))
    if warnings:
        st.sidebar.markdown(
            f'<div style="margin-top: 1.5rem;"><span style="font-size: 0.75rem; text-transform: uppercase; color: {COLORS["accent_amber"]}; font-weight: 700;">警告 ({len(warnings)})</span></div>',
            unsafe_allow_html=True)
        for item in warnings[:3]:
            st.sidebar.markdown(
                f'<div style="font-size: 0.8rem; color: {COLORS["text_secondary"]}; margin-top: 0.5rem; padding: 0.5rem; background: {COLORS["bg_tertiary"]}; border-left: 2px solid {COLORS["accent_amber"]}; border-radius: 4px;">{item}</div>',
                unsafe_allow_html=True)

    return page, refresh


# ==================== 页面渲染 ====================
def render_hero(data: Dict[str, Any]) -> None:
    prediction = data["prediction"]
    summary = data["summary"]

    st.markdown(
        f"""
        <div class="minimal-header">
            <h1>四川省 GDP 混频预测驾驶舱</h1>
            <div class="subtitle">
                基于动态因子模型与混合频率采样的实时预测系统 | 
                当前模型: <b>{summary.get('selected_model', 'Unknown')}</b> | 
                预测季度: <b>{prediction.get('prediction_quarter', '-')}</b>
            </div>
            <div class="status-badge">
                System Operational
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(data: Dict[str, Any]) -> None:
    prediction = data["prediction"]
    summary = data["summary"]
    metrics = data["backtest"].get("metrics", {}) or {}
    leaderboard_df = data["leaderboard_df"]
    window_df = data["window_df"]
    snapshot = data["snapshot"]
    warnings = list(dict.fromkeys((data["status"].get("warnings") or []) + (data["audit"].get("warnings") or [])))

    if warnings:
        render_warning_pills(warnings)
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    ci = prediction.get("confidence_interval", {}) or {}
    status = data["status"]
    n_active = status.get("n_features_active", status.get("n_features", 0))
    n_total = status.get("n_features_total", status.get("n_features", 0))

    with c1:
        st.markdown(
            f"""
            <div class="kpi-card accent-left">
                <div class="label">当前季度预测值</div>
                <div class="value">{fmt_number(prediction.get('prediction_value'))}</div>
                <div class="meta">预测季度：{prediction.get('prediction_quarter') or '-'}<br>置信区间：[{fmt_number(ci.get('lower'))}, {fmt_number(ci.get('upper'))}]</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi-card accent-purple">
                <div class="label">模型配置</div>
                <div class="value" style="font-size: 1.8rem;">{summary.get('selected_model') or '-'}</div>
                <div class="meta">训练样本：{summary.get('n_rows') or 0} 行<br>特征：{n_active} 维（使用） / {n_total} 维（面板）</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        rmse = metrics.get('rmse')
        r2 = metrics.get('r2')
        st.markdown(
            f"""
            <div class="kpi-card accent-green">
                <div class="label">回测性能 · 扩展窗口 · level 空间</div>
                <div class="value" style="font-size: 1.8rem;">RMSE {fmt_number(rmse, 3) if rmse else '-'}</div>
                <div class="meta">R² = {fmt_number(r2, 3) if r2 else '-'} | 32 窗口 | level 口径</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        dir_acc = metrics.get('direction_accuracy')
        dir_acc_text = f"{dir_acc:.1%}" if dir_acc is not None else '-'
        st.markdown(
            f"""
            <div class="kpi-card accent-left" style="border-left: 3px solid {COLORS['accent_amber']};">
                <div class="label">方向准确率 · level 回测</div>
                <div class="value">{dir_acc_text}</div>
                <div class="meta">预测变化方向与实际一致的比例<br>n = {metrics.get('direction_pairs', 31)} 个方向对</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='height: 1px; background: linear-gradient(90deg, transparent, #2a2e37, transparent); margin: 2rem 0;'></div>",
        unsafe_allow_html=True)

    left, right = st.columns([2, 1], gap="large")
    with left:
        render_section("Trend Analysis", "回测趋势对比", "真实值与预测值的时间序列对比，检验模型稳定性")
        fig = create_backtest_line(window_df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("当前没有可视化的回测窗口结果。")

    with right:
        render_section("Data Snapshot", "最新数据快照", "当前接入的最新季度与月度数据")
        st.dataframe(mapping_df(snapshot), use_container_width=True, hide_index=True, height=300)

    b1, b2 = st.columns([1, 1], gap="large")
    with b1:
        render_section("Model Selection", "候选模型对比", "验证集 · delta 空间 · 单次尾部切分（最近 12 季度）")
        fig = create_leaderboard_chart(leaderboard_df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        if not leaderboard_df.empty:
            st.caption("⚠ 此处 RMSE 为验证集 · delta 空间（Δy_t = y_t − y_{t-1}），与上方 KPI 的回测 · level 空间 RMSE 不可直接比较。leaderboard 仅用于模型选择排序，不反映 level 预测精度。")
            with st.expander("查看详细指标"):
                st.caption("MAPE 在目标值接近零时会失真（2020 年前后 GDP 累计同比曾降至 -3% 附近），本项目以 RMSE 与方向准确率为主要评价指标。")
                st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

    with b2:
        render_section("Prediction Summary", "预测摘要", "目标指标、基准值与最终预测")
        pred_data = {
            "目标指标": prediction.get("target_indicator"),
            "预测季度": prediction.get("prediction_quarter"),
            "预测值": fmt_number(prediction.get("prediction_value"), 2),
            "基准值": fmt_number(prediction.get("benchmark_value"), 2),
            "选用模型": prediction.get("model_name"),
            "最新数据季度": prediction.get("based_on_latest_quarter"),
        }
        st.dataframe(mapping_df(pred_data), use_container_width=True, hide_index=True)

        notes = prediction.get("notes", []) or []
        if notes:
            st.markdown(
                f'<div style="margin-top: 1rem; font-size: 0.8rem; color: {COLORS["text_muted"]};">预测注记</div>',
                unsafe_allow_html=True)
            for note in notes:
                st.markdown(
                    f'<div style="padding: 0.5rem 0; color: {COLORS["text_secondary"]}; font-size: 0.9rem; border-bottom: 1px solid {COLORS["border"]};">• {_format_prediction_note(note)}</div>',
                    unsafe_allow_html=True)


def render_nowcast(data: Dict[str, Any]) -> None:
    prediction = data["prediction"]
    components = prediction.get("components", {}) or {}

    render_section("Nowcast", "现时预测拆解", "预测值的构成分析：基准部分 + 模型修正")

    col1, col2, col3 = st.columns(3)
    ci = prediction.get("confidence_interval", {}) or {}

    with col1:
        st.markdown(
            f"""
            <div class="kpi-card accent-left">
                <div class="label">基准预测 (Benchmark)</div>
                <div class="value">{fmt_number(prediction.get('benchmark_value'))}</div>
                <div class="meta">基于上期目标或 AR 基准模型的线性外推</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        residual_val = None
        if components:
            base = prediction.get("benchmark_value")
            pred_val = prediction.get("prediction_value")
            try:
                residual_val = float(pred_val) - float(base)
            except (TypeError, ValueError):
                residual_val = None
        sign = "+" if residual_val and residual_val > 0 else ""
        st.markdown(
            f"""
            <div class="kpi-card accent-purple">
                <div class="label">模型修正幅度</div>
                <div class="value" style="color: {COLORS["accent_cyan"] if residual_val and residual_val > 0 else COLORS["accent_amber"]};">
                    {sign}{fmt_number(residual_val)}
                </div>
                <div class="meta">高频因子与混频模型带来的非线性调整</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi-card accent-green">
                <div class="label">最终预测值</div>
                <div class="value">{fmt_number(prediction.get('prediction_value'))}</div>
                <div class="meta">95% 置信区间：[{fmt_number(ci.get('lower'))}, {fmt_number(ci.get('upper'))}]</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div style='height: 1px; background: linear-gradient(90deg, transparent, #2a2e37, transparent); margin: 2rem 0;'></div>",
        unsafe_allow_html=True)

    # Chronos / TSLM 残差修正状态
    chronos_state = prediction.get("chronos_state", "unknown")
    try:
        chronos_correction = float(prediction.get("chronos_correction", 0.0))
    except (TypeError, ValueError):
        chronos_correction = 0.0

    if chronos_state == "ready":
        chronos_text = f"TSLM 残差修正：已启用，修正量 {chronos_correction:+.3f} 个百分点"
        chronos_color = COLORS["accent_cyan"]
    elif chronos_state == "failed":
        reason = str(prediction.get("chronos_failure_reason", ""))
        if "未安装" in reason or "依赖" in reason:
            chronos_text = "TSLM 残差修正：当前部署环境未启用（无 torch）"
        else:
            chronos_text = f"TSLM 残差修正：{reason}，已跳过"
        chronos_color = COLORS["accent_amber"]
    else:
        chronos_text = "TSLM 残差修正：未加载"
        chronos_color = COLORS["text_muted"]
    st.markdown(
        f'<div style="padding:0.5rem 1rem; background:{COLORS["bg_tertiary"]}; border-radius:8px; font-size:0.85rem; color:{chronos_color};">{chronos_text}</div>',
        unsafe_allow_html=True)

    # 若本地不可用，尝试展示静态参考结果
    if chronos_state == "failed":
        _ref_path = Path(__file__).parent / "assets" / "chronos_reference.json"
        if _ref_path.exists():
            try:
                _ref = json.loads(_ref_path.read_text(encoding="utf-8"))
                with st.expander("📄 查看 TSLM 本地参考结果（静态，非本次运行产生）", expanded=False):
                    st.caption(
                        f"⚠ 以下为预先计算的静态结果，非本次运行产生。"
                        f" 生成时间：{_ref.get('generated_at','?')[:19]}，"
                        f" 数据截止：{_ref.get('data_vintage','?')}。"
                        f" 部署环境未安装 torch，无法实时运行 TSLM 残差修正。"
                    )
                    ref_rows = []
                    for mname, r in (_ref.get("results") or {}).items():
                        ref_rows.append({
                            "模型": mname,
                            "RMSE": f'{r["rmse"]:.4f}',
                            "MAE": f'{r["mae"]:.4f}',
                            "R²": f'{r["r2"]:+.4f}',
                            "方向准确率": f'{r["direction_accuracy"]:.1%}',
                            "窗口数": r["n_windows"],
                        })
                    if ref_rows:
                        st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)
                        st.caption(
                            "Chronos (amazon/chronos-bolt-tiny) 残差修正在本地 32 窗口回测中"
                            " 未优于纯线性 ridge_midas，因此线上主预测不依赖 TSLM 修正。"
                        )
            except Exception:
                pass

    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        fig = create_component_pie(components)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("当前没有可视化的 prediction components。")

    with right:
        comp_df = mapping_df(components)
        if not comp_df.empty:
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
        else:
            st.caption("当前预测结果没有拆解项。")

        notes = prediction.get("notes", []) or []
        if notes:
            st.markdown(
                f'<div style="margin-top: 1.5rem; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; color: {COLORS["text_muted"]}; font-weight: 700;">模型注记</div>',
                unsafe_allow_html=True)
            for note in notes:
                st.markdown(
                    f'<div style="margin-top: 0.5rem; padding: 0.75rem; background: {COLORS["bg_tertiary"]}; border-radius: 8px; font-size: 0.9rem; color: {COLORS["text_secondary"]}; border-left: 3px solid {COLORS["accent_cyan"]};">{_format_prediction_note(note)}</div>',
                    unsafe_allow_html=True)


def render_backtest(data: Dict[str, Any]) -> None:
    metrics = data["backtest"].get("metrics", {}) or {}
    window_df = data["window_df"]
    summary = data["summary"]

    render_section("Backtest", "历史回测分析", "扩展窗口交叉验证 · level 空间 · 32 窗口")

    cols = st.columns(4)
    metric_items = [
        ("MAE · level", fmt_number(metrics.get("mae"), 3), COLORS["text_primary"]),
        ("RMSE · level", fmt_number(metrics.get("rmse"), 3), COLORS["accent_cyan"]),
        ("MAPE · level", fmt_pct_decimal(metrics.get("mape"), 2), COLORS["accent_amber"]),
        ("R² · level", fmt_number(metrics.get("r2"), 3), COLORS["accent_green"]),
    ]
    for col, (name, val, color) in zip(cols, metric_items):
        col.markdown(
            f"""
            <div style="text-align: center; padding: 1rem; background: {COLORS["bg_secondary"]}; border-radius: 12px; border: 1px solid {COLORS["border"]};">
                <div style="font-size: 0.75rem; color: {COLORS["text_muted"]}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">{name}</div>
                <div style="font-size: 1.5rem; font-weight: 800; color: {color}; font-feature-settings: 'tnum';">{val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        fig = create_backtest_line(window_df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("当前没有可视化的回测结果。")

    with right:
        fig = create_error_bar(window_df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("当前没有可视化的误差结果。")

    if not window_df.empty:
        with st.expander("查看详细窗口数据"):
            st.dataframe(window_df, use_container_width=True, hide_index=True)

    # ---- 基准模型对照表 ----
    baseline_comparison = data["backtest"].get("baseline_comparison", []) or []
    if baseline_comparison:
        st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
        render_section("Benchmark Comparison", "基准模型对比", "ARIMA / 朴素基准 与主模型的回测性能对照")
        main_model_name = data["summary"].get("selected_model", "主模型")

        # 构建表格行：主模型 + 各基准
        rows = []
        rows.append({
            "模型": f"⭐ {main_model_name}",
            "RMSE": fmt_number(metrics.get("rmse"), 3),
            "MAE": fmt_number(metrics.get("mae"), 3),
            "方向准确率": fmt_pct_decimal(metrics.get("direction_accuracy"), 1),
            "相对改进 (RMSE)": "—",
        })
        for bl in baseline_comparison:
            bl_name = bl.get("model_name", "?")
            imp = bl.get("rmse_improvement_pct", 0.0)
            imp_str = f"+{imp:.1f}%" if imp > 0 else (f"{imp:.1f}%" if imp < 0 else "0.0%")
            rows.append({
                "模型": bl_name,
                "RMSE": fmt_number(bl.get("rmse"), 3),
                "MAE": fmt_number(bl.get("mae"), 3),
                "方向准确率": fmt_pct_decimal(bl.get("direction_accuracy"), 1),
                "相对改进 (RMSE)": imp_str,
            })

        comparison_df = pd.DataFrame(rows)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        n_matched = sum(1 for bl in baseline_comparison if bl.get("n_windows") == data["backtest"].get("n_windows", -1))
        st.caption(
            f"相对改进 = (基准RMSE − 主模型RMSE) / 基准RMSE，正数表示主模型更优。"
            f" 窗口数一致: {n_matched}/{len(baseline_comparison)}。"
        )

    # ---- 方向准确率对比 ----
    from sc_macro_agent.models.backtesting import ExpandingWindowBacktester

    try:
        engine = load_engine()
        if engine.feature_artifacts is not None:
            panel = engine.feature_artifacts.training_panel.copy()
            panel_t, base_series = engine._apply_target_transform(panel)
            feat_cols = engine.feature_artifacts.feature_columns
            target_col = engine.feature_artifacts.target_column

            bt_cfg = engine.config.backtest
            mdl_cfg = engine.config.model
            backtester = ExpandingWindowBacktester(bt_cfg, mdl_cfg)

            dir_acc_rows = []
            all_names = list(mdl_cfg.candidate_models) + list(bt_cfg.baseline_models)
            seen = set()
            for mname in all_names:
                if mname in seen:
                    continue
                seen.add(mname)
                try:
                    bt_r = backtester.run(
                        panel=panel_t, feature_cols=feat_cols, target_col=target_col,
                        selected_model_name=mname, base_series=base_series,
                    )
                    da = bt_r["metrics"].get("direction_accuracy", 0.0)
                    dp = bt_r["metrics"].get("direction_pairs", 0)
                    dir_acc_rows.append({
                        "模型": mname,
                        "方向准确率": da,
                        "方向对数": dp,
                        "is_main": mname == (summary.get("selected_model") or ""),
                    })
                except Exception:
                    pass

            if dir_acc_rows:
                st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
                render_section(
                    "Direction Accuracy",
                    "方向准确率对比",
                    "预测的同比增速变化方向（相对上期上升/下降）与实际一致的比例",
                )
                # Build bars
                dir_acc_rows.sort(key=lambda x: x["方向准确率"], reverse=True)
                labels = [r["模型"] for r in dir_acc_rows]
                values = [r["方向准确率"] for r in dir_acc_rows]
                bar_colors = []
                for r in dir_acc_rows:
                    if r["is_main"]:
                        bar_colors.append(COLORS["accent_cyan"])
                    elif r["模型"] == "almon_midas":
                        bar_colors.append(COLORS["accent_purple"])
                    else:
                        bar_colors.append(COLORS["border_hover"])

                fig_dir = go.Figure()
                fig_dir.add_trace(go.Bar(
                    x=values,
                    y=labels,
                    orientation="h",
                    marker_color=bar_colors,
                    text=[f"{v:.1%}" for v in values],
                    textposition="outside",
                    textfont=dict(color=COLORS["text_primary"], size=11),
                    hovertemplate="<b>%{y}</b><br>方向准确率: %{x:.1%}<extra></extra>",
                ))
                fig_dir = apply_base_style(fig_dir)
                fig_dir.update_layout(
                    height=320,
                    margin=dict(l=150, r=40, t=10, b=10),
                    xaxis=dict(title="<b>方向准确率</b>", tickformat=".0%", range=[0, 1.0]),
                    yaxis=dict(title=""),
                    showlegend=False,
                    bargap=0.4,
                )
                st.plotly_chart(fig_dir, use_container_width=True, config={"displayModeBar": False})
                n_pairs = dir_acc_rows[0]["方向对数"] if dir_acc_rows else 31
                st.caption(
                    f"方向准确率 = 预测的 Δ 符号（正/负）与实际 Δ 符号一致的比例，"
                    f"n = {n_pairs} 个方向对。"
                    f" 青色 = 主模型（{summary.get('selected_model', '?')}），"
                    f"紫色 = almon_midas。"
                )
    except Exception:
        pass  # 取不到数据时整块跳过


def render_factors(data: Dict[str, Any]) -> None:
    top_features_df = data["top_features_df"]
    summary = data["summary"]

    render_section("Features & Importance", "特征分析与重要性", "白名单特征选择 + 模型系数排序（DFM 已停用）")

    # Show feature count from current model
    engine = load_engine()
    if engine.feature_artifacts is not None:
        feats = engine.feature_artifacts.feature_columns
        st.caption(f"当前特征集（{engine.config.features.target_transform} 模式）: {len(feats)} 个特征")
        # 实时从 feature_artifacts 取特征列表，不依赖 artifacts/final/feature_list.json
        fl_df = pd.DataFrame([{"feature": f} for f in feats])
        st.dataframe(fl_df, use_container_width=True, hide_index=True)

    score_df = feature_score_df(top_features_df)
    if not score_df.empty:
        fig = create_feature_chart(score_df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    elif not top_features_df.empty:
        st.dataframe(top_features_df, use_container_width=True, hide_index=True)
    else:
        st.info("当前没有特征重要性结果。运行模型训练后刷新。")

    # ---- MIDAS 权重曲线 ----
    leaderboard = summary.get("leaderboard", []) or []
    has_almon = any(
        (isinstance(e, dict) and e.get("model_name") == "almon_midas") or e == "almon_midas"
        for e in leaderboard
    )
    # Also check if selected model is almon_midas
    selected_model = summary.get("selected_model", "")
    almon_available = has_almon or selected_model == "almon_midas"

    if almon_available:
        try:
            engine = load_engine()
            if engine.feature_artifacts is not None and engine.selected_model is not None:
                from sc_macro_agent.models.almon_midas import AlmonMIDASModel

                # 获取当前训练面板和特征
                panel = engine.feature_artifacts.training_panel.copy()
                feature_cols = engine.feature_artifacts.feature_columns
                target_col = engine.feature_artifacts.target_column

                # 使用与 train() 一致的目标变换
                panel, _ = engine._apply_target_transform(panel)

                X_train = panel[feature_cols]
                y_train = panel[target_col]

                almon = AlmonMIDASModel(
                    theta_l2=engine.config.model.almon_theta_l2,
                    theta1_bounds=engine.config.model.almon_theta1_bounds,
                    theta2_bounds=engine.config.model.almon_theta2_bounds,
                )
                try:
                    almon.fit(X_train, y_train)
                    curves = almon.get_weight_curves()
                    if curves:
                        st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
                        render_section(
                            "MIDAS Almon Weights",
                            "MIDAS 权重曲线",
                            "Exponential Almon Lag 权重函数：B(k;θ) = exp(θ1·k + θ2·k²) / Σ exp(θ1·j + θ2·j²)，对应申报书公式 (2)",
                        )

                        # Check if any indicator has boundary issues
                        any_boundary = any(c.get("at_boundary", False) for c in curves)

                        fig = go.Figure()
                        colors = ["#22d3ee", "#a78bfa", "#34d399", "#fbbf24", "#f87171", "#60a5fa"]
                        for i, curve in enumerate(curves):
                            indicator = curve.get("indicator", "?")
                            weights = curve.get("weights", [])
                            theta1 = curve.get("theta1", 0.0)
                            theta2 = curve.get("theta2", 0.0)
                            beta = curve.get("beta", 0.0)
                            at_bound = curve.get("at_boundary", False)
                            ent = curve.get("weight_entropy", None)
                            if not weights:
                                continue
                            k_vals = list(range(len(weights)))

                            # Label: add ⚠ if at boundary
                            label = indicator[:36]
                            if at_bound:
                                label += " ⚠"

                            fig.add_trace(go.Scatter(
                                x=k_vals,
                                y=weights,
                                mode="lines+markers",
                                name=label,
                                line=dict(
                                    width=2.5,
                                    color=colors[i % len(colors)],
                                    dash="dash" if at_bound else "solid",
                                ),
                                marker=dict(size=6),
                                hovertemplate=(
                                    f"<b>{indicator}</b><br>"
                                    f"Lag %{{x}}: 权重 %{{y:.4f}}<br>"
                                    f"θ1={theta1:.3f}, θ2={theta2:.3f}, β={beta:.3f}"
                                    + (", 参数触界" if at_bound else "")
                                    + (f", ent={ent:.3f}" if ent is not None else "")
                                    + "<extra></extra>"
                                ),
                            ))

                        fig = apply_base_style(fig)
                        fig.update_layout(
                            height=380,
                            margin=dict(l=10, r=10, t=10, b=10),
                            xaxis=dict(
                                title="<b>滞后阶 k</b>（0 = 当季最新月）",
                                tickmode="linear",
                                dtick=1,
                            ),
                            yaxis=dict(title="<b>权重</b>", range=[0, 1.05]),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                x=0.5,
                                xanchor="center",
                                font=dict(size=10),
                            ),
                            hovermode="x unified",
                        )
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                        st.caption(
                            "每条曲线的权重和为 1。衰减速度由 θ1、θ2 决定，"
                            "对应申报书公式 (2) B(k;θ) = exp(θ1·k + θ2·k²) / Σ_j exp(θ1·j + θ2·j²)。"
                        )
                        if any_boundary:
                            st.warning(
                                "⚠ 部分指标参数触界——该指标的滞后结构在当前数据下未能稳定识别，"
                                "权重结果仅供参考。虚线 = 参数在可行域边界上。"
                            )
                except Exception:
                    pass  # 取不到数据时整块跳过，不报错
        except Exception:
            pass  # 取不到数据时整块跳过，不报错


def render_data_quality(data: Dict[str, Any]) -> None:
    items_df = data["items_df"]
    checks_df = data["checks_df"]
    summary_df = data["summary_df"]
    signal_overview = data["signal_overview"]

    render_section("Data Audit", "数据质量与可用性", "先审计后建模：覆盖率检查与质量报告")

    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        fig = create_availability_chart(items_df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("当前没有可视化的数据可用性结果。")

    with right:
        st.dataframe(mapping_df(signal_overview), use_container_width=True, hide_index=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📋 可用性明细", "🔍 质量检查", "📊 审计摘要", "📈 平稳性检验 (ADF)"])
    with tab1:
        if not items_df.empty:
            st.dataframe(items_df, use_container_width=True, hide_index=True)
        else:
            st.info("没有数据可用性明细。")
    with tab2:
        if not checks_df.empty:
            st.dataframe(checks_df, use_container_width=True, hide_index=True)
        else:
            st.info("没有质量检查结果。")
    with tab3:
        if not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            st.caption("当前没有审计摘要表。")
    with tab4:
        adf_tests = data["audit"].get("adf_tests", []) or []
        if adf_tests:
            # Build styled display
            adf_rows = []
            for item in adf_tests:
                stat_val = f'{item["adf_stat"]:.4f}' if item["adf_stat"] is not None else "-"
                p_val = f'{item["p_value"]:.4f}' if item["p_value"] is not None else "-"
                is_target = "目标变量" in item.get("indicator", "")
                adf_rows.append({
                    "指标": item["indicator"],
                    "n": item["n_obs"],
                    "ADF 统计量": stat_val,
                    "p 值": p_val,
                    "滞后阶": item.get("used_lag", "-") if item.get("used_lag") is not None else "-",
                    "1% 临界值": f'{item["critical_1pct"]:.4f}' if item.get("critical_1pct") is not None else "-",
                    "5% 临界值": f'{item["critical_5pct"]:.4f}' if item.get("critical_5pct") is not None else "-",
                    "结论": item["conclusion"],
                    "_is_stat": item.get("is_stationary", False),
                    "_is_target": is_target,
                })
            adf_df = pd.DataFrame(adf_rows)

            # Color-code: stationary=green, non-stationary=amber, target rows highlighted
            def _adf_style(row):
                styles = []
                for col in adf_df.columns:
                    if col.startswith("_"):
                        continue
                    if row.get("_is_target"):
                        styles.append("background-color: rgba(34,211,238,0.08); font-weight: 600")
                    elif row.get("_is_stat"):
                        styles.append("color: #34d399")
                    else:
                        styles.append("color: #fbbf24")
                return styles

            display_cols = [c for c in adf_df.columns if not c.startswith("_")]
            styled = adf_df[display_cols].style.apply(_adf_style, axis=1)
            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.caption(
                "原假设 H₀：序列存在单位根（非平稳）。"
                " p < 0.05 拒绝原假设，判定为平稳。"
                " 滞后阶由 AIC 自动选择。"
                " 青色高亮行 = 目标变量，绿色 = 平稳，琥珀色 = 非平稳。"
            )
        else:
            st.info("暂无 ADF 平稳性检验结果。运行数据审计后刷新。")


def render_agent(data: Dict[str, Any]) -> None:
    agent_steps_df = data["agent_steps_df"]
    summary = data["summary"]
    prediction = data["prediction"]

    render_section("Agent Workflow", "Agent 工作流与执行痕迹", "自动化流程的逐步执行记录与原始输出")

    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        if not agent_steps_df.empty:
            st.dataframe(agent_steps_df, use_container_width=True, hide_index=True, height=400)
        else:
            st.info("当前没有 Agent steps。")

    with right:
        agent_df = mapping_df(summary.get("agent", {}))
        st.dataframe(agent_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("📄 查看预测 JSON"):
            st.json(prediction, expanded=False)
    with col2:
        with st.expander("📄 查看运行总结 JSON"):
            st.json(summary, expanded=False)


# ==================== v2.1 新页面 ====================

def render_agent_v2(data: Dict[str, Any]) -> None:
    """Agent 过程页：四角色协作追踪。"""
    render_section("Agent Workflow", "AI Agent 协作流水线", "Data → Model → Analyst → Critic 四角色协作过程追踪")

    if st.button("▶ 启动 Agent 流水线", use_container_width=True):
        engine = load_engine()
        from sc_macro_agent.agents import AgentOrchestrator
        orch = AgentOrchestrator(config=engine.config)
        with st.status("Agent 流水线运行中...", expanded=True) as status_box:
            try:
                result = orch.run(engine, on_step=lambda msg: status_box.update(label=msg, state="running"))
            except Exception as exc:
                status_box.update(label="流水线执行失败", state="error")
                st.error(f"流水线执行失败：{exc}")
                return
            status_box.update(label=f"流水线完成 — 状态: {result.get('status','?')}", state="complete")

        final_status = result.get("status", "?")
        # Critic 是可选质检环节，其失败属于对外部模型服务的降级，不是系统故障。
        # 前端如实告知"未质检"，而非呈现为错误状态。
        if final_status == "review_failed":
            st.info("本次审阅环节未完成（模型服务响应超时），简报已正常生成。下方简报内容可正常查看，但未经自动质检。")

        # ---- 简报摘要（前 200 字 + 展开按钮） ----
        briefing = result.get("briefing", "")
        review = result.get("review", {})
        if review.get("critic_error"):
            st.caption("注：本次简报未经 Critic 自动质检")
        with st.expander("📄 简报摘要（点击展开全文）", expanded=False):
            st.markdown(briefing)
            st.download_button("⬇ 下载简报 (.md)", briefing,
                               file_name=f"briefing_{pd.Timestamp.now().strftime('%Y%m%d')}.md")
        st.caption(f"**简报预览**（{len(briefing)} 字）：{briefing[:200]}…")

        st.markdown("---")

        # ---- 协作时间线 ----
        st.markdown("### 四角色协作时间线")
        rewrite_rounds = result.get("rewrite_rounds", 0)
        if rewrite_rounds > 0:
            st.warning(f"共发生 {rewrite_rounds} 次审阅驳回后重写，详见下方 Critic 段。")

        for step in result.get("steps", []):
            name = step.get("name", "?")
            elapsed = step.get("elapsed_s", 0)
            meta = step.get("result") or step.get("review") or {}

            if "data_agent" in name:
                st.markdown(f"**🔍 DataAgent**  `{elapsed:.1f}s`")
                st.caption(f"数据模式：{meta.get('latest_quarter','?')}，{meta.get('usable_indicators',0)} 个可用指标，{'无阻断' if meta.get('data_ok') else '存在阻断问题'}")

            elif "model_agent" in name:
                st.markdown(f"**📊 ModelAgent**  `{elapsed:.1f}s`")
                parts = [f"模型：{meta.get('model_name','?')}"]
                if meta.get("backtest_rmse") is not None:
                    parts.append(f"回测 RMSE：{meta['backtest_rmse']:.2f}")
                if meta.get("direction_accuracy") is not None:
                    parts.append(f"方向准确率：{meta['direction_accuracy']:.1%}")
                if meta.get("vs_baseline_ratio") is not None:
                    parts.append(f"vs 基准比值：{meta['vs_baseline_ratio']:.2f}")
                st.caption(" | ".join(parts))
                st.caption(f"预测：{meta.get('prediction_quarter','?')} 增速 {meta.get('prediction_value','?'):.2f}% "
                           f"（实际 {meta.get('actual_value','?')}，误差 {meta.get('nowcast_error','?'):.2f}）")

            elif "analyst_agent" in name:
                st.markdown(f"**✍️ AnalystAgent**  `{elapsed:.1f}s`")
                bl = meta.get("briefing_length", len(briefing))
                st.caption(f"生成了 {bl} 字经济简报")

            elif "critic" in name:
                passed = meta.get("passed", False)
                icon = "✅" if passed else "⚠️"
                st.markdown(f"{icon} **CriticAgent**  `{elapsed:.1f}s`")
                if meta.get("critic_error"):
                    st.warning("审阅未完成 —— 模型服务响应超时，本次跳过自动质检")
                elif passed:
                    st.success(f"审阅通过 —— {meta.get('summary','')}")
                else:
                    st.warning(f"审阅未通过 —— {meta.get('summary','')}")
                issues = meta.get("issues", [])
                if issues:
                    issues_df = pd.DataFrame(issues)
                    st.dataframe(issues_df, use_container_width=True, hide_index=True)

            st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

        # ---- 脚标 ----
        u = result.get("token_usage", {})
        if u.get("is_mock"):
            st.warning("当前为降级模式输出，未调用真实模型（未配置 DEEPSEEK_API_KEY 或调用失败）。")
        st.caption(f"Token: {u.get('total_tokens',0)} | 费用: ¥{u.get('est_cost_cny',0):.4f} | 重写: {rewrite_rounds}轮")


def render_briefing_page(data: Dict[str, Any]) -> None:
    """AI 简报成果页：四角色协作的最终产出。"""
    render_section("AI Briefing", "AI 经济简报", "四角色协作生成的经济简报成果")

    st.markdown(f'<div style="padding:0.5rem 1rem;background:{COLORS["bg_tertiary"]};border-radius:8px;font-size:0.85rem;color:{COLORS["text_secondary"]};">'
                f'当前数据截至: {data.get("status",{}).get("dataset_mode","?")} 模式</div>',
                unsafe_allow_html=True)
    st.markdown("")

    if st.button("🤖 生成简报", use_container_width=True, type="primary"):
        engine = load_engine()
        from sc_macro_agent.agents import AgentOrchestrator
        orch = AgentOrchestrator(config=engine.config)
        with st.status("正在生成简报…", expanded=True) as s:
            try:
                result = orch.run(engine, on_step=lambda msg: s.update(label=msg, state="running"))
            except Exception as exc:
                s.update(label="流水线执行失败", state="error")
                st.error(f"流水线执行失败：{exc}")
                return
            s.update(label=f"简报生成完成 — 状态: {result.get('status','?')}", state="complete")

        final_status = result.get("status", "?")
        if final_status == "review_failed":
            st.info("本次审阅环节未完成（模型服务响应超时），简报已正常生成。下方简报内容可正常查看，但未经自动质检。")

        u = result.get("token_usage", {})
        if u.get("is_mock"):
            st.warning("当前为降级模式输出，未调用真实模型（未配置 DEEPSEEK_API_KEY 或调用失败）。")
        st.markdown(result.get("briefing", ""))

        # 审阅结论 + issues（折叠 steps 明细到展开区）
        rev = result.get("review", {})
        with st.expander("查看审阅结论与流水线步骤"):
            st.markdown("**审阅结论**")
            st.write(rev.get("summary", "无"))
            if rev.get("issues"):
                st.dataframe(pd.DataFrame(rev["issues"]), use_container_width=True, hide_index=True)
            elif rev.get("passed"):
                st.success("审阅通过，无问题。")
            if rev.get("critic_error"):
                st.warning("本次审阅未完成，简报未经自动质检")

            steps_df = pd.DataFrame([
                {"步骤": s.get("name", "?"), "耗时(s)": s.get("elapsed_s", 0),
                 "备注": _step_brief(s)}
                for s in result.get("steps", [])
            ])
            st.dataframe(steps_df, use_container_width=True, hide_index=True)

        st.caption(f"Token: {u.get('total_tokens',0)} | 费用: ¥{u.get('est_cost_cny',0):.4f} | 重写: {result.get('rewrite_rounds',0)}轮")


@st.cache_resource(show_spinner="正在构建检索索引…")
def get_rag_service():
    """RAG 服务模块级缓存：避免每次渲染重建 TF-IDF 语料。"""
    from sc_macro_agent.rag_service import RAGService
    engine = load_engine()
    return RAGService(config=engine.config, engine=engine)


def _submit_question(rag, q: str) -> None:
    """统一提交入口：预设按钮和 chat_input 都走此函数。

    先 append user 消息到 session_state（crash safety），
    再手动渲染用户气泡 + assistant 占位（spinner 在 assistant 内部），
    最后 append assistant 结果并 rerun 交由正常渲染循环接管。
    """
    msgs = st.session_state["rag_messages"]

    # 1. append user（即使在 ask() 崩溃后也能保留）
    msgs.append({"role": "user", "content": q, "meta": None})

    # 2. build history（不含刚 append 的这条）
    history = [{"role": m["role"], "content": m["content"]} for m in msgs[:-1]]

    # 3. 即时渲染：用户气泡 + assistant 占位（spinner 在气泡内部）
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("思考中…"):
            try:
                result = rag.ask(q, history=history)
            except Exception as exc:
                result = {
                    "answer": f"查询失败：{exc}",
                    "sources": [], "route": "rag_no_hit", "tool_calls": [],
                    "rewrite_keywords": None, "rewrite_applied": False,
                    "rewrite_reason": str(exc), "elapsed_s": 0, "n_history_used": 0,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": 0,
                              "total_tokens": 0, "est_cost_cny": 0, "n_llm_calls": 0},
                }

    # 4. append assistant
    msgs.append({
        "role": "assistant",
        "content": result["answer"],
        "meta": {
            "route": result.get("route", ""),
            "tool_calls": result.get("tool_calls", []),
            "rewrite_keywords": result.get("rewrite_keywords"),
            "rewrite_applied": result.get("rewrite_applied", False),
            "sources": result.get("sources", []),
            "usage": result.get("usage", {}),
            "elapsed_s": result.get("elapsed_s", 0),
        },
    })

    # 5. rerun —— 下一轮渲染循环接管全部历史
    st.rerun()


def _fmt_tool_badge(tool_calls: list[dict]) -> str:
    """格式化工具调用为单行徽标文本。"""
    parts: list[str] = []
    for tc in tool_calls:
        name = tc.get("name", "?")
        args = tc.get("arguments", {})
        summary = tc.get("result_summary", {})

        # 构建参数串
        arg_parts: list[str] = []
        indicator = args.get("indicator", "")
        if indicator:
            arg_parts.append(str(indicator))
        year = args.get("year")
        if year:
            arg_parts.append(str(year))
        quarter = args.get("quarter")
        if quarter:
            arg_parts.append(f"Q{quarter}")
        month = args.get("month")
        if month:
            arg_parts.append(f"{month}月")
        region = args.get("region")
        if region:
            arg_parts.append(str(region))
        args_str = ", ".join(arg_parts)

        # 结果摘要
        result_str = ""
        if summary.get("found") is True:
            val = summary.get("value")
            indicator_name = summary.get("matched_indicator", "")
            if indicator_name:
                result_str = f"{indicator_name} → {val}"
            else:
                result_str = f"→ {val}"
        elif summary.get("available") is True:
            result_str = "✓"
        elif summary.get("candidates"):
            cands = summary["candidates"]
            result_str = f"候选: {', '.join(c['name'] for c in cands)}"
        elif name == "list_indicators":
            result_str = "已列出"

        if result_str:
            parts.append(f"{name}({args_str}) {result_str}")
        else:
            parts.append(f"{name}({args_str})")
    return " ｜ ".join(parts)


def render_rag_page(data: Dict[str, Any]) -> None:
    """RAG 数据问答页 —— 多轮聊天界面。"""
    render_section("Data Q&A", "AI 数据问答", "基于项目数据和模型产出的智能问答")
    st.caption(f"⚙ build {_build_stamp()} · chat v2")

    # Warning if no API key
    if not os.environ.get("DEEPSEEK_API_KEY"):
        st.warning("⚠️ 未设置 DEEPSEEK_API_KEY，回答为 mock 降级模式。设置环境变量后重启以启用真实 AI 问答。")

    rag = get_rag_service()

    # ---- 运行时自检：确认 RAGService 是新版（ask() 接受 history 参数）----
    _sig = inspect.signature(rag.ask)
    if "history" not in _sig.parameters:
        from sc_macro_agent import rag_service as _rs_mod
        params_list = ", ".join(str(p) for p in _sig.parameters.values())
        rag_ver = getattr(_rs_mod, "RAG_SERVICE_VERSION", "未定义")
        rag_path = getattr(_rs_mod, "__file__", "未知")
        st.error(
            f"⚠ 检测到旧版 RAGService 实例（ask() 不接受 history 参数）。\n\n"
            f"诊断信息：\n"
            f"- 当前签名参数：({params_list})\n"
            f"- RAG_SERVICE_VERSION：{rag_ver}\n"
            f"- 模块路径：{rag_path}\n\n"
            f"这通常是 Streamlit 进程未重启或 cache_resource 未清理导致的。\n"
            f"请终止进程后执行：streamlit cache clear && streamlit run app.py"
        )
        st.stop()

    # ---- 部署环境自检 ----
    with st.expander("部署环境", expanded=False):
        from sc_macro_agent import rag_service as _rs_mod
        st.caption(f"APP_VERSION: {APP_VERSION}")
        st.caption(f"RAG_SERVICE_VERSION: {getattr(_rs_mod, 'RAG_SERVICE_VERSION', '?')}")
        st.caption(f"DEEPSEEK_API_KEY: {'已设置' if os.environ.get('DEEPSEEK_API_KEY') else '未设置'}")
        st.caption(f"语料文档: {len(rag.documents)} 篇 (card={len(rag.idx_card)}, doc={len(rag.idx_doc)})")

        # 语料文档命中来源三态：docs/（源码）→ artifacts/final/（运行产物）→ 内置常量兜底
        _src_map = {
            "docs": "docs/（源码）",
            "artifacts": "artifacts/final/（运行产物）",
            "builtin": "内置占位（源文件缺失）",
            "skip": "未注入",
        }
        _doc_sources = getattr(rag, "doc_sources", {})
        for _doc in ("known_limitations.md", "data_lineage.md", "methodology.md"):
            _src = _doc_sources.get(_doc, "?")
            st.caption(f"{_doc} 命中来源: {_src_map.get(_src, _src)}")

    # ---- 会话状态 ----
    if "rag_messages" not in st.session_state:
        st.session_state["rag_messages"] = []

    # ---- 工具栏 ----
    c_tool1, c_tool2 = st.columns([1, 5])
    with c_tool1:
        if st.button("🗑 清空对话", use_container_width=True):
            st.session_state["rag_messages"] = []
            st.rerun()
    with c_tool2:
        show_details = st.toggle("显示技术细节", value=False)

    # ---- 渲染全部历史气泡 ----
    for msg in st.session_state["rag_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and msg.get("meta"):
                meta = msg["meta"]
                route = meta.get("route", "")
                elapsed = meta.get("elapsed_s", 0)
                usage = meta.get("usage", {})
                tool_calls = meta.get("tool_calls", [])
                sources = meta.get("sources", [])
                rw_keywords = meta.get("rewrite_keywords")

                # 技术细节行（toggle 控制）
                if show_details:
                    # 路由徽标
                    if route == "tool" and tool_calls:
                        st.caption(f"🔧 {_fmt_tool_badge(tool_calls)}")
                    elif route == "rag":
                        kw_str = rw_keywords if rw_keywords else "（原始提问）"
                        st.caption(f"🔍 检索词：{kw_str} ｜ {len(sources)} 条来源")
                    elif route == "rag_no_hit":
                        st.caption("🔍 未检索到匹配数据，以下为系统能力说明")

                    # 耗时 + token + 费用
                    tt = usage.get("total_tokens", 0)
                    cost = usage.get("est_cost_cny", 0)
                    st.caption(
                        f"耗时 {elapsed:.1f}s ｜ "
                        f"{tt} tokens ｜ "
                        f"¥{cost:.4f}"
                    )

                # 参考来源（始终可展开）
                if sources:
                    with st.expander("参考来源"):
                        for i, src in enumerate(sources[:5]):
                            s_text = src.get("text", "") or "(无正文)"
                            s_score = src.get("score")
                            try:
                                s_score_str = f"{float(s_score):.3f}" if s_score is not None else "-"
                            except (TypeError, ValueError):
                                s_score_str = "-"
                            s_pool = src.get("metadata", {}).get("pool", "?")
                            st.caption(
                                f"[{i+1}] pool={s_pool} score={s_score_str} ｜ "
                                f"{s_text[:200]}"
                            )

    # ---- 预设问题（仅在无历史时显示）----
    if not st.session_state["rag_messages"]:
        st.markdown("**试试问：**")
        presets = [
            "2024年三季度四川GDP增速是多少",
            "全国PMI最近怎么样",
            "哪个模型RMSE最低",
            "你能回答什么",
        ]
        cols = st.columns(4)
        q_clicked = None
        for i, preset in enumerate(presets):
            if cols[i].button(preset, key=f"preset_{i}", use_container_width=True):
                q_clicked = preset

        # 别名行（自动生成，不手写）
        from sc_macro_agent.rag_service import get_indicator_aliases
        aliases = get_indicator_aliases()
        alias_pairs = [f"{a}={c}" for a, c in list(aliases.items())[:6]]
        alias_line = " / ".join(alias_pairs)
        st.caption(f"支持简称：{alias_line}；也支持追问，如「那2023年呢」")

        if q_clicked:
            _submit_question(rag, q_clicked)

    # ---- 聊天输入 ----
    if user_q := st.chat_input("输入问题..."):
        _submit_question(rag, user_q)

    # ---- 累计用量 ----
    msgs = st.session_state["rag_messages"]
    total_tokens = 0
    total_cost = 0.0
    n_user_msgs = 0
    for msg in msgs:
        if msg["role"] == "user" and msg.get("meta") is None:
            n_user_msgs += 1
        if msg["role"] == "assistant" and msg.get("meta"):
            u = msg["meta"].get("usage", {})
            total_tokens += u.get("total_tokens", 0)
            total_cost += u.get("est_cost_cny", 0)
    if n_user_msgs > 0:
        st.caption(
            f"本次会话：{n_user_msgs} 轮 ｜ "
            f"{total_tokens} tokens ｜ "
            f"¥{total_cost:.4f}"
        )


# ================================================================
# LLM Trace 页面
# ================================================================

# 价格常量（与 llm/client.py 一致，2026-07核对）
_TRACE_PROMPT_PRICE_PER_1K = 0.001       # CNY / 1K input tokens (cache miss)
_TRACE_COMPLETION_PRICE_PER_1K = 0.002   # CNY / 1K output tokens
_TRACE_CACHE_HIT_RATIO = 1.0 / 50         # 缓存命中输入约为未命中的 1/50


def _trace_cost(prompt_tokens: int, completion_tokens: int, cache_hit_tokens: int) -> float:
    """单条 trace 成本估算。cached_tokens 是 prompt_tokens 的子集。"""
    miss = prompt_tokens - cache_hit_tokens
    return (
        miss * _TRACE_PROMPT_PRICE_PER_1K / 1000
        + cache_hit_tokens * _TRACE_PROMPT_PRICE_PER_1K * _TRACE_CACHE_HIT_RATIO / 1000
        + completion_tokens * _TRACE_COMPLETION_PRICE_PER_1K / 1000
    )


def _fmt_latency(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    return f"{ms:.0f}ms"


def render_llm_traces(_data: Dict[str, Any]) -> None:
    from sc_macro_agent.llm.client import LLMClient

    client = LLMClient.get_instance()
    traces_dir = client.traces_dir

    render_section("LLM Traces", "LLM 调用追踪", "每次 API 调用的完整记录：系统提示词、用户输入、响应内容、Token 消耗")

    # --- 状态条 ---
    if client.is_mock:
        st.markdown(
            f"""<div style="padding:0.6rem 1rem; background:{COLORS['bg_secondary']};
            border:1px solid {COLORS['accent_amber']}; border-radius:8px;
            color:{COLORS['accent_amber']}; font-weight:600;">
            ⚠️ LLM 处于 MOCK 降级模式 —— 所有响应均为占位文本，非真实模型输出
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div style="padding:0.6rem 1rem; background:{COLORS['bg_secondary']};
            border:1px solid {COLORS['accent_green']}; border-radius:8px;
            color:{COLORS['accent_green']}; font-weight:600;">
            ✅ LLM 在线 — deepseek-v4-flash
            </div>""",
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # --- 日期选择 ---
    date_files: list[str] = []
    if traces_dir.exists():
        for fp in sorted(traces_dir.glob("*.jsonl"), reverse=True):
            date_files.append(fp.stem)  # YYYY-MM-DD
    else:
        date_files = []

    if not date_files:
        st.info("暂无 LLM 调用记录。运行一次 Agent 流水线后数据将在此显示。")
        return

    selected_date = st.selectbox("选择日期", date_files, key="trace_date")
    trace_path = traces_dir / f"{selected_date}.jsonl"
    if not trace_path.exists():
        st.info("该日期暂无调用记录。")
        return

    # 加载全部记录
    raw_lines: list[str] = []
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_lines.append(line)
    if not raw_lines:
        st.info("该日期暂无调用记录。")
        return

    import json
    traces: list[dict] = []
    for line in raw_lines:
        try:
            traces.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    if not traces:
        st.info("该日期暂无有效调用记录。")
        return

    # --- 聚合统计 ---
    total_calls = len(traces)
    total_prompt = sum(t.get("prompt_tokens", 0) or 0 for t in traces)
    total_completion = sum(t.get("completion_tokens", 0) or 0 for t in traces)
    total_cached = sum(t.get("cache_hit_tokens", 0) or 0 for t in traces)
    total_tokens = total_prompt + total_completion
    total_cost = sum(
        _trace_cost(
            t.get("prompt_tokens", 0) or 0,
            t.get("completion_tokens", 0) or 0,
            t.get("cache_hit_tokens", 0) or 0,
        )
        for t in traces
    )
    # 平均延迟只统计成功的真实调用
    real_traces = [t for t in traces if not t.get("is_mock") and not t.get("error")]
    if real_traces:
        avg_latency_ms = sum(t.get("latency_ms", 0) or 0 for t in real_traces) / len(real_traces)
    else:
        avg_latency_ms = 0.0

    cache_hit_rate = (total_cached / total_prompt * 100) if total_prompt > 0 else 0.0

    # --- KPI 四卡片 ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card accent-left">
                <div class="label">总调用数</div>
                <div class="value">{total_calls}</div>
                <div class="meta">选中日期: {selected_date}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-card accent-purple">
                <div class="label">总 Token</div>
                <div class="value">{total_tokens:,}</div>
                <div class="meta">prompt {total_prompt:,} + completion {total_completion:,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card accent-green">
                <div class="label">估算成本</div>
                <div class="value">¥{total_cost:.6f}</div>
                <div class="meta">缓存命中率 {cache_hit_rate:.1f}%（分母: prompt_tokens）</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-card accent-left">
                <div class="label">平均延迟</div>
                <div class="value">{_fmt_latency(avg_latency_ms)}</div>
                <div class="meta">仅统计 {len(real_traces)}/{total_calls} 次成功真实调用</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # --- 按 caller 堆叠柱状图 ---
    from collections import defaultdict
    caller_tokens: dict[str, dict[str, int]] = defaultdict(lambda: {"prompt": 0, "completion": 0})
    for t in traces:
        c = t.get("caller", "unknown") or "unknown"
        caller_tokens[c]["prompt"] += t.get("prompt_tokens", 0) or 0
        caller_tokens[c]["completion"] += t.get("completion_tokens", 0) or 0

    if caller_tokens:
        callers = sorted(caller_tokens.keys())
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=callers,
            y=[caller_tokens[c]["prompt"] for c in callers],
            name="prompt",
            marker_color=COLORS["accent_cyan"],
            hovertemplate="prompt: %{y:,}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            x=callers,
            y=[caller_tokens[c]["completion"] for c in callers],
            name="completion",
            marker_color=COLORS["accent_purple"],
            hovertemplate="completion: %{y:,}<extra></extra>",
        ))
        fig.update_layout(
            barmode="stack",
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="",
            yaxis_title="Token",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        )
        fig = apply_base_style(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("无调用数据可图表化。")

    # --- 提示词版本表 ---
    st.markdown(f'<div style="font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em; color:{COLORS["text_muted"]}; margin-top:1.5rem; margin-bottom:0.5rem; font-weight:600;">提示词版本</div>', unsafe_allow_html=True)

    from collections import Counter
    pv_groups: dict[tuple, list[dict]] = defaultdict(list)
    for t in traces:
        key = (t.get("prompt_id", "?") or "?", t.get("prompt_version", "?") or "?")
        pv_groups[key].append(t)

    version_rows = []
    for (pid, ver), items in pv_groups.items():
        calls = len(items)
        total_tok = sum((it.get("prompt_tokens", 0) or 0) + (it.get("completion_tokens", 0) or 0) for it in items)
        err_count = sum(1 for it in items if it.get("error"))
        mock_count = sum(1 for it in items if it.get("is_mock"))
        real_items = [it for it in items if not it.get("is_mock") and not it.get("error")]
        avg_ms = sum(it.get("latency_ms", 0) or 0 for it in real_items) / max(len(real_items), 1)
        version_rows.append({
            "prompt_id": pid,
            "version": ver,
            "调用次数": calls,
            "平均 token": round(total_tok / max(calls, 1)),
            "平均延迟": _fmt_latency(avg_ms),
            "mock 数": mock_count,
            "错误数": err_count,
        })

    if version_rows:
        version_rows.sort(key=lambda r: r["调用次数"], reverse=True)
        st.dataframe(pd.DataFrame(version_rows), use_container_width=True, hide_index=True)
    else:
        st.info("无版本数据。")

    # --- 明细表格 ---
    st.markdown(f'<div style="font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em; color:{COLORS["text_muted"]}; margin-top:1.5rem; margin-bottom:0.5rem; font-weight:600;">调用明细（共 {total_calls} 条）</div>', unsafe_allow_html=True)

    detail_rows = []
    for t in traces:
        ts = t.get("timestamp", "") or ""
        if "T" in str(ts):
            time_str = str(ts).split("T")[1][:8]
        else:
            time_str = str(ts)[:8]
        pt = t.get("prompt_tokens", 0) or 0
        ct = t.get("completion_tokens", 0) or 0
        is_err = bool(t.get("error"))
        is_mk = bool(t.get("is_mock"))
        if is_err:
            status = "❌"
        elif is_mk:
            status = "⚠️"
        else:
            status = "✅"
        detail_rows.append({
            "时间": time_str,
            "caller": t.get("caller", "?") or "?",
            "prompt@ver": f"{t.get('prompt_id','?') or '?'}@{t.get('prompt_version','?') or '?'}",
            "tokens in/out": f"{pt:,}/{ct:,}",
            "延迟": _fmt_latency(t.get("latency_ms", 0) or 0),
            "状态": status,
        })

    detail_df = pd.DataFrame(detail_rows)
    row_labels = [f"#{i}" for i in range(len(detail_rows))]
    st.dataframe(detail_df, use_container_width=True, hide_index=True, height=min(len(detail_rows) * 36 + 38, 400))

    # --- 行详情（可展开） ---
    st.markdown(f'<div style="font-size:0.85rem; text-transform:uppercase; letter-spacing:0.08em; color:{COLORS["text_muted"]}; margin-top:1rem; margin-bottom:0.5rem; font-weight:600;">查看详情</div>', unsafe_allow_html=True)
    selected_label = st.selectbox("选择行", row_labels, key="trace_detail_row")
    selected_idx = row_labels.index(selected_label)
    t = traces[selected_idx]

    with st.expander("System Prompt", expanded=False):
        st.text_area("system", value=t.get("system", ""), height=200, disabled=True, key="detail_system", label_visibility="collapsed")
    with st.expander("User Prompt", expanded=False):
        st.text_area("user", value=t.get("user", ""), height=200, disabled=True, key="detail_user", label_visibility="collapsed")
    with st.expander("Response", expanded=False):
        st.text_area("response", value=t.get("response", ""), height=200, disabled=True, key="detail_resp", label_visibility="collapsed")


# ================================================================
# main()
# ================================================================
def main() -> None:
    _tick("main:enter")
    # trace 落盘路径必须在 main() 中设置，不能放在 @st.cache_resource 内部：
    # 缓存命中时函数体完全跳过，set_artifact_dir 不会执行
    from sc_macro_agent.config import AppConfig
    from sc_macro_agent.llm.client import LLMClient
    LLMClient.set_artifact_dir(AppConfig.from_env().data.resolve_artifact_dir(create=False))
    # 在 main() 渲染 status，不放在 @st.cache_resource 内部：
    # 冷缓存时展示初始化进度并 显式 complete；热缓存时瞬间完成，人眼不可感知
    with st.status("正在初始化引擎…", expanded=False) as _init_status:
        data = load_view_data(1)
        _init_err = getattr(data["engine"], "_init_error", None)
        if _init_err:
            _init_status.update(label="引擎初始化失败", state="error")
        else:
            _init_status.update(label="引擎初始化完成", state="complete")
    _tick("main:after_load_view_data")
    # 引擎初始化失败时在页面顶部显式提示（不要只 print 到 stderr）
    if _init_err:
        st.warning(f"流水线初始化失败，部分功能不可用：{_init_err}")
    # 顶部导航兜底（移动端侧边栏图标可能被遮挡）
    st.session_state.setdefault("current_page", PAGE_NAMES[0])
    _prime_nav_widget("_nav_top")
    with st.expander("📑 页面导航", expanded=False):
        st.selectbox(
            "跳转到", PAGE_NAMES,
            key="_nav_top",
            on_change=_sync_page_from,
            args=("_nav_top",),
            label_visibility="collapsed",
        )
    st.caption("📑 使用上方导航切换页面，或点击左上角图标展开侧边栏")
    page, refresh = sidebar_controls(data)

    if refresh:
        load_engine.clear()
        load_view_data.clear()
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    render_hero(data)
    _tick("main:after_render_hero")

    # 路由：优先从 session_state 读取（覆盖 sidebar_controls 的返回值以保持同步）
    page = st.session_state.current_page
    if "概览驾驶舱" in page:
        render_overview(data)
    elif "现时预测" in page:
        render_nowcast(data)
    elif "历史回测" in page:
        render_backtest(data)
    elif "因子分析" in page:
        render_factors(data)
    elif "数据质量" in page:
        render_data_quality(data)
    elif "Agent" in page:
        render_agent_v2(data)
    elif "AI 简报" in page:
        render_briefing_page(data)
    elif "数据问答" in page:
        render_rag_page(data)
    elif "LLM 追踪" in page:
        render_llm_traces(data)
    else:
        render_agent_v2(data)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        st.error(f"应用启动失败: {exc}")
        st.code(traceback.format_exc())

# streamlit run "D:\PythonProject\macro_nowcasting\sc_macro_agent_project\app.py" --server.port 8501