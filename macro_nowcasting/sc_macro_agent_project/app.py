from __future__ import annotations

import sys
import traceback

# --- Phase 1: bare-minimum imports for error display ---
try:
    import streamlit as st
except ImportError as e:
    # streamlit itself is missing — nothing we can do; let it crash with a clear message
    raise RuntimeError(f"streamlit 未安装，请检查 requirements.txt: {e}") from e

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
    header {{visibility: hidden;}}
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
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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


def safe_df(data: Any) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, list):
        return pd.DataFrame(data)
    if isinstance(data, dict):
        return pd.DataFrame([data])
    return pd.DataFrame()


def mapping_df(mapping: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([{"字段": k, "值": v} for k, v in mapping.items()]) if mapping else pd.DataFrame(
        columns=["字段", "值"])


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
    import os, sys
    _tick("load_engine:enter")
    if _TIMING_ON:
        print(f"[ENV] SC_MACRO_LIGHT_MODE={os.environ.get('SC_MACRO_LIGHT_MODE')}", file=sys.stderr, flush=True)
    config = AppConfig.from_env()
    engine = PredictionEngine(config=config)
    _tick("load_engine:engine_constructed")
    from sc_macro_agent.llm.client import LLMClient
    LLMClient.set_artifact_dir(config.data.resolve_artifact_dir(create=False))
    # 默认跑完整流水线：审计 → 构建特征 → 训练 → 回测 → 预测；
    # SC_MACRO_LIGHT_MODE=true 时退回轻量（仅审计+构建特征）。
    # （原 SC_MACRO_FULL_PIPELINE 变量已废弃）
    light_mode = os.environ.get("SC_MACRO_LIGHT_MODE", "").lower() == "true"
    _tick("load_engine:before_run_agent")
    try:
        engine.initialize()
        if light_mode:
            with st.status("轻量模式：仅审计数据与构建特征…", expanded=False) as _status:
                engine.audit_data(save_artifacts=False)
                engine.build_features()
                _status.update(label="轻量模式初始化完成（未训练/未回测）", state="complete")
        else:
            with st.status("正在初始化引擎（审计数据 → 构建特征 → 训练模型 → 运行回测）…", expanded=False) as _status:
                engine.audit_data(save_artifacts=False)
                _status.update(label="正在构建特征…")
                engine.build_features()
                _status.update(label="正在训练模型…")
                engine.train()
                _status.update(label="正在运行回测…")
                try:
                    engine.backtest()
                except Exception as bt_exc:
                    engine.warnings.append(str(bt_exc))
                    engine.agent.record_warning(str(bt_exc))
                _status.update(label="正在生成预测…")
                engine.predict_next()
                _status.update(label="引擎初始化完成", state="complete")
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
@st.cache_resource(show_spinner="正在加载模型与数据，首次约需 10 秒…")
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
        height=320,
        margin=dict(l=10, r=40, t=10, b=10),
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
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
        xaxis_title="<b>季度</b>",
        yaxis_title="<b>GDP 数值</b>",
        hovermode="x unified",
    )
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
    fig.update_layout(
        height=320,
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

    page = st.sidebar.radio(
        "导航",
        ["🏠 概览驾驶舱", "🔮 现时预测", "📈 历史回测", "🔍 因子分析", "🧪 数据质量",
         "⚙️ Agent 工作流", "🤖 AI 简报", "💬 数据问答", "📊 LLM 追踪"],
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

    cols = st.sidebar.columns(2)
    cols[0].metric("模型", summary.get("selected_model") or "-", label_visibility="collapsed")
    cols[1].metric("模式", status.get("dataset_mode") or "-", label_visibility="collapsed")

    cols = st.sidebar.columns(2)
    cols[0].metric("样本", status.get("n_rows") or 0, label_visibility="collapsed")
    cols[1].metric("特征", status.get("n_features") or 0, label_visibility="collapsed")

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
        st.sidebar.caption(f"Chronos: {cs} | 修正: {prediction.get('chronos_correction',0):.2f}")

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

    c1, c2, c3 = st.columns(3)
    ci = prediction.get("confidence_interval", {}) or {}

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
                <div class="meta">训练样本：{summary.get('n_rows') or 0} 行<br>特征数量：{summary.get('n_features') or 0} 维</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        rmse = metrics.get('rmse')
        r2 = metrics.get('r2')
        dir_acc = metrics.get('direction_accuracy')
        dir_acc_text = f"{dir_acc:.1%}" if dir_acc is not None else '-'
        st.markdown(
            f"""
            <div class="kpi-card accent-green">
                <div class="label">回测性能（32窗口·含疫情期）</div>
                <div class="value" style="font-size: 1.8rem;">RMSE {fmt_number(rmse, 3) if rmse else '-'}</div>
                <div class="meta">R² = {fmt_number(r2, 3) if r2 else '-'} | 方向准确率 {dir_acc_text}</div>
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
        render_section("Model Selection", "候选模型对比", "基于验证集（近12季·差分口径）的模型自动选择")
        fig = create_leaderboard_chart(leaderboard_df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        if not leaderboard_df.empty:
            st.caption("此处 RMSE 为验证集差分口径，与上方 32 窗口回测（level 口径，含 2020 年断点）不可直接比较。")
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

    # Chronos 残差修正状态（默认开启，不改变其行为，仅做状态展示）
    chronos_state = prediction.get("chronos_state", "unknown")
    try:
        chronos_correction = float(prediction.get("chronos_correction", 0.0))
    except (TypeError, ValueError):
        chronos_correction = 0.0
    if chronos_state == "ready":
        chronos_text = f"Chronos 残差修正：已启用，修正量 {chronos_correction:+.3f} 个百分点"
        chronos_color = COLORS["accent_cyan"]
    elif chronos_state == "failed":
        chronos_text = "Chronos 残差修正：模型加载失败，已跳过（不影响主预测）"
        chronos_color = COLORS["accent_amber"]
    else:
        chronos_text = "Chronos 残差修正：未加载"
        chronos_color = COLORS["text_muted"]
    st.markdown(
        f'<div style="padding:0.5rem 1rem; background:{COLORS["bg_tertiary"]}; border-radius:8px; font-size:0.85rem; color:{chronos_color};">{chronos_text}</div>',
        unsafe_allow_html=True)

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

    render_section("Backtest", "历史回测分析", "滚动窗口交叉验证与误差分布")

    cols = st.columns(4)
    metric_items = [
        ("MAE", fmt_number(metrics.get("mae"), 3), COLORS["text_primary"]),
        ("RMSE", fmt_number(metrics.get("rmse"), 3), COLORS["accent_cyan"]),
        ("MAPE", fmt_pct_decimal(metrics.get("mape"), 2), COLORS["accent_amber"]),
        ("R²", fmt_number(metrics.get("r2"), 3), COLORS["accent_green"]),
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

    tab1, tab2, tab3 = st.tabs(["📋 可用性明细", "🔍 质量检查", "📊 审计摘要"])
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
    """Agent 页：接入新 orchestrator 四角色流水线。"""
    render_section("Agent Workflow", "AI Agent 协作流水线", "Data → Model → Analyst → Critic 四角色协作")

    if st.button("▶ 启动 Agent 流水线", use_container_width=True):
        with st.status("Agent 流水线运行中...", expanded=True) as status_box:
            engine = load_engine()
            from sc_macro_agent.agents import AgentOrchestrator
            orch = AgentOrchestrator(config=engine.config)
            result = orch.run(engine)

            for s in result.get("steps", []):
                name = s.get("name", "?")
                elapsed = s.get("elapsed_s", 0)
                if "data_agent" in name:
                    st.write(f"✅ 数据审计完成 ({elapsed:.1f}s)")
                elif "model_agent" in name:
                    mr = s.get("result", {})
                    st.write(f"✅ 模型预测完成 ({elapsed:.1f}s) — {mr.get('prediction_quarter','?')}: {fmt_number(mr.get('prediction_value'), 2)}%")
                elif "analyst_agent" in name:
                    st.write(f"✅ 简报生成完成 ({elapsed:.1f}s)")
                elif "critic" in name:
                    rev = s.get("review", {})
                    passed = rev.get("passed", False)
                    icon = "✅" if passed else "⚠️"
                    st.write(f"{icon} 审阅完成 ({elapsed:.1f}s) — {'通过' if passed else '存在 ' + str(len(rev.get('issues',[]))) + ' 个问题'}")
                else:
                    st.write(f"⏳ {name} ({elapsed:.1f}s)")

            status_box.update(label=f"流水线完成 — 状态: {result.get('status','?')}", state="complete")

        final_status = result.get("status", "?")
        if final_status in ("passed_review", "failed_review", "unreviewed"):
            st.markdown("### 简报")
            st.markdown(result.get("briefing", "无内容"))

            review = result.get("review", {})
            st.markdown("### 审阅结论")
            summary = review.get("summary", "无")
            if isinstance(summary, dict):
                st.json(summary)
            else:
                st.write(summary)
            if review.get("issues"):
                issues_df = pd.DataFrame(review["issues"])
                st.dataframe(issues_df, use_container_width=True, hide_index=True)

            usage = result.get("token_usage", {})
            st.caption(f"Token: {usage.get('total_tokens',0)} | 费用: ¥{usage.get('est_cost_cny',0):.4f} | 重写: {result.get('rewrite_rounds',0)}轮")

            # Download button
            st.download_button("⬇ 下载简报 (.md)", result.get("briefing", ""),
                               file_name=f"briefing_{pd.Timestamp.now().strftime('%Y%m%d')}.md")
        else:
            st.error(f"流水线状态异常: {final_status}")

    # Show historical briefings
    st.markdown("---")
    st.markdown("### 历史简报")
    briefings_dir = load_engine().config.data.resolve_artifact_dir(create=False) / "briefings"
    if briefings_dir.exists():
        files = sorted(briefings_dir.glob("briefing_*.md"), reverse=True)
        if files:
            selected = st.selectbox("选择历史简报", [f.name for f in files])
            if selected:
                content = (briefings_dir / selected).read_text(encoding="utf-8")
                st.markdown(content)
        else:
            st.info("暂无历史简报")
    else:
        st.info("暂无历史简报")


def render_briefing_page(data: Dict[str, Any]) -> None:
    """AI 简报页：简化版一键生成。"""
    render_section("AI Briefing", "AI 经济简报生成", "基于最新数据自动生成四段式经济简报，经 AI 审阅后输出")

    st.markdown(f'<div style="padding:0.5rem 1rem;background:{COLORS["bg_tertiary"]};border-radius:8px;font-size:0.85rem;color:{COLORS["text_secondary"]};">'
                f'当前数据截至: {data.get("status",{}).get("dataset_mode","?")} 模式</div>',
                unsafe_allow_html=True)
    st.markdown("")

    if st.button("🤖 生成简报", use_container_width=True, type="primary"):
        with st.status("正在生成...", expanded=True) as s:
            engine = load_engine()
            from sc_macro_agent.agents import AgentOrchestrator
            orch = AgentOrchestrator(config=engine.config)
            result = orch.run(engine)

            for step in result.get("steps", []):
                nm = step.get("name","?")
                el = step.get("elapsed_s",0)
                if "data" in nm: st.write(f"✅ 数据审计 ({el:.1f}s)")
                elif "model" in nm: st.write(f"✅ 模型预测 ({el:.1f}s)")
                elif "analyst" in nm: st.write(f"✅ 简报撰写 ({el:.1f}s)")
                elif "critic" in nm:
                    rv = step.get("review",{})
                    st.write(f"{'✅' if rv.get('passed') else '⚠️'} 审阅 ({el:.1f}s)")
            s.update(label="生成完成", state="complete")

        st.markdown(result.get("briefing", ""))
        rev = result.get("review", {})
        if rev.get("issues"):
            with st.expander(f"审阅发现 {len(rev['issues'])} 个问题"):
                st.dataframe(pd.DataFrame(rev["issues"]), use_container_width=True, hide_index=True)
        u = result.get("token_usage", {})
        st.caption(f"Token: {u.get('total_tokens',0)} | 费用: ¥{u.get('est_cost_cny',0):.4f}")


@st.cache_resource(show_spinner="正在构建检索索引…")
def get_rag_service():
    """RAG 服务模块级缓存：避免每次渲染重建 TF-IDF 语料。"""
    from sc_macro_agent.rag_service import RAGService
    engine = load_engine()
    return RAGService(config=engine.config, engine=engine)


def render_rag_page(data: Dict[str, Any]) -> None:
    """RAG 数据问答页。"""
    render_section("Data Q&A", "AI 数据问答", "基于项目数据和模型产出的智能问答")

    # Warning if no API key
    import os
    if not os.environ.get("DEEPSEEK_API_KEY"):
        st.warning("⚠️ 未设置 DEEPSEEK_API_KEY，回答为 mock 降级模式。设置环境变量后重启以启用真实 AI 问答。")

    rag = get_rag_service()

    # Preset questions
    st.markdown("**快捷提问：**")
    presets = [
        "2024年三季度四川GDP增速是多少",
        "这个模型比基准好多少",
        "数据有哪些已知局限",
        "2026年一季度的预测是多少",
    ]
    cols = st.columns(4)
    q = None
    for i, preset in enumerate(presets):
        if cols[i].button(preset, key=f"preset_{i}", use_container_width=True):
            q = preset

    # Chat input
    user_q = st.chat_input("输入问题...")
    if user_q:
        q = user_q

    if q:
        with st.spinner("检索中..."):
            result = rag.ask(q)
        st.markdown(f"**问：** {q}")
        st.markdown(f"**答：** {result['answer']}")
        with st.expander("参考来源"):
            for i, src in enumerate(result.get("sources", [])[:3]):
                st.caption(f"[{i+1}] 相似度={src['score']:.3f} | {src['text'][:200]}")


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
    data = load_view_data(1)
    _tick("main:after_load_view_data")
    # 引擎初始化失败时在页面顶部显式提示（不要只 print 到 stderr）
    _init_err = getattr(data["engine"], "_init_error", None)
    if _init_err:
        st.warning(f"流水线初始化失败，部分功能不可用：{_init_err}")
    page, refresh = sidebar_controls(data)

    if refresh:
        load_engine.clear()
        load_view_data.clear()
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    render_hero(data)
    _tick("main:after_render_hero")

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
    elif page == "💬 数据问答":
        render_rag_page(data)
    elif page == "📊 LLM 追踪":
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