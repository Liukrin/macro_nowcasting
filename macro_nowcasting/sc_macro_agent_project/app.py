from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sc_macro_agent import AppConfig, PredictionEngine

st.set_page_config(
    page_title="四川省 GDP 混频预测系统",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
    config = AppConfig.from_env()
    engine = PredictionEngine(config=config)
    engine.run_agent(goal="audit_build_train_backtest_report", save_artifacts=False)
    return engine


@st.cache_data(show_spinner=False)
def load_view_data(_: int) -> Dict[str, Any]:
    engine = load_engine()
    status = engine.get_status()
    summary = engine.summarize()
    prediction = summary.get("prediction") or engine.predict_next()
    audit_result = engine.audit_result or {}
    backtest = engine.backtest_result or {}
    factor_summary = engine.get_factor_summary() if hasattr(engine, "get_factor_summary") else {}
    snapshot = engine.data_manager.get_latest_snapshot()
    availability = engine.data_manager.get_data_availability()
    signal_overview = engine.data_manager.build_training_signal_overview()
    leaderboard_df = safe_df(summary.get("leaderboard", []))
    top_features_df = safe_df(summary.get("top_features", []))
    agent_steps_df = safe_df(summary.get("agent", {}).get("steps", []))
    window_df = safe_df(backtest.get("window_results", []))
    checks_df = safe_df(audit_result.get("checks", []))
    items_df = safe_df(availability.get("items", []))
    summary_df = safe_df(audit_result.get("summary", []))

    registry = engine.feature_artifacts.feature_registry if engine.feature_artifacts is not None else None
    family_df = pd.DataFrame(columns=["family", "count"])
    region_df = pd.DataFrame(columns=["region", "count"])
    if registry is not None:
        family_df = pd.DataFrame(list(registry.summary_by_family().items()), columns=["family", "count"])
        region_df = pd.DataFrame(list(registry.summary_by_region().items()), columns=["region", "count"])

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
        ["🏠 概览驾驶舱", "🔮 现时预测", "📈 历史回测", "🔍 因子分析", "🧪 数据质量", "⚙️ Agent 工作流"],
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
        st.markdown(
            f"""
            <div class="kpi-card accent-green">
                <div class="label">回测性能</div>
                <div class="value" style="font-size: 1.8rem;">RMSE {fmt_number(rmse, 3) if rmse else '-'}</div>
                <div class="meta">R² = {fmt_number(r2, 3) if r2 else '-'} | MAPE {fmt_pct_decimal(metrics.get('mape'), 2)}</div>
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
        render_section("Model Selection", "候选模型对比", "基于回测表现的模型自动选择 leaderboard")
        fig = create_leaderboard_chart(leaderboard_df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        if not leaderboard_df.empty:
            with st.expander("查看详细指标"):
                st.dataframe(leaderboard_df, use_container_width=True, hide_index=True)

    with b2:
        render_section("Prediction Summary", "预测摘要", "目标指标、基准值与最终预测")
        pred_data = {
            "目标指标": prediction.get("target_indicator"),
            "预测季度": prediction.get("prediction_quarter"),
            "预测值": prediction.get("prediction_value"),
            "基准值": prediction.get("benchmark_value"),
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
                    f'<div style="padding: 0.5rem 0; color: {COLORS["text_secondary"]}; font-size: 0.9rem; border-bottom: 1px solid {COLORS["border"]};">• {note}</div>',
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
                    f'<div style="margin-top: 0.5rem; padding: 0.75rem; background: {COLORS["bg_tertiary"]}; border-radius: 8px; font-size: 0.9rem; color: {COLORS["text_secondary"]}; border-left: 3px solid {COLORS["accent_cyan"]};">{note}</div>',
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
    factor_summary = data["factor_summary"]
    top_features_df = data["top_features_df"]
    family_df = data["family_df"]
    region_df = data["region_df"]

    render_section("Factor Analysis", "因子分析与解释性", "动态因子模型（DFM）与特征重要性")

    cols = st.columns(3)
    cols[0].metric("提取因子数", factor_summary.get("n_factors", 0))
    cols[1].metric("累计解释方差", fmt_pct_decimal(factor_summary.get("cumulative_variance"), 1))
    cols[2].metric("模型拟合状态", "已拟合" if factor_summary.get("is_fitted") else "未拟合")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")
    with left:
        fig = create_factor_variance_chart(factor_summary)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("当前没有可视化的因子解释方差。")

    with right:
        top_loadings = factor_summary.get("top_loadings") or []
        if top_loadings:
            loading_rows = []
            for block in top_loadings:
                factor_name = block.get("factor")
                for item in block.get("top_loadings", [])[:5]:
                    loading_rows.append({
                        "因子": factor_name,
                        "特征": item.get("feature"),
                        "载荷": item.get("loading"),
                        "绝对载荷": item.get("abs_loading"),
                    })
            st.dataframe(pd.DataFrame(loading_rows), use_container_width=True, hide_index=True, height=300)
        else:
            st.info("当前没有 top loadings 可展示。")

    st.markdown(
        "<div style='height: 1px; background: linear-gradient(90deg, transparent, #2a2e37, transparent); margin: 2rem 0;'></div>",
        unsafe_allow_html=True)

    score_df = feature_score_df(top_features_df)
    f1, f2 = st.columns([1.2, 0.8], gap="large")

    with f1:
        render_section("Features", "特征重要性排序", "基于模型系数与树形结构的综合评分")
        fig = create_feature_chart(score_df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        elif not top_features_df.empty:
            st.dataframe(top_features_df, use_container_width=True, hide_index=True)
        else:
            st.info("当前没有特征重要性结果。")

    with f2:
        render_section("Registry", "特征注册表分布", "按类别与地区划分的特征库统计")
        inner1, inner2 = st.columns(2)
        fig1 = create_registry_pie(family_df, "family")
        fig2 = create_registry_pie(region_df, "region")
        if fig1 is not None:
            inner1.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
        if fig2 is not None:
            inner2.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        if family_df.empty and region_df.empty:
            st.info("当前没有 feature registry 统计信息。")


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


def main() -> None:
    data = load_view_data(1)
    page, refresh = sidebar_controls(data)

    if refresh:
        load_engine.clear()
        load_view_data.clear()
        st.cache_data.clear()
        st.rerun()

    render_hero(data)

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
    else:
        render_agent(data)


if __name__ == "__main__":
    main()

# streamlit run "D:\PythonProject\macro_nowcasting\sc_macro_agent_project\app.py" --server.port 8501