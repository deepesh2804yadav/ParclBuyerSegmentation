#!/usr/bin/env python3
"""
app.py
══════════════════════════════════════════════════════════════════════════════
Parcl Buyer Intelligence Dashboard
Project : Machine Learning-Based Buyer Segmentation and Investment Profiling
Client  : Parcl Co. Limited
══════════════════════════════════════════════════════════════════════════════

Four-tab Streamlit analytics dashboard powered by Phase 2 ML outputs.
Consumes:
  • data/processed/segmented_buyers.csv
  • models/kmeans_model.joblib
  • models/scaler.joblib
  • models/encoder.joblib

Tab layout
──────────
  Tab 1  Buyer Segmentation Overview   — KPIs + cluster distribution
  Tab 2  Investor Behavior Dashboard   — behavioral cross-tabs per cluster
  Tab 3  Geographic Buyer Analysis     — country / region breakdowns
  Tab 4  Segment Insights Panel        — descriptive stats + cohort profiles

REQUIREMENTS
────────────
  pip install streamlit plotly pandas joblib

USAGE
─────
  streamlit run app.py        (run from project root)
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION  (must be the very first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Parcl | Buyer Intelligence",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# ── File paths ─────────────────────────────────────────────────────────────────
_ROOT        = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(_ROOT, "data", "processed", "segmented_buyers.csv")
KMEANS_PATH  = os.path.join(_ROOT, "models", "kmeans_model.joblib")
SCALER_PATH  = os.path.join(_ROOT, "models", "scaler.joblib")
ENCODER_PATH = os.path.join(_ROOT, "models", "encoder.joblib")

# ── Brand colour palette ───────────────────────────────────────────────────────
BRAND_NAVY   = "#0F172A"
BRAND_BLUE   = "#3B82F6"
BRAND_TEAL   = "#06B6D4"
CARD_BG      = "#FFFFFF"
PAGE_BG      = "#F1F5F9"
MUTED        = "#64748B"

# Segment / cluster colours — up to 8 clusters
CLUSTER_PALETTE: list[str] = [
    "#3B82F6",  # 0 blue
    "#8B5CF6",  # 1 violet
    "#EC4899",  # 2 pink
    "#10B981",  # 3 emerald
    "#F59E0B",  # 4 amber
    "#EF4444",  # 5 red
    "#06B6D4",  # 6 cyan
    "#F97316",  # 7 orange
]

# Cohort ground-truth colours (used in Tab 4 only)
COHORT_PALETTE: dict[str, str] = {
    "C1_Global_Investor":  "#3B82F6",
    "C2_First_Time_Buyer": "#10B981",
    "C3_Corporate_Buyer":  "#8B5CF6",
    "C4_Luxury_Investor":  "#F59E0B",
    "Noise":               "#94A3B8",
}

# Human-readable cohort labels
COHORT_LABELS: dict[str, str] = {
    "C1_Global_Investor":  "C1 · Global Investors",
    "C2_First_Time_Buyer": "C2 · First-Time Buyers",
    "C3_Corporate_Buyer":  "C3 · Corporate Buyers",
    "C4_Luxury_Investor":  "C4 · Luxury Investors",
    "Noise":               "Background Noise",
}

# Cohort descriptions displayed in Tab 4 insight cards
COHORT_DESCRIPTIONS: dict[str, dict] = {
    "C1_Global_Investor": {
        "icon": "🌐",
        "title": "Global Investors",
        "summary": "High-net-worth individuals and corporates investing across international markets.",
        "bullets": [
            "International focus: UAE, Singapore, UK, Germany",
            "Investment-dominant (92%)",
            "Low loan dependency — largely self-funded",
            "High satisfaction scores (avg ≈ 4.5 / 5)",
        ],
    },
    "C2_First_Time_Buyer": {
        "icon": "🏠",
        "title": "First-Time Buyers",
        "summary": "Young domestic buyers purchasing their first primary residence.",
        "bullets": [
            "Youngest cohort — average age 22–36",
            "Personal use only (95%)",
            "Highest mortgage dependency (88%)",
            "US-centric, digitally engaged",
        ],
    },
    "C3_Corporate_Buyer": {
        "icon": "🏢",
        "title": "Corporate Buyers",
        "summary": "Institutional entities transacting at scale in major business hubs.",
        "bullets": [
            "100% corporate client type",
            "Agent & corporate partnership channels",
            "Global business hubs: UAE, US, UK, SG, DE",
            "Investment + portfolio diversification focus",
        ],
    },
    "C4_Luxury_Investor": {
        "icon": "💎",
        "title": "Luxury Investors",
        "summary": "Ultra-high-net-worth buyers in premium global markets.",
        "bullets": [
            "Highest satisfaction — 80% rate score 5 / 5",
            "Premium markets: Monaco, Switzerland, UAE",
            "Investment-only strategy (95%)",
            "Near-zero loan dependency (5%)",
        ],
    },
}

# Shared Plotly layout defaults
_CHART_FONT = dict(family="Inter, Arial, sans-serif", color=BRAND_NAVY, size=12)
_CHART_BASE = dict(
    font=_CHART_FONT,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=45, b=10),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="rgba(0,0,0,0.08)",
        borderwidth=1,
        font=dict(size=11),
    ),
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Inter, Arial"),
)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS INJECTION
# ══════════════════════════════════════════════════════════════════════════════

def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Page background ─────────────────────────────────────────────── */
        .stApp { background-color: #F1F5F9; }

        /* ── Remove default top padding ──────────────────────────────────── */
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

        /* ── Sidebar ─────────────────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background-color: #1E293B;
        }
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] .stMarkdown h3,
        [data-testid="stSidebar"] .stMarkdown h4,
        [data-testid="stSidebar"] .stMarkdown li {
            color: #CBD5E1 !important;
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stMultiSelect label {
            color: #E2E8F0 !important;
            font-weight: 500 !important;
        }
        [data-testid="stSidebar"] [data-testid="stMetricValue"] {
            color: #FFFFFF !important;
        }

        /* ── Tab bar ─────────────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #FFFFFF;
            border-radius: 10px;
            padding: 4px 6px;
            gap: 2px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 22px;
            font-weight: 500;
            font-size: 0.88rem;
            color: #475569;
            border: none;
        }
        .stTabs [aria-selected="true"] {
            background-color: #3B82F6 !important;
            color: #FFFFFF !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }

        /* ── Dividers ────────────────────────────────────────────────────── */
        hr { border-color: #E2E8F0; margin: 1rem 0; }

        /* ── Streamlit branding ──────────────────────────────────────────── */
        #MainMenu { visibility: hidden; }
        footer     { visibility: hidden; }
        header     { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING  (cached)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading buyer data …")
def load_data() -> pd.DataFrame:
    """Load segmented_buyers.csv and enrich with derived columns."""
    if not os.path.exists(DATA_PATH):
        st.error(
            f"Dataset not found at `{DATA_PATH}`.  "
            "Run **model_pipeline.py** first."
        )
        st.stop()

    df = pd.read_csv(DATA_PATH, low_memory=False)

    # ── Derived columns ────────────────────────────────────────────────────
    df["cluster_name"] = df["kmeans_cluster"].apply(
        lambda c: f"Segment {c}"
    )
    df["is_investor"]    = (df["acquisition_purpose"] == "Investment").astype(int)
    df["has_loan"]       = (df["loan_applied"] == "Yes").astype(int)
    df["is_corporate"]   = (df["client_type"] == "Corporate").astype(int)
    df["cohort_display"] = df["cohort_label"].map(COHORT_LABELS).fillna(df["cohort_label"])
    return df


@st.cache_resource(show_spinner="Loading models …")
def load_models() -> dict:
    """Load all serialised model artifacts into a dict."""
    models = {}
    for name, path in [
        ("kmeans",  KMEANS_PATH),
        ("scaler",  SCALER_PATH),
        ("encoder", ENCODER_PATH),
    ]:
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def metric_card(
    label: str,
    value: str,
    subtitle: str = "",
    color: str = BRAND_BLUE,
) -> str:
    """Return an HTML metric card string for use with st.markdown."""
    sub_html = (
        f'<p style="color:{MUTED};font-size:0.78rem;margin:0.25rem 0 0;">{subtitle}</p>'
        if subtitle
        else ""
    )
    return f"""
    <div style="background:{CARD_BG};border-radius:12px;padding:1.1rem 1.4rem;
                box-shadow:0 1px 4px rgba(0,0,0,0.07);border-top:4px solid {color};
                height:100%;min-height:100px;">
      <p style="color:{MUTED};font-size:0.7rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.08em;margin:0 0 0.4rem 0;">{label}</p>
      <p style="color:{BRAND_NAVY};font-size:1.85rem;font-weight:800;
                line-height:1;margin:0;">{value}</p>
      {sub_html}
    </div>"""


def section_header(title: str, subtitle: str = "") -> None:
    """Render a styled section header."""
    sub = (
        f'<p style="color:{MUTED};font-size:0.85rem;margin:0.2rem 0 1rem;">'
        f"{subtitle}</p>"
        if subtitle
        else '<div style="margin-bottom:1rem;"></div>'
    )
    st.markdown(
        f'<h4 style="color:{BRAND_NAVY};font-size:1rem;font-weight:700;'
        f'margin:0;">{title}</h4>{sub}',
        unsafe_allow_html=True,
    )


def get_cluster_color_map(df: pd.DataFrame) -> dict[str, str]:
    """Map each cluster_name to a hex colour from CLUSTER_PALETTE."""
    names = sorted(df["cluster_name"].unique())
    return {name: CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i, name in enumerate(names)}


def apply_chart_defaults(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply shared layout defaults to any Plotly figure."""
    fig.update_layout(**_CHART_BASE)
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=13, color=BRAND_NAVY, family="Inter, Arial, sans-serif"),
                x=0,
                xanchor="left",
                pad=dict(b=10),
            )
        )
    return fig


def build_cluster_profile(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Compute a per-group business metric profile table.

    Columns returned
    ────────────────
    Size, Avg Age, Avg Satisfaction, Corporate %, Investment %, Loan Yes %
    """
    g = df.groupby(group_col)
    return pd.DataFrame({
        "Size":           g.size(),
        "Avg Age":        g["age"].mean().round(1),
        "Avg Satisfaction": g["satisfaction_score"].mean().round(2),
        "Corporate %":    (g["is_corporate"].mean() * 100).round(1),
        "Investment %":   (g["is_investor"].mean()  * 100).round(1),
        "Loan Yes %":     (g["has_loan"].mean()      * 100).round(1),
    })


def check_empty(df: pd.DataFrame) -> bool:
    """Show a warning and return True if DataFrame is empty after filtering."""
    if df.empty:
        st.warning(
            "⚠️  No records match the current filter combination.  "
            "Try widening your selections."
        )
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def build_sidebar(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """
    Render sidebar branding, filters, and model info.
    Returns the filtered DataFrame.
    """
    with st.sidebar:

        # ── Branding ──────────────────────────────────────────────────────────
        st.markdown(
            """
            <div style="text-align:center;padding:1.2rem 0 1rem;
                        border-bottom:1px solid #334155;">
              <div style="font-size:2.2rem;">🏡</div>
              <div style="color:#F8FAFC;font-weight:800;font-size:1.15rem;
                          margin-top:0.3rem;letter-spacing:-0.02em;">
                Parcl Analytics
              </div>
              <div style="color:#94A3B8;font-size:0.75rem;margin-top:0.1rem;">
                Buyer Intelligence Platform
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Filters ───────────────────────────────────────────────────────────
        st.markdown(
            '<p style="color:#94A3B8;font-size:0.7rem;font-weight:700;'
            'text-transform:uppercase;letter-spacing:0.1em;'
            'margin:1.2rem 0 0.6rem;">⚙️ Dashboard Filters</p>',
            unsafe_allow_html=True,
        )

        # Country multiselect
        all_countries = sorted(df["country"].unique())
        sel_countries = st.multiselect(
            "🌍 Country",
            options=all_countries,
            default=all_countries,
            help="Select one or more countries to focus on.",
        )

        # Region — dynamically constrained by country selection
        if sel_countries:
            region_pool = sorted(
                df[df["country"].isin(sel_countries)]["region"].unique()
            )
        else:
            region_pool = sorted(df["region"].unique())

        sel_regions = st.multiselect(
            "📍 Region",
            options=region_pool,
            default=region_pool,
            help="Regions update automatically based on selected countries.",
        )

        # Acquisition purpose
        purpose_opts = ["All"] + sorted(df["acquisition_purpose"].unique())
        sel_purpose  = st.selectbox(
            "🎯 Acquisition Purpose",
            options=purpose_opts,
            help="Filter by buyer acquisition intent.",
        )

        # Client type
        type_opts     = ["All"] + sorted(df["client_type"].unique())
        sel_ctype     = st.selectbox(
            "👤 Client Type",
            options=type_opts,
            help="Filter by Individual or Corporate clients.",
        )

        # Reset button
        st.markdown("<div style='margin-top:0.6rem;'></div>", unsafe_allow_html=True)
        if st.button("↺  Reset All Filters", use_container_width=True):
            st.rerun()

        # ── Apply filters ─────────────────────────────────────────────────────
        filtered = df.copy()
        if sel_countries:
            filtered = filtered[filtered["country"].isin(sel_countries)]
        if sel_regions:
            filtered = filtered[filtered["region"].isin(sel_regions)]
        if sel_purpose != "All":
            filtered = filtered[filtered["acquisition_purpose"] == sel_purpose]
        if sel_ctype != "All":
            filtered = filtered[filtered["client_type"] == sel_ctype]

        # ── Filter summary ────────────────────────────────────────────────────
        pct = len(filtered) / len(df) * 100
        st.markdown(
            f'<div style="background:#0F172A;border-radius:8px;padding:0.8rem 1rem;'
            f'margin-top:1rem;">'
            f'<p style="color:#94A3B8;font-size:0.7rem;margin:0 0 0.2rem;">Showing</p>'
            f'<p style="color:#F8FAFC;font-size:1.3rem;font-weight:800;margin:0;">'
            f'{len(filtered):,}</p>'
            f'<p style="color:#94A3B8;font-size:0.72rem;margin:0;">'
            f'of {len(df):,} records ({pct:.0f}%)</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Model info ────────────────────────────────────────────────────────
        if models:
            km = models.get("kmeans")
            k  = km.n_clusters if km else df["kmeans_cluster"].nunique()
            st.markdown(
                f'<div style="margin-top:1.2rem;padding-top:1rem;'
                f'border-top:1px solid #334155;">'
                f'<p style="color:#94A3B8;font-size:0.7rem;font-weight:700;'
                f'text-transform:uppercase;letter-spacing:0.1em;margin:0 0 0.5rem;">'
                f'🤖 Model Info</p>'
                f'<p style="color:#CBD5E1;font-size:0.8rem;margin:0 0 0.2rem;">'
                f'Algorithm: <b>K-Means</b></p>'
                f'<p style="color:#CBD5E1;font-size:0.8rem;margin:0 0 0.2rem;">'
                f'Clusters (k): <b>{k}</b></p>'
                f'<p style="color:#CBD5E1;font-size:0.8rem;margin:0;">'
                f'Inertia: <b>{km.inertia_:,.0f}</b></p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BUYER SEGMENTATION OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def tab_segmentation_overview(df: pd.DataFrame, df_full: pd.DataFrame) -> None:
    """
    KPI metrics, cluster distribution charts, and purpose breakdown.
    Uses kmeans_cluster (ML output) as the primary grouping variable.
    """
    if check_empty(df):
        return

    color_map = get_cluster_color_map(df_full)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    section_header("📊 Key Performance Indicators", "Aggregated across all filtered records.")
    k1, k2, k3, k4 = st.columns(4)

    total_buyers     = len(df)
    avg_satisfaction = df["satisfaction_score"].mean()
    loan_rate        = df["has_loan"].mean() * 100
    dominant_seg     = df["cluster_name"].value_counts().index[0]
    dominant_pct     = df["cluster_name"].value_counts().iloc[0] / len(df) * 100

    k1.markdown(
        metric_card("Total Buyers", f"{total_buyers:,}",
                    f"{len(df_full):,} in full dataset", BRAND_BLUE),
        unsafe_allow_html=True,
    )
    k2.markdown(
        metric_card("Dominant Segment", dominant_seg,
                    f"{dominant_pct:.0f}% of filtered records", "#8B5CF6"),
        unsafe_allow_html=True,
    )
    k3.markdown(
        metric_card("Avg Satisfaction", f"{avg_satisfaction:.2f} / 5",
                    f"{'▲' if avg_satisfaction > 3.5 else '▼'} vs mid-scale 3.5",
                    "#10B981"),
        unsafe_allow_html=True,
    )
    k4.markdown(
        metric_card("Loan Application Rate", f"{loan_rate:.1f}%",
                    "Buyers who applied for financing", "#F59E0B"),
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # ── Charts Row 1: Cluster Donut  |  Cohort Label Distribution ────────────
    section_header("🍩 Cluster & Segment Distribution")
    c1, c2 = st.columns(2)

    with c1:
        # ML cluster donut
        cluster_counts = (
            df["cluster_name"]
            .value_counts()
            .reset_index()
            .rename(columns={"cluster_name": "Segment", "count": "Count"})
        )
        fig_donut = px.pie(
            cluster_counts,
            names="Segment",
            values="Count",
            hole=0.55,
            color="Segment",
            color_discrete_map=color_map,
        )
        fig_donut.update_traces(
            textposition="outside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Buyers: %{value:,}<br>Share: %{percent}<extra></extra>",
        )
        fig_donut = apply_chart_defaults(fig_donut, "ML Cluster Distribution")
        fig_donut.update_layout(showlegend=False, margin=dict(l=10, r=10, t=50, b=30))
        st.plotly_chart(fig_donut, use_container_width=True)

    with c2:
        # Ground-truth cohort horizontal bar
        cohort_counts = (
            df["cohort_display"]
            .value_counts()
            .reset_index()
            .rename(columns={"cohort_display": "Cohort", "count": "Count"})
        )
        # Map colours via original cohort_label key
        cohort_color_map = {
            COHORT_LABELS.get(k, k): v for k, v in COHORT_PALETTE.items()
        }
        fig_bar = px.bar(
            cohort_counts,
            x="Count",
            y="Cohort",
            orientation="h",
            color="Cohort",
            color_discrete_map=cohort_color_map,
            text="Count",
        )
        fig_bar.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Count: %{x:,}<extra></extra>",
        )
        fig_bar.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
        fig_bar = apply_chart_defaults(fig_bar, "Ground-Truth Cohort Breakdown")
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Chart Row 2: Acquisition Purpose × Cluster ───────────────────────────
    section_header("🎯 Acquisition Purpose by Segment")
    purpose_seg = (
        df.groupby(["cluster_name", "acquisition_purpose"])
        .size()
        .reset_index(name="Count")
    )
    fig_grouped = px.bar(
        purpose_seg,
        x="cluster_name",
        y="Count",
        color="acquisition_purpose",
        barmode="group",
        color_discrete_sequence=["#3B82F6", "#10B981"],
        labels={"cluster_name": "Segment", "acquisition_purpose": "Purpose"},
        text="Count",
    )
    fig_grouped.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{legendgroup}: %{y:,}<extra></extra>",
    )
    fig_grouped = apply_chart_defaults(fig_grouped, "Acquisition Purpose Split by Segment")
    fig_grouped.update_layout(margin=dict(t=55, b=10))
    st.plotly_chart(fig_grouped, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INVESTOR BEHAVIOR DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def tab_investor_behavior(df: pd.DataFrame, df_full: pd.DataFrame) -> None:
    """
    Cross-tabulation charts exploring behavioral signals across ML clusters.
    """
    if check_empty(df):
        return

    color_map = get_cluster_color_map(df_full)

    # ── Row 1: Loan Rate + Satisfaction  ──────────────────────────────────────
    section_header(
        "💳 Financing & Satisfaction by Segment",
        "Key financial and satisfaction indicators per ML cluster.",
    )
    c1, c2 = st.columns(2)

    with c1:
        loan_rate = (
            df.groupby("cluster_name")["has_loan"]
            .mean()
            .mul(100)
            .reset_index(name="Loan Rate (%)")
        )
        fig_loan = px.bar(
            loan_rate,
            x="cluster_name",
            y="Loan Rate (%)",
            color="cluster_name",
            color_discrete_map=color_map,
            text="Loan Rate (%)",
            labels={"cluster_name": "Segment"},
        )
        fig_loan.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Loan rate: %{y:.1f}%<extra></extra>",
        )
        fig_loan.update_layout(showlegend=False, yaxis=dict(range=[0, 105]))
        fig_loan = apply_chart_defaults(fig_loan, "Loan Application Rate by Segment")
        st.plotly_chart(fig_loan, use_container_width=True)

    with c2:
        sat_dist = df[["cluster_name", "satisfaction_score"]].copy()
        fig_violin = px.violin(
            sat_dist,
            x="cluster_name",
            y="satisfaction_score",
            color="cluster_name",
            color_discrete_map=color_map,
            box=True,
            points=False,
            labels={"cluster_name": "Segment", "satisfaction_score": "Score"},
        )
        fig_violin.update_layout(showlegend=False, yaxis=dict(range=[0.5, 5.5]))
        fig_violin = apply_chart_defaults(fig_violin, "Satisfaction Score Distribution")
        st.plotly_chart(fig_violin, use_container_width=True)

    # ── Row 2: 100% Stacked Acquisition Purpose ───────────────────────────────
    section_header(
        "🎯 Acquisition Purpose Split (Normalised)",
        "Investment vs Personal Use proportion within each segment.",
    )
    purpose_norm = (
        df.groupby(["cluster_name", "acquisition_purpose"])
        .size()
        .groupby(level=0, group_keys=False)
        .apply(lambda x: (x / x.sum() * 100).round(1))
        .reset_index(name="Percentage")
    )
    fig_stacked = px.bar(
        purpose_norm,
        x="cluster_name",
        y="Percentage",
        color="acquisition_purpose",
        barmode="stack",
        color_discrete_sequence=["#3B82F6", "#10B981"],
        text="Percentage",
        labels={"cluster_name": "Segment", "acquisition_purpose": "Purpose"},
    )
    fig_stacked.update_traces(
        texttemplate="%{text:.0f}%",
        textposition="inside",
        textfont_color="white",
        hovertemplate="<b>%{x}</b> — %{fullData.name}<br>%{y:.1f}%<extra></extra>",
    )
    fig_stacked.update_layout(yaxis=dict(ticksuffix="%"))
    fig_stacked = apply_chart_defaults(
        fig_stacked, "Investment vs Personal Use (%) by Segment"
    )
    st.plotly_chart(fig_stacked, use_container_width=True)

    # ── Row 3: Referral Channel Mix ───────────────────────────────────────────
    section_header(
        "📡 Referral Channel Mix by Segment",
        "How buyers in each segment were acquired.",
    )
    channel_data = (
        df.groupby(["cluster_name", "referral_channel"])
        .size()
        .reset_index(name="Count")
    )
    fig_channel = px.bar(
        channel_data,
        x="cluster_name",
        y="Count",
        color="referral_channel",
        barmode="stack",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        labels={"cluster_name": "Segment", "referral_channel": "Channel"},
        hover_data={"Count": True},
    )
    fig_channel.update_traces(
        hovertemplate="<b>%{x}</b> — %{fullData.name}<br>Buyers: %{y:,}<extra></extra>"
    )
    fig_channel = apply_chart_defaults(
        fig_channel, "Referral Channel Composition by Segment"
    )
    st.plotly_chart(fig_channel, use_container_width=True)

    # ── Row 4: Age × Satisfaction Scatter ────────────────────────────────────
    section_header(
        "📐 Age vs Satisfaction — Cluster Scatter",
        "Visual separation between segments in age–satisfaction space.",
    )
    sample = df.sample(min(len(df), 1500), random_state=42)
    fig_scatter = px.scatter(
        sample,
        x="age",
        y="satisfaction_score",
        color="cluster_name",
        color_discrete_map=color_map,
        opacity=0.55,
        size_max=6,
        labels={
            "age":                "Age (years)",
            "satisfaction_score": "Satisfaction Score",
            "cluster_name":       "Segment",
        },
        hover_data={"country": True, "acquisition_purpose": True},
    )
    fig_scatter.update_traces(marker=dict(size=5))
    fig_scatter = apply_chart_defaults(
        fig_scatter, "Age vs Satisfaction Score (sample ≤ 1,500)"
    )
    fig_scatter.update_layout(
        xaxis=dict(gridcolor="#E2E8F0"),
        yaxis=dict(gridcolor="#E2E8F0", range=[0.5, 5.5]),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GEOGRAPHIC BUYER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def tab_geographic_analysis(df: pd.DataFrame, df_full: pd.DataFrame) -> None:
    """
    Country and region breakdowns across buyer profiles and ML clusters.
    """
    if check_empty(df):
        return

    color_map = get_cluster_color_map(df_full)

    # ── Country buyer volume (horizontal bar) ─────────────────────────────────
    section_header(
        "🌍 Buyer Volume by Country",
        "Total filtered buyers per country, ordered by volume.",
    )
    country_counts = (
        df.groupby("country")
        .size()
        .reset_index(name="Buyers")
        .sort_values("Buyers", ascending=True)
    )
    fig_country = px.bar(
        country_counts,
        x="Buyers",
        y="country",
        orientation="h",
        color="Buyers",
        color_continuous_scale=["#BFDBFE", BRAND_BLUE],
        text="Buyers",
        labels={"country": "Country"},
    )
    fig_country.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Buyers: %{x:,}<extra></extra>",
    )
    fig_country.update_layout(coloraxis_showscale=False, yaxis_title="")
    fig_country = apply_chart_defaults(fig_country, "Total Buyers by Country")
    fig_country.update_layout(height=max(350, len(country_counts) * 30))
    st.plotly_chart(fig_country, use_container_width=True)

    # ── Treemap  |  Cluster Heatmap ──────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        section_header("🗺️ Country → Region Drill-Down")
        treemap_df = (
            df.groupby(["country", "region"])
            .size()
            .reset_index(name="Buyers")
        )
        fig_tree = px.treemap(
            treemap_df,
            path=["country", "region"],
            values="Buyers",
            color="Buyers",
            color_continuous_scale=["#BFDBFE", BRAND_BLUE, "#1E3A5F"],
            hover_data={"Buyers": True},
        )
        fig_tree.update_traces(
            hovertemplate="<b>%{label}</b><br>Buyers: %{value:,}<extra></extra>",
            textinfo="label+value",
        )
        fig_tree = apply_chart_defaults(
            fig_tree, "Buyers by Country & Region (Treemap)"
        )
        fig_tree.update_layout(coloraxis_showscale=False, margin=dict(t=50, l=5, r=5, b=5))
        st.plotly_chart(fig_tree, use_container_width=True)

    with c2:
        section_header("🔥 Country × ML Segment Heatmap")
        top_countries = (
            df["country"].value_counts().head(10).index.tolist()
        )
        heat_df = (
            df[df["country"].isin(top_countries)]
            .groupby(["country", "cluster_name"])
            .size()
            .reset_index(name="Count")
        )
        heat_pivot = heat_df.pivot(
            index="country", columns="cluster_name", values="Count"
        ).fillna(0)

        fig_heat = px.imshow(
            heat_pivot,
            color_continuous_scale=["#EFF6FF", "#1D4ED8"],
            text_auto=True,
            aspect="auto",
            labels={"x": "Segment", "y": "Country", "color": "Buyers"},
        )
        fig_heat.update_traces(
            hovertemplate="Country: <b>%{y}</b><br>Segment: <b>%{x}</b><br>"
                          "Buyers: %{z:,}<extra></extra>",
            textfont=dict(size=11),
        )
        fig_heat = apply_chart_defaults(
            fig_heat, "Heatmap: Top-10 Countries × ML Segment"
        )
        fig_heat.update_layout(
            coloraxis_showscale=False,
            xaxis_title="",
            yaxis_title="",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Stacked Country × Purpose ─────────────────────────────────────────────
    section_header(
        "🎯 Acquisition Purpose Mix — Top 10 Countries",
        "Investment vs Personal Use split per country.",
    )
    top10 = df["country"].value_counts().head(10).index
    cp_df = (
        df[df["country"].isin(top10)]
        .groupby(["country", "acquisition_purpose"])
        .size()
        .reset_index(name="Count")
    )
    fig_cp = px.bar(
        cp_df,
        x="country",
        y="Count",
        color="acquisition_purpose",
        barmode="stack",
        color_discrete_sequence=["#3B82F6", "#10B981"],
        labels={"country": "Country", "acquisition_purpose": "Purpose"},
    )
    fig_cp.update_traces(
        hovertemplate="<b>%{x}</b> — %{fullData.name}<br>Buyers: %{y:,}<extra></extra>"
    )
    fig_cp = apply_chart_defaults(
        fig_cp, "Acquisition Purpose by Country (Top 10)"
    )
    st.plotly_chart(fig_cp, use_container_width=True)

    # ── Client Type by Country ────────────────────────────────────────────────
    section_header("🏢 Corporate vs Individual Mix — Top 10 Countries")
    ct_df = (
        df[df["country"].isin(top10)]
        .groupby(["country", "client_type"])
        .size()
        .groupby(level=0, group_keys=False)
        .apply(lambda x: (x / x.sum() * 100).round(1))
        .reset_index(name="Percentage")
    )
    fig_ct = px.bar(
        ct_df,
        x="country",
        y="Percentage",
        color="client_type",
        barmode="stack",
        color_discrete_sequence=["#6366F1", "#F59E0B"],
        text="Percentage",
        labels={"country": "Country", "client_type": "Client Type"},
    )
    fig_ct.update_traces(
        texttemplate="%{text:.0f}%",
        textposition="inside",
        textfont_color="white",
    )
    fig_ct.update_layout(yaxis=dict(ticksuffix="%"))
    fig_ct = apply_chart_defaults(
        fig_ct, "Corporate vs Individual Split by Country (%)"
    )
    st.plotly_chart(fig_ct, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SEGMENT INSIGHTS PANEL
# ══════════════════════════════════════════════════════════════════════════════

def tab_segment_insights(df: pd.DataFrame) -> None:
    """
    Descriptive statistics per ground-truth cohort (C1–C4) and a radar chart
    comparing cohort profiles across five normalised business dimensions.
    Uses cohort_label for accurate segment interpretation.
    """
    if check_empty(df):
        return

    # ── Introductory note ─────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="background:#EFF6FF;border-left:4px solid {BRAND_BLUE};
                    border-radius:6px;padding:0.85rem 1.1rem;margin-bottom:1.2rem;">
          <b style="color:{BRAND_NAVY};">ℹ️  About This Panel</b>
          <p style="color:{MUTED};font-size:0.85rem;margin:0.3rem 0 0;">
            This tab uses the <code>cohort_label</code> column — the ground-truth
            cluster injected in Phase 1 — to provide interpretable segment profiles
            for C1, C2, C3, and C4.  The ML clusters in Tabs 1–3 are unsupervised
            assignments from K-Means.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Profile statistics table ──────────────────────────────────────────────
    section_header(
        "📋 Cohort Profile Statistics",
        "Mean metrics per cohort label across the filtered dataset.",
    )

    profile = build_cluster_profile(df, "cohort_label")
    profile.index = profile.index.map(lambda x: COHORT_LABELS.get(x, x))
    profile = profile.sort_values("Size", ascending=False)

    # Style the dataframe
    def _style_profile(styler):
        bar_color  = "#BFDBFE"
        styler = (
            styler
            .format({
                "Size":              "{:,.0f}",
                "Avg Age":           "{:.1f} yrs",
                "Avg Satisfaction":  "{:.2f} / 5",
                "Corporate %":       "{:.1f}%",
                "Investment %":      "{:.1f}%",
                "Loan Yes %":        "{:.1f}%",
            })
            .background_gradient(subset=["Avg Satisfaction"], cmap="Blues", vmin=1, vmax=5)
            .background_gradient(subset=["Investment %"],     cmap="Greens")
            .background_gradient(subset=["Loan Yes %"],       cmap="Oranges")
            .set_properties(**{"text-align": "center"})
            .set_table_styles([{
                "selector": "th",
                "props": [
                    ("background-color", BRAND_NAVY),
                    ("color", "white"),
                    ("font-size", "0.78rem"),
                    ("text-transform", "uppercase"),
                    ("letter-spacing", "0.05em"),
                    ("padding", "8px 12px"),
                ],
            }])
        )
        return styler

    st.dataframe(
        profile.style.pipe(_style_profile),
        use_container_width=True,
        height=240,
    )

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # ── Cohort description cards (2 × 2) ─────────────────────────────────────
    section_header(
        "💡 Cohort Insight Cards",
        "Qualitative profiles for each of the four target buyer segments.",
    )

    main_cohorts = ["C1_Global_Investor", "C2_First_Time_Buyer",
                    "C3_Corporate_Buyer",  "C4_Luxury_Investor"]
    cols = st.columns(2)

    for i, cohort_key in enumerate(main_cohorts):
        info  = COHORT_DESCRIPTIONS.get(cohort_key, {})
        color = COHORT_PALETTE.get(cohort_key, BRAND_BLUE)
        bullets_html = "".join(
            f'<li style="color:{MUTED};font-size:0.82rem;margin:0.2rem 0;">'
            f'{b}</li>' for b in info.get("bullets", [])
        )

        # Pull live stats from the filtered data for this cohort
        mask       = df["cohort_label"] == cohort_key
        cohort_df  = df[mask]
        n          = len(cohort_df)
        avg_sat    = cohort_df["satisfaction_score"].mean() if n > 0 else 0
        invest_pct = cohort_df["is_investor"].mean() * 100 if n > 0 else 0

        card_html = f"""
        <div style="background:white;border-radius:12px;padding:1.2rem 1.4rem;
                    box-shadow:0 1px 4px rgba(0,0,0,0.08);
                    border-top:5px solid {color};height:100%;margin-bottom:1rem;">
          <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.5rem;">
            <span style="font-size:1.6rem;">{info.get("icon","")}</span>
            <div>
              <p style="font-weight:800;font-size:0.95rem;color:{BRAND_NAVY};
                        margin:0;">{info.get("title","")}</p>
              <p style="color:{MUTED};font-size:0.75rem;margin:0;">
                {n:,} records in filtered view
              </p>
            </div>
          </div>
          <p style="color:#475569;font-size:0.83rem;margin:0.5rem 0 0.6rem;
                    font-style:italic;">"{info.get("summary","")}"</p>
          <ul style="padding-left:1rem;margin:0 0 0.8rem;">{bullets_html}</ul>
          <div style="display:flex;gap:1rem;padding-top:0.6rem;
                      border-top:1px solid #F1F5F9;">
            <div>
              <p style="color:{MUTED};font-size:0.68rem;text-transform:uppercase;
                        letter-spacing:0.06em;margin:0;">Avg Satisfaction</p>
              <p style="color:{color};font-size:1.15rem;font-weight:800;margin:0;">
                {avg_sat:.2f}</p>
            </div>
            <div>
              <p style="color:{MUTED};font-size:0.68rem;text-transform:uppercase;
                        letter-spacing:0.06em;margin:0;">Investment %</p>
              <p style="color:{color};font-size:1.15rem;font-weight:800;margin:0;">
                {invest_pct:.0f}%</p>
            </div>
          </div>
        </div>"""

        cols[i % 2].markdown(card_html, unsafe_allow_html=True)

    # ── Radar chart — multi-cohort comparison ─────────────────────────────────
    section_header(
        "📡 Radar Comparison — C1 to C4",
        "Normalised 0–1 profiles across five business dimensions per cohort.",
    )

    # Build normalised metrics per cohort
    radar_metrics = ["Satisfaction", "Investment Focus",
                     "Self-Funded", "Corporate Presence", "Seniority"]

    fig_radar = go.Figure()

    for cohort_key in main_cohorts:
        mask  = df["cohort_label"] == cohort_key
        cdf   = df[mask]
        if cdf.empty:
            continue
        color = COHORT_PALETTE[cohort_key]
        label = COHORT_LABELS[cohort_key]

        values = [
            cdf["satisfaction_score"].mean()    / 5,                    # Satisfaction (0-1)
            cdf["is_investor"].mean(),                                    # Investment Focus
            1 - cdf["has_loan"].mean(),                                  # Self-Funded (inv of loan)
            cdf["is_corporate"].mean(),                                   # Corporate Presence
            (cdf["age"].mean() - 20) / 55,                               # Seniority (norm 20-75)
        ]
        # Close the polygon
        r      = values + [values[0]]
        theta  = radar_metrics + [radar_metrics[0]]

        fig_radar.add_trace(go.Scatterpolar(
            r=r,
            theta=theta,
            fill="toself",
            fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba")
                            if color.startswith("rgb") else color + "26",
            line=dict(color=color, width=2.5),
            name=label,
            hovertemplate=f"<b>{label}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
        ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=9, color=MUTED),
                gridcolor="#E2E8F0",
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=BRAND_NAVY),
                gridcolor="#E2E8F0",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        **{k: v for k, v in _CHART_BASE.items() if k not in ("plot_bgcolor",)},
    )
    fig_radar = apply_chart_defaults(
        fig_radar,
        "Segment Radar: Satisfaction · Investment · Self-Funded · Corporate · Seniority",
    )
    fig_radar.update_layout(height=480)
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Cohort satisfaction histogram ─────────────────────────────────────────
    section_header("📊 Satisfaction Score Distribution by Cohort")
    hist_df = df[df["cohort_label"].isin(main_cohorts)].copy()
    hist_df["Cohort"] = hist_df["cohort_label"].map(COHORT_LABELS)

    fig_hist = px.histogram(
        hist_df,
        x="satisfaction_score",
        color="Cohort",
        barmode="overlay",
        opacity=0.65,
        nbins=5,
        color_discrete_map={v: COHORT_PALETTE[k] for k, v in COHORT_LABELS.items()},
        labels={"satisfaction_score": "Satisfaction Score", "Cohort": "Cohort"},
    )
    fig_hist.update_traces(
        hovertemplate="Score %{x}<br>Count: %{y:,}<extra></extra>"
    )
    fig_hist.update_layout(
        xaxis=dict(tickmode="linear", dtick=1),
        bargap=0.05,
    )
    fig_hist = apply_chart_defaults(
        fig_hist, "Satisfaction Score Histogram — C1 to C4"
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    inject_css()

    # ── Load data & models ────────────────────────────────────────────────────
    df_full = load_data()
    models  = load_models()

    # ── Page header ───────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,{BRAND_NAVY} 0%,#1E3A5F 100%);
                    padding:1.4rem 2rem;border-radius:14px;margin-bottom:1.4rem;">
          <div style="display:flex;align-items:center;gap:0.8rem;">
            <span style="font-size:2rem;">🏡</span>
            <div>
              <h1 style="color:white;margin:0;font-size:1.5rem;font-weight:800;
                          letter-spacing:-0.02em;">
                Parcl Buyer Intelligence Dashboard
              </h1>
              <p style="color:#94A3B8;margin:0.15rem 0 0;font-size:0.85rem;">
                ML-Based Buyer Segmentation &amp; Investment Profiling
                &nbsp;·&nbsp;
                <span style="color:{BRAND_TEAL};">{len(df_full):,} total records</span>
                &nbsp;·&nbsp;
                <span style="color:{BRAND_TEAL};">
                  {df_full["kmeans_cluster"].nunique()}-cluster K-Means
                </span>
              </p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar filters → filtered DataFrame ─────────────────────────────────
    df_filtered = build_sidebar(df_full, models)

    # ── Main tab layout ───────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊  Segmentation Overview",
        "📈  Investor Behavior",
        "🌍  Geographic Analysis",
        "💡  Segment Insights",
    ])

    with tab1:
        tab_segmentation_overview(df_filtered, df_full)

    with tab2:
        tab_investor_behavior(df_filtered, df_full)

    with tab3:
        tab_geographic_analysis(df_filtered, df_full)

    with tab4:
        tab_segment_insights(df_filtered)


if __name__ == "__main__":
    main()
