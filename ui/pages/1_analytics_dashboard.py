"""
Sprint 2 — Task 2.4: Analytics Dashboard (Streamlit page 1).

All Fabric queries run through FabricAnalytics (src/analytics/queries.py).
Results are cached for 5 minutes via @st.cache_data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Allow imports from repo root regardless of launch directory
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analytics.predictions import DropoutPredictor
from src.analytics.queries import FabricAnalytics

# ── Palette ───────────────────────────────────────────────────────────────────
THEME = "plotly_dark"
TEAL = "#14b8a6"
AMBER = "#f59e0b"
SLATE = "#64748b"
RED = "#ef4444"
GREEN = "#22c55e"
DEPT_COLORS = px.colors.qualitative.Safe

# ── Page config (also set in app.py; harmless to set again for direct run) ────
st.set_page_config(
    page_title="Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Connection (cached at session level) ──────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to Fabric SQL...")
def get_analytics() -> FabricAnalytics:
    return FabricAnalytics.from_env()


@st.cache_resource(show_spinner="Training dropout model...")
def get_predictor(_analytics: FabricAnalytics) -> DropoutPredictor:
    return DropoutPredictor.from_analytics(_analytics)


# ── Cached query functions ────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_kpis() -> dict:
    return get_analytics().get_overview_kpis()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_departments() -> list[str]:
    return get_analytics().get_departments()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_courses(department: str) -> list[dict]:
    return get_analytics().get_courses(department)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_funnel(course_id: str) -> list[dict]:
    return get_analytics().get_completion_funnel(course_id)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_heatmap(course_id: str) -> list[dict]:
    return get_analytics().get_engagement_heatmap(course_id)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_skill_gaps(department: str) -> list[dict]:
    return get_analytics().get_department_skill_gaps(department)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_course_ranking() -> list[dict]:
    return get_analytics().get_course_ranking()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_format_comparison() -> list[dict]:
    return get_analytics().get_completion_by_format()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_at_risk(course_id: str) -> list[dict]:
    return get_analytics().get_at_risk_learners(course_id)


# ── Chart builders ────────────────────────────────────────────────────────────

def _funnel_chart(rows: list[dict]) -> go.Figure:
    if not rows:
        return go.Figure().update_layout(template=THEME, title="No data")

    df = pd.DataFrame(rows)
    # Color gradient: green (high completion) → red (low completion)
    pcts = df["module_completion_pct"].fillna(0).tolist()
    colors = [
        f"rgb({int(255 * (1 - p/100))}, {int(180 * p/100)}, 60)"
        for p in pcts
    ]

    fig = go.Figure(go.Bar(
        y=df["module_id"],
        x=df["module_completion_pct"],
        orientation="h",
        marker_color=colors,
        text=df["module_completion_pct"].apply(lambda v: f"{v:.0f}%" if pd.notna(v) else "—"),
        textposition="outside",
        customdata=df[["learners_reached", "learners_completed", "avg_quiz_score"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Reached: %{customdata[0]}<br>"
            "Completed: %{customdata[1]}<br>"
            "Completion: %{x:.1f}%<br>"
            "Avg quiz score: %{customdata[2]:.0%}<extra></extra>"
        ),
    ))
    fig.update_layout(
        template=THEME,
        title="Module Completion Funnel",
        xaxis_title="Completion %",
        yaxis_title="Module",
        xaxis=dict(range=[0, 115]),
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _heatmap_chart(rows: list[dict]) -> go.Figure:
    if not rows:
        return go.Figure().update_layout(template=THEME, title="No data")

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index="module_id", columns="hour_of_day",
        values="event_count", aggfunc="sum", fill_value=0
    )
    # Ensure all 24 hours are present
    for h in range(24):
        if h not in pivot.columns:
            pivot[h] = 0
    pivot = pivot.sort_index()[sorted(pivot.columns)]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="Teal",
        hovertemplate="Module: %{y}<br>Hour: %{x}<br>Events: %{z}<extra></extra>",
    ))
    fig.update_layout(
        template=THEME,
        title="Engagement Heatmap (events by module × hour UTC)",
        xaxis_title="Hour of day (UTC)",
        yaxis_title="Module",
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _radar_chart(rows: list[dict], department: str) -> go.Figure:
    if not rows:
        return go.Figure().update_layout(template=THEME, title="No data")

    df = pd.DataFrame(rows).head(10)   # limit to top-10 skills by gap
    skills = df["skill_name"].tolist()
    actual = df["avg_proficiency"].tolist()
    target = [3.5] * len(skills)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=actual + [actual[0]],
        theta=skills + [skills[0]],
        fill="toself",
        name="Actual",
        line_color=TEAL,
        fillcolor=f"rgba(20,184,166,0.2)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=target + [target[0]],
        theta=skills + [skills[0]],
        name="Target (3.5)",
        line=dict(color=AMBER, dash="dash"),
        fill=None,
    ))
    fig.update_layout(
        template=THEME,
        title=f"Skill Proficiency — {department}",
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        legend=dict(orientation="h", y=-0.15),
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _format_comparison_chart(rows: list[dict]) -> go.Figure:
    if not rows:
        return go.Figure().update_layout(template=THEME, title="No data")

    df = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Avg Completion %",
        x=df["course_format"],
        y=df["avg_completion_rate"],
        marker_color=TEAL,
        text=df["avg_completion_rate"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        yaxis="y1",
    ))
    fig.add_trace(go.Bar(
        name="Avg Quiz Score",
        x=df["course_format"],
        y=(df["avg_score"] * 100).round(1),
        marker_color=AMBER,
        text=(df["avg_score"] * 100).apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        yaxis="y1",
        opacity=0.8,
    ))
    fig.update_layout(
        template=THEME,
        title="Completion & Score by Course Format",
        barmode="group",
        yaxis=dict(title="Percentage", range=[0, 115]),
        legend=dict(orientation="h", y=1.1),
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def _scatter_chart(rows: list[dict]) -> go.Figure:
    if not rows:
        return go.Figure().update_layout(template=THEME, title="No data")

    df = pd.DataFrame(rows)
    df["avg_score_pct"] = (df["avg_score"].fillna(0) * 100).round(1)

    median_cr = df["completion_rate_pct"].median()
    median_score = df["avg_score_pct"].median()

    fig = px.scatter(
        df,
        x="completion_rate_pct",
        y="avg_score_pct",
        size="enrolled",
        color="department",
        color_discrete_sequence=DEPT_COLORS,
        hover_name="title",
        hover_data={"course_format": True, "skill_level": True,
                    "enrolled": True, "department": False},
        template=THEME,
        labels={
            "completion_rate_pct": "Completion Rate (%)",
            "avg_score_pct": "Avg Quiz Score (%)",
            "enrolled": "Enrolled",
        },
        title="Course Effectiveness (size = enrolled learners)",
        size_max=30,
    )

    # Quadrant lines at medians
    fig.add_vline(x=median_cr, line_dash="dash", line_color=SLATE, opacity=0.5)
    fig.add_hline(y=median_score, line_dash="dash", line_color=SLATE, opacity=0.5)

    fig.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def _risk_badge(score: float | None) -> str:
    if score is None:
        return "—"
    if score >= 0.65:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


# ── Main page ─────────────────────────────────────────────────────────────────

def main() -> None:
    st.title("Learning Analytics Dashboard")

    # ── Connection guard ──────────────────────────────────────────────────────
    try:
        analytics = get_analytics()
    except Exception as exc:
        st.error(
            f"**Could not connect to Fabric SQL analytics endpoint.**\n\n"
            f"`{exc}`\n\n"
            "Check your `.env` values: `FABRIC_SQL_ENDPOINT` and `FABRIC_LAKEHOUSE_NAME`."
        )
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Filters")

        try:
            departments = fetch_departments()
        except Exception as exc:
            st.error(f"Failed to load departments: {exc}")
            departments = []

        selected_dept = st.selectbox("Department", departments or ["(none)"])

        try:
            courses = fetch_courses(selected_dept)
        except Exception:
            courses = []

        course_options = {c["title"]: c["course_id"] for c in courses}
        selected_course_title = st.selectbox("Course", list(course_options.keys()) or ["(none)"])
        selected_course_id = course_options.get(selected_course_title, "")

        st.divider()
        st.caption("Data refreshes every 5 minutes from Fabric SQL.")

        # Optional: show dropout model info
        with st.expander("Dropout model info"):
            try:
                predictor = get_predictor(analytics)
                info = predictor.model_info
                st.metric("ROC-AUC", f"{info['roc_auc']:.3f}")
                st.metric("Training samples", f"{info['n_samples']:,}")
                st.write("**Feature importances:**")
                fi_df = pd.DataFrame(
                    info["feature_importances"].items(),
                    columns=["Feature", "Importance"]
                ).sort_values("Importance", ascending=False)
                st.dataframe(fi_df, hide_index=True, width='stretch')
            except Exception as exc:
                st.warning(f"Model not available: {exc}")

    # ── Row 0: KPI bar ────────────────────────────────────────────────────────
    try:
        kpis = fetch_kpis()
    except Exception as exc:
        st.error(f"Failed to load KPIs: {exc}")
        kpis = {}

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Learners",    f"{kpis.get('total_learners', 0):,}")
    col2.metric("Active This Month", f"{kpis.get('active_this_month', 0):,}")
    col3.metric("Avg Completion",    f"{kpis.get('avg_completion_rate', 0):.1f}%")
    col4.metric("Total Courses",     f"{kpis.get('total_courses', 0):,}")

    st.divider()

    # ── Row 1: Funnel | Heatmap ───────────────────────────────────────────────
    if selected_course_id:
        col_a, col_b = st.columns(2)
        with col_a:
            try:
                funnel_rows = fetch_funnel(selected_course_id)
                st.plotly_chart(_funnel_chart(funnel_rows), width='stretch')
            except Exception as exc:
                st.error(f"Funnel: {exc}")

        with col_b:
            try:
                heatmap_rows = fetch_heatmap(selected_course_id)
                st.plotly_chart(_heatmap_chart(heatmap_rows), width='stretch')
            except Exception as exc:
                st.error(f"Heatmap: {exc}")
    else:
        st.info("Select a course in the sidebar to see the funnel and engagement heatmap.")

    # ── Row 2: Skill radar | Format comparison ────────────────────────────────
    col_c, col_d = st.columns(2)
    with col_c:
        try:
            gap_rows = fetch_skill_gaps(selected_dept)
            st.plotly_chart(_radar_chart(gap_rows, selected_dept), width='stretch')
        except Exception as exc:
            st.error(f"Skill gaps: {exc}")

    with col_d:
        try:
            fmt_rows = fetch_format_comparison()
            st.plotly_chart(_format_comparison_chart(fmt_rows), width='stretch')
        except Exception as exc:
            st.error(f"Format comparison: {exc}")

    # ── Row 3: Course scatter ──────────────────────────────────────────────────
    try:
        ranking_rows = fetch_course_ranking()
        st.plotly_chart(_scatter_chart(ranking_rows), width='stretch')
    except Exception as exc:
        st.error(f"Course scatter: {exc}")

    # ── Row 4: At-risk learners table ─────────────────────────────────────────
    if selected_course_id:
        st.subheader("At-Risk Learners")
        try:
            at_risk_rows = fetch_at_risk(selected_course_id)
            if at_risk_rows:
                df_risk = pd.DataFrame(at_risk_rows)

                # Add risk score from local model if available
                try:
                    predictor = get_predictor(analytics)
                    df_scored = predictor.predict_for_course(analytics, selected_course_id)
                    if not df_scored.empty:
                        df_risk = df_risk.merge(
                            df_scored[["learner_email", "dropout_risk"]],
                            on="learner_email", how="left"
                        )
                        df_risk["risk_tier"] = df_risk["dropout_risk"].apply(
                            lambda s: predictor.risk_label(s) if pd.notna(s) else "—"
                        )
                        df_risk["intervention"] = df_risk.apply(
                            lambda r: predictor.suggested_intervention(
                                r.get("dropout_risk", 0.5),
                                r.get("days_inactive", 0),
                            ),
                            axis=1,
                        )
                except Exception:
                    df_risk["risk_tier"] = "—"
                    df_risk["intervention"] = "—"

                # Format columns for display
                display_cols = {
                    "learner_name": "Name",
                    "department": "Dept",
                    "modules_started": "Modules Started",
                    "days_inactive": "Days Inactive",
                    "avg_score": "Avg Score",
                    "risk_tier": "Risk",
                    "intervention": "Suggested Action",
                }
                df_display = df_risk.rename(columns=display_cols)[
                    [c for c in display_cols.values() if c in df_risk.rename(columns=display_cols).columns]
                ]

                def _color_risk(val):
                    colors = {"High": "color: #ef4444", "Medium": "color: #f59e0b", "Low": "color: #22c55e"}
                    return colors.get(str(val), "")

                styled = df_display.style.map(_color_risk, subset=["Risk"]) if "Risk" in df_display.columns else df_display.style
                st.dataframe(styled, width='stretch', hide_index=True)
                st.caption(f"{len(df_risk)} learners inactive >7 days who haven't completed the course.")
            else:
                st.success("No at-risk learners for this course.")
        except Exception as exc:
            st.error(f"At-risk learners: {exc}")
    else:
        st.info("Select a course to view at-risk learners.")


if __name__ == "__main__":
    main()
else:
    main()
