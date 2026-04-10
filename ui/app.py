"""
Sprint 2 — Task 2.5: Main Streamlit entry point.

Launch:
  streamlit run ui/app.py

Pages are discovered from ui/pages/ via st.navigation (Streamlit >= 1.36).
"""
from __future__ import annotations

import sys
import textwrap
import tempfile
from pathlib import Path

import streamlit as st

# Allow imports from repo root when launched via `streamlit run ui/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NovaTech Learning Platform",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "NovaTech Learning Analytics Platform — Azure PoC (Sprint 2)",
    },
)

# ── Global sidebar branding ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 12px 0 4px 0;'>
            <span style='font-size:2.2rem;'>&#127891;</span><br>
            <span style='font-size:1.05rem; font-weight:700; color:#14b8a6;'>NovaTech</span><br>
            <span style='font-size:0.78rem; color:#94a3b8;'>Learning Platform</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()


# ── Helper: write a stub page file for not-yet-built pages ───────────────────

def _ensure_stub(filename: str, title: str, sprint: str) -> str:
    """
    Write a minimal stub .py file to ui/pages/ if it doesn't exist,
    so st.navigation doesn't crash when pages aren't built yet.
    Returns the absolute path.
    """
    path = Path(__file__).parent / "pages" / filename
    if not path.exists():
        path.write_text(
            textwrap.dedent(f"""\
                import streamlit as st
                st.title("{title}")
                st.info("Coming in {sprint} — check back soon!")
            """),
            encoding="utf-8",
        )
    return str(path)


# ── Navigation ────────────────────────────────────────────────────────────────
pages_dir = Path(__file__).parent / "pages"

analytics_page = st.Page(
    str(pages_dir / "1_analytics_dashboard.py"),
    title="Analytics Dashboard",
    icon="📊",
    default=True,
)

consultant_page = st.Page(
    _ensure_stub("2_ai_consultant.py", "AI Educational Consultant", "Sprint 3"),
    title="AI Consultant",
    icon="🤖",
)

generator_page = st.Page(
    _ensure_stub("3_course_generator.py", "Course Generator", "Sprint 4"),
    title="Course Generator",
    icon="✏️",
)

pg = st.navigation(
    {
        "Analytics": [analytics_page],
        "AI Features": [consultant_page, generator_page],
    }
)
pg.run()
