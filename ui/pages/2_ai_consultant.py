"""
Sprint 3 — Task 3.6: AI Educational Consultant chat page.

Layout:
  Sidebar — learner picker, department filter, search settings, FoundryIQ toggle
  Main    — chat interface + collapsible Sources panel

Two chat modes:
  - FoundryIQ (toggle ON):  agent-backed via MCP knowledge base, conversation managed server-side
  - Manual RAG (toggle OFF): embed → AI Search → inject context → chat completions (original path)
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import streamlit as st

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
# Suppress noisy Azure/HTTP libs at DEBUG level
for _noisy in ("azure", "httpx", "httpcore", "openai._base_client", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analytics.queries import FabricAnalytics
from src.analytics.recommender import CourseRecommender
from src.foundry.agent import get_openai_client as _get_openai_client
from src.foundry.chat import build_learner_context, chat, stream_chat  # noqa: F401
from src.foundry.foundry_iq_chat import (
    chat_with_agent,
    create_conversation,
    get_foundry_iq_openai_client,
)
from src.search.searcher import CourseSearcher

st.set_page_config(
    page_title="AI Educational Consultant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to Fabric SQL...")
def get_analytics() -> FabricAnalytics:
    return FabricAnalytics.from_env()


@st.cache_resource(show_spinner="Connecting to Foundry...")
def get_openai_client():
    return _get_openai_client()


@st.cache_resource(show_spinner="Connecting to FoundryIQ...")
def get_foundry_iq_client():
    return get_foundry_iq_openai_client()


@st.cache_resource(show_spinner="Connecting to AI Search...")
def get_searcher():
    openai_client = get_openai_client()
    return CourseSearcher.from_env(openai_client)


@st.cache_resource(show_spinner="Training course recommender...")
def get_recommender() -> CourseRecommender:
    return CourseRecommender.from_analytics(get_analytics())


@st.cache_data(ttl=300, show_spinner=False)
def fetch_courses() -> list[dict]:
    return get_analytics().get_courses()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_departments() -> list[str]:
    return get_analytics().get_departments()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_learners(department: str) -> list[dict]:
    # skills and completed_courses are complex types not exposed by Fabric SQL endpoint
    sql = """
        SELECT learner_id, name, email, department, job_title, hire_date
        FROM dim_learners
        WHERE department = ?
        ORDER BY name
    """
    return get_analytics()._rows(sql, [department])


@st.cache_data(ttl=300, show_spinner=False)
def fetch_skill_gaps(department: str) -> list[dict]:
    return get_analytics().get_department_skill_gaps(department)


# ── Session state init ────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "learner_context" not in st.session_state:
    st.session_state.learner_context = ""
if "selected_learner" not in st.session_state:
    st.session_state.selected_learner = None
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None


def _reset_conversation():
    st.session_state.chat_history = []
    st.session_state.last_sources = []
    st.session_state.conversation_id = None


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Learner Profile")

    # Department picker
    try:
        departments = fetch_departments()
    except Exception as exc:
        st.error(f"Cannot load departments: {exc}")
        departments = []

    dept = st.selectbox("Department", departments or ["(none)"])

    # Learner picker
    learners: list[dict] = []
    if dept and dept != "(none)":
        try:
            learners = fetch_learners(dept)
        except Exception as exc:
            st.error(f"Cannot load learners: {exc}")

    learner_names = {l["name"]: l for l in learners}
    selected_name = st.selectbox(
        "Learner",
        ["(no learner selected)"] + list(learner_names.keys()),
    )

    if selected_name != "(no learner selected)":
        learner = learner_names[selected_name]
        if st.session_state.selected_learner != learner["email"]:
            # Learner changed — rebuild context and reset chat + conversation
            st.session_state.selected_learner = learner["email"]
            _reset_conversation()
            try:
                gaps = fetch_skill_gaps(dept)
            except Exception:
                gaps = []
            completed = learner.get("completed_courses") or []
            st.session_state.learner_context = build_learner_context(learner, gaps, completed)

        # Compact profile card
        with st.expander("Profile", expanded=True):
            st.write(f"**{learner['name']}**")
            st.write(f"{learner.get('job_title', '')}  ·  {learner.get('department', '')}")
            skills = learner.get("skills") or []
            if skills:
                st.write("**Skills (lowest first):**")
                for s in sorted(skills, key=lambda x: x.get("proficiency_level", 0))[:6]:
                    prof = s.get("proficiency_level", 0)
                    color = "🔴" if prof < 2.5 else ("🟡" if prof < 3.5 else "🟢")
                    st.write(f"{color} {s['skill_name']} — {prof:.1f}/5")
        # Collaborative filtering recommendations
        with st.expander("Recommended for you", expanded=False):
            try:
                recommender = get_recommender()
                recs = recommender.recommend(learner["email"], top_n=5)
                if recs:
                    course_map = {
                        c["course_id"]: c["title"]
                        for c in fetch_courses()
                    }
                    for i, r in enumerate(recs, 1):
                        title = course_map.get(r["course_id"], r["course_id"])
                        st.write(f"{i}. {title}")
                else:
                    st.write("No recommendations yet — not enough activity data.")
            except Exception as exc:
                st.warning(f"Recommender unavailable: {exc}")
    else:
        if st.session_state.selected_learner is not None:
            st.session_state.selected_learner = None
            st.session_state.learner_context = ""
            _reset_conversation()

    st.divider()
    st.subheader("Search settings")
    dept_filter = st.checkbox("Filter search to learner's department", value=True)
    top_k = st.slider("Courses retrieved per query", 3, 10, 5)

    # FoundryIQ toggle — only shown when knowledge base is configured
    foundry_iq_available = bool(os.environ.get("FOUNDRY_IQ_KB_NAME"))
    use_foundry_iq = False
    if foundry_iq_available:
        st.divider()
        use_foundry_iq = st.toggle("Use FoundryIQ agent", value=False,
                                   help="Route queries through a Foundry agent backed by a FoundryIQ MCP knowledge base.")
    st.divider()

    if st.button("Clear conversation"):
        _reset_conversation()
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────

st.title("AI Educational Consultant")

if use_foundry_iq:
    st.caption("Mode: FoundryIQ agent (MCP knowledge base)")
else:
    st.caption("Mode: Manual RAG (AI Search + chat completions)")

if not st.session_state.selected_learner:
    st.info(
        "Select a department and learner in the sidebar to personalise recommendations. "
        "You can also chat without a profile."
    )

# Render prior turns
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Sources from last turn
if st.session_state.last_sources:
    sources_list = st.session_state.last_sources
    # FoundryIQ sources are raw text dicts; RAG sources are structured course dicts
    is_structured = sources_list and "course_id" in sources_list[0]
    label = (
        f"Sources — {len(sources_list)} courses retrieved from AI Search"
        if is_structured
        else f"Sources — {len(sources_list)} result(s) from FoundryIQ knowledge base"
    )
    with st.expander(label, expanded=False):
        if is_structured:
            for i, src in enumerate(sources_list, 1):
                skills_str = ", ".join(src.get("skills_taught") or [])
                cr = src.get("completion_rate")
                cr_str = f"{cr:.0f}%" if cr is not None else "N/A"
                st.markdown(
                    f"**{i}. {src['title']}** `{src['course_id']}`  \n"
                    f"{src.get('department','?')} · {src.get('skill_level','?')} · "
                    f"{src.get('format','?')} · {src.get('duration_hours','?')}h  \n"
                    f"Skills: {skills_str or '—'}  |  Completion rate: {cr_str}"
                )
                if i < len(sources_list):
                    st.divider()
        else:
            for i, src in enumerate(sources_list, 1):
                st.markdown(f"**{i}.** {src.get('raw', str(src))}")
                if i < len(sources_list):
                    st.divider()

# Chat input
prompt = st.chat_input("Ask about courses, skill gaps, or learning paths …")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if use_foundry_iq:
            # ── FoundryIQ path ────────────────────────────────────────────────
            try:
                fiq_client = get_foundry_iq_client()
            except Exception as exc:
                st.error(f"**Cannot connect to FoundryIQ.**\n\n`{exc}`\n\n"
                         "Check `FOUNDRY_PROJECT_ENDPOINT` in `.env` and that the agent is registered "
                         "(`uv run python scripts/07_setup_foundry_iq.py`).")
                st.stop()

            # Lazy-create conversation for this session
            if st.session_state.conversation_id is None:
                try:
                    st.session_state.conversation_id = create_conversation(fiq_client)
                except Exception as exc:
                    st.error(f"Cannot create FoundryIQ conversation: {exc}")
                    st.stop()

            try:
                reply, sources = chat_with_agent(
                    message=prompt,
                    conversation_id=st.session_state.conversation_id,
                    openai_client=fiq_client,
                    learner_context=st.session_state.learner_context,
                )
                st.markdown(reply)
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.session_state.last_sources = sources
            except Exception as exc:
                st.error(f"FoundryIQ chat error: {exc}")

        else:
            # ── Manual RAG path ───────────────────────────────────────────────
            try:
                openai_client = get_openai_client()
            except Exception as exc:
                st.error(f"**Cannot connect to Foundry.**\n\n`{exc}`\n\n"
                         "Check `FOUNDRY_PROJECT_ENDPOINT` in `.env`.")
                st.stop()

            try:
                searcher = get_searcher()
            except Exception as exc:
                st.warning(f"AI Search unavailable — answering without course retrieval. ({exc})")
                searcher = None

            dept_search_filter = (
                dept if (st.session_state.selected_learner and dept_filter and dept != "(none)")
                else None
            )

            try:
                reply, updated_history, sources = chat(
                    message=prompt,
                    history=st.session_state.chat_history,
                    openai_client=openai_client,
                    searcher=searcher,
                    learner_context=st.session_state.learner_context,
                    department_filter=dept_search_filter,
                    top_results=top_k,
                )
                st.markdown(reply)
                st.session_state.chat_history = updated_history
                st.session_state.last_sources = sources
            except Exception as exc:
                st.error(f"Chat error: {exc}")

    st.rerun()
