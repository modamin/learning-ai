"""
Sprint 3 — Task 3.5: RAG chat with learner context.

Flow per user turn:
  1. Embed the user message via Foundry
  2. Run hybrid search against AI Search to retrieve relevant courses
  3. Build a context block: learner profile + search results
  4. Call gpt-4o-mini chat completions (streaming) with agent system prompt
  5. Return: assistant reply, updated history, retrieved course sources

The conversation history is maintained by the caller (Streamlit session state)
as a plain list of {"role": ..., "content": ...} dicts.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterator, TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from openai import OpenAI
    from src.search.searcher import CourseSearcher

from src.foundry.agent import SYSTEM_PROMPT

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

logger = logging.getLogger(__name__)


# ── Learner context builder ───────────────────────────────────────────────────

def build_learner_context(learner: dict, skill_gaps: list[dict], completed_courses: list[str]) -> str:
    """
    Format a learner's profile as a context string to inject into the system prompt.

    Parameters
    ----------
    learner          : row from dim_learners (name, email, department, job_title, skills)
    skill_gaps       : rows from gold_department_skills for the learner's department
    completed_courses: list of course_ids already completed
    """
    lines = [
        f"LEARNER PROFILE:",
        f"  Name:       {learner.get('name', 'Unknown')}",
        f"  Department: {learner.get('department', 'Unknown')}",
        f"  Job title:  {learner.get('job_title', 'Unknown')}",
        f"  Hire date:  {learner.get('hire_date', 'Unknown')}",
    ]

    # Current skills
    skills = learner.get("skills") or []
    if skills:
        lines.append("\nCURRENT SKILL PROFICIENCY (out of 5.0, target is 3.5):")
        for s in sorted(skills, key=lambda x: x.get("proficiency_level", 0)):
            prof = s.get("proficiency_level", 0)
            gap_marker = " <-- GAP" if prof < 3.5 else ""
            lines.append(f"  {s['skill_name']:30s} {prof:.1f}{gap_marker}")

    # Department-level skill gaps
    if skill_gaps:
        top_gaps = [g for g in skill_gaps if (g.get("skill_gap") or 0) > 0][:5]
        if top_gaps:
            lines.append("\nTOP DEPARTMENT SKILL GAPS:")
            for g in top_gaps:
                lines.append(
                    f"  {g['skill_name']:30s} avg={g['avg_proficiency']:.2f}  gap={g['skill_gap']:.2f}"
                )

    # Completed courses
    if completed_courses:
        lines.append(f"\nALREADY COMPLETED ({len(completed_courses)} courses):")
        for cid in completed_courses[:10]:
            lines.append(f"  {cid}")
        if len(completed_courses) > 10:
            lines.append(f"  ... and {len(completed_courses) - 10} more")

    return "\n".join(lines)


def _format_search_results(results: list[dict]) -> str:
    """Format AI Search results as a structured context block."""
    if not results:
        return "No courses found matching this query."

    lines = ["AVAILABLE COURSES FROM CATALOG:"]
    for i, r in enumerate(results, 1):
        skills = ", ".join(r.get("skills_taught") or [])
        lines += [
            f"\n[{i}] {r['title']} (ID: {r['course_id']})",
            f"    Department: {r.get('department', '?')}  |  Level: {r.get('skill_level', '?')}",
            f"    Format: {r.get('format', '?')}  |  Duration: {r.get('duration_hours', '?')}h",
            f"    Skills taught: {skills}",
            f"    Completion rate: {r.get('completion_rate') or 'N/A'}%  |  Avg score: {r.get('avg_score') or 'N/A'}",
            f"    Description: {(r.get('description') or '')[:200]}",
        ]
    return "\n".join(lines)


# ── Main chat function ────────────────────────────────────────────────────────

def chat(
    message: str,
    history: list[dict],
    openai_client: "OpenAI",
    searcher: "CourseSearcher",
    learner_context: str = "",
    department_filter: str | None = None,
    top_results: int = 5,
) -> tuple[str, list[dict], list[dict]]:
    """
    Send one user turn and return the assistant reply.

    Parameters
    ----------
    message          : user's latest message
    history          : prior turns [{"role": "user"|"assistant", "content": ...}]
    openai_client    : from AIProjectClient.get_openai_client()
    searcher         : CourseSearcher instance
    learner_context  : formatted string from build_learner_context()
    department_filter: optional department to narrow search
    top_results      : how many courses to retrieve from AI Search

    Returns
    -------
    (reply_text, updated_history, source_courses)
    """
    # 1. Retrieve relevant courses from AI Search
    sources = []
    if searcher is not None:
        try:
            sources = searcher.search(
                message,
                department=department_filter,
                top=top_results,
                semantic=True,
            )
        except Exception as exc:
            logger.warning("Search failed, proceeding without results: %s", exc)

    # 2. Build the enriched system message
    search_block = _format_search_results(sources)
    system_content = (
        SYSTEM_PROMPT
        + ("\n\n" + learner_context if learner_context else "")
        + f"\n\n<search_results>\n{search_block}\n</search_results>"
    )

    # 3. Assemble messages
    model = os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-4o-mini")
    messages = [{"role": "system", "content": system_content}] + history + [{"role": "user", "content": message}]

    # 4. Call chat completions
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=1500,
    )
    reply = response.choices[0].message.content

    # 5. Update history (without system message — keeps it clean for next turn)
    updated_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": reply},
    ]

    return reply, updated_history, sources


def stream_chat(
    message: str,
    history: list[dict],
    openai_client: "OpenAI",
    searcher: "CourseSearcher",
    learner_context: str = "",
    department_filter: str | None = None,
    top_results: int = 5,
) -> tuple[Iterator[str], list[dict], list[dict]]:
    """
    Streaming version of chat(). Returns a token iterator, updated history, and sources.

    Usage in Streamlit:
        token_iter, history, sources = stream_chat(...)
        reply = st.write_stream(token_iter)
    """
    # ── DEBUG ─────────────────────────────────────────────────────────────────
    logger.debug("=== USER INPUT ===\n%s", message)
    logger.debug("=== DEPARTMENT FILTER === %s", department_filter)

    # Retrieve + build system context (same as non-streaming)
    sources = []
    if searcher is not None:
        try:
            sources = searcher.search(message, department=department_filter, top=top_results, semantic=True)
        except Exception as exc:
            logger.warning("Search failed: %s", exc)

    logger.debug("=== AI SEARCH QUERY === %r  filter=%r  top=%d", message, department_filter, top_results)
    logger.debug("=== AI SEARCH RESULTS (%d) ===", len(sources))
    for i, s in enumerate(sources, 1):
        logger.debug("  [%d] %s (%s)  score=%.4f", i, s.get("title"), s.get("course_id"), s.get("@search.score", 0))

    search_block = _format_search_results(sources)
    system_content = (
        SYSTEM_PROMPT
        + ("\n\n" + learner_context if learner_context else "")
        + f"\n\n<search_results>\n{search_block}\n</search_results>"
    )
    model = os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-4o-mini")
    messages = [{"role": "system", "content": system_content}] + history + [{"role": "user", "content": message}]

    logger.debug("=== MESSAGES SENT TO MODEL (%s) ===", model)
    for m in messages:
        role = m["role"]
        content_preview = m["content"][:500] + ("…" if len(m["content"]) > 500 else "")
        logger.debug("  [%s] %s", role.upper(), content_preview)

    stream = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=1500,
        stream=True,
    )

    # Collect full reply for history while yielding tokens
    collected: list[str] = []

    def _token_gen() -> Iterator[str]:
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if token:
                collected.append(token)
                yield token
        logger.debug("=== MODEL OUTPUT ===\n%s", "".join(collected))

    full_reply_ref: list[str] = collected   # shared reference
    updated_history = history + [{"role": "user", "content": message}]
    # Note: assistant message is appended after streaming completes — caller must do this.
    # Return a sentinel so the UI can append after st.write_stream finishes.

    return _token_gen(), updated_history, sources, full_reply_ref
