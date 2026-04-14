"""
Sprint 3 — FoundryIQ chat: agent-backed RAG via MCP knowledge base.

Flow per user turn:
  1. Prepend learner context to the user message (if provided)
  2. Call openai_client.responses.create() with the agent reference and conversation ID
  3. The agent autonomously calls the FoundryIQ MCP tool to retrieve courses
  4. Return: assistant reply, extracted source citations

Conversation history is managed server-side by the Foundry conversation object.
The caller only needs to store the conversation_id across turns.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    pass

from src.foundry.agent import get_agent_name, get_project_client

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

logger = logging.getLogger(__name__)


# ── Client factory ────────────────────────────────────────────────────────────

def get_foundry_iq_openai_client():
    """
    Return an OpenAI client from AIProjectClient.get_openai_client().

    This uses scope https://ai.azure.com/.default and requires the
    'Azure AI Developer' role on the AI Services resource.
    """
    project = get_project_client()
    client = project.get_openai_client()
    logger.info("FoundryIQ OpenAI client ready.")
    return client


# ── Conversation management ───────────────────────────────────────────────────

def create_conversation(openai_client) -> str:
    """
    Create a new server-side conversation and return its ID.
    Call once per Streamlit session (or when the user clears / switches learner).
    """
    conversation = openai_client.conversations.create()
    logger.info("FoundryIQ conversation created: %s", conversation.id)
    return conversation.id


# ── Main chat function ────────────────────────────────────────────────────────

def chat_with_agent(
    message: str,
    conversation_id: str,
    openai_client,
    agent_name: str | None = None,
    learner_context: str = "",
) -> tuple[str, list[dict]]:
    """
    Send one user turn to the FoundryIQ agent.

    Parameters
    ----------
    message         : Raw user message text.
    conversation_id : ID from create_conversation() — accumulates history server-side.
    openai_client   : From get_foundry_iq_openai_client().
    agent_name      : Defaults to get_agent_name() (env var / .agent_name file).
    learner_context : Formatted string from build_learner_context(). If non-empty,
                      prepended to the user message so the agent knows who it's advising.

    Returns
    -------
    (reply_text, sources) where sources is a list of dicts extracted from the
    MCP tool output items in the response.
    """
    agent_name = agent_name or get_agent_name()

    # Inject learner context into the user message
    if learner_context:
        enriched_input = f"{learner_context}\n\n---\n\n{message}"
    else:
        enriched_input = message

    logger.debug("=== FOUNDRY IQ USER INPUT ===\n%s", message)
    logger.debug("=== FOUNDRY IQ AGENT === %s  conversation=%s", agent_name, conversation_id)

    response = openai_client.responses.create(
        conversation=conversation_id,
        input=enriched_input,
        extra_body={
            "agent_reference": {
                "name": agent_name,
                "type": "agent_reference",
            }
        },
    )

    reply = response.output_text
    logger.debug("=== FOUNDRY IQ REPLY ===\n%s", reply)
    logger.debug("=== FOUNDRY IQ RAW OUTPUT ===\n%s", response.output)

    sources = _extract_sources_from_response(response)
    return reply, sources


# ── Source extraction ─────────────────────────────────────────────────────────

def _extract_sources_from_response(response) -> list[dict]:
    """
    Walk response.output looking for MCP tool-call output items from the
    course-catalog-kb server. Parse their content into source dicts for
    display in the UI sources panel.

    Returns an empty list (never raises) — shape is explored at DEBUG level.
    """
    sources: list[dict] = []
    try:
        for item in response.output or []:
            # Tool call output items vary by SDK version; inspect defensively
            item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
            if item_type not in ("mcp_call", "tool_call", "tool_result", "function_call_output"):
                continue

            # Try to read the server label to confirm it's from our KB tool
            server_label = (
                getattr(item, "server_label", None)
                or (item.get("server_label") if isinstance(item, dict) else None)
            )
            if server_label and server_label != "course-catalog-kb":
                continue

            # Extract content — may be a string or list of content blocks
            content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else None)
            if not content:
                continue

            if isinstance(content, str):
                sources.append({"raw": content})
            elif isinstance(content, list):
                for block in content:
                    text = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else str(block))
                    if text:
                        sources.append({"raw": text})

    except Exception as exc:
        logger.warning("Could not extract sources from FoundryIQ response: %s", exc)

    return sources
