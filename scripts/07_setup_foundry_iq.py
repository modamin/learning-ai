"""
Sprint 3 — Task 3.7 (script): Set up FoundryIQ agent with MCP knowledge base.

Steps:
  1. Create / update the ARM project connection (links Foundry → AI Search KB via MCP)
  2. Register (or update) the Educational Consultant agent with the MCP tool attached

Prerequisites (manual portal steps — see plan):
  A. Enable system-assigned managed identity on the Foundry project in Azure Portal
  B. Grant that managed identity "Search Index Data Reader" on the AI Search resource
  C. Create a FoundryIQ knowledge base in the AI Search portal (select course-catalog index)
  D. Set new env vars in .env: AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP,
     FOUNDRY_IQ_KB_NAME, FOUNDRY_KB_CONNECTION_NAME

Usage:
  uv run python scripts/07_setup_foundry_iq.py
  uv run python scripts/07_setup_foundry_iq.py --model gpt-4o
  uv run python scripts/07_setup_foundry_iq.py --skip-connection   # if connection already exists
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.foundry.agent import (
    AGENT_NAME,
    _parse_project_resource_id,
    create_or_update_agent,
    create_project_connection,
    get_project_client,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_REQUIRED_ENV = [
    "FOUNDRY_PROJECT_ENDPOINT",
    "AZURE_SUBSCRIPTION_ID",
    "AZURE_RESOURCE_GROUP",
    "FOUNDRY_IQ_KB_NAME",
    "FOUNDRY_KB_CONNECTION_NAME",
    "AZURE_SEARCH_ENDPOINT",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up FoundryIQ agent with MCP knowledge base.")
    parser.add_argument("--model", default=None, help="Model deployment name (overrides env var).")
    parser.add_argument("--agent-name", default=None, help="Agent name (default: educational-consultant).")
    parser.add_argument("--kb-name", default=None, help="Knowledge base name (overrides FOUNDRY_IQ_KB_NAME).")
    parser.add_argument("--connection-name", default=None, help="Connection name (overrides FOUNDRY_KB_CONNECTION_NAME).")
    parser.add_argument("--skip-connection", action="store_true",
                        help="Skip project connection creation (use if connection already exists).")
    args = parser.parse_args()

    # ── Validate env vars ─────────────────────────────────────────────────────
    missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        logger.error("Missing required env vars: %s", missing)
        logger.error(
            "Add them to .env:\n"
            "  AZURE_SUBSCRIPTION_ID=<guid>\n"
            "  AZURE_RESOURCE_GROUP=<rg-name>\n"
            "  FOUNDRY_IQ_KB_NAME=<kb-name>          # created in AI Search portal\n"
            "  FOUNDRY_KB_CONNECTION_NAME=<name>      # e.g. kb-mcp-connection\n"
        )
        sys.exit(1)

    # ── Show parsed resource ID for visibility ────────────────────────────────
    try:
        resource_id = _parse_project_resource_id()
        logger.info("Foundry project resource ID: %s", resource_id)
    except Exception as exc:
        logger.error("Cannot parse project resource ID: %s", exc)
        sys.exit(1)

    connection_name = args.connection_name or os.environ["FOUNDRY_KB_CONNECTION_NAME"]
    kb_name         = args.kb_name or os.environ["FOUNDRY_IQ_KB_NAME"]
    agent_name      = args.agent_name or AGENT_NAME
    model           = args.model or os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-4o-mini")

    # ── Step 1: Create / update project connection ────────────────────────────
    connection_id: str | None = None
    if args.skip_connection:
        logger.info("--skip-connection set — reusing existing connection '%s'.", connection_name)
        connection_id = f"{resource_id}/connections/{connection_name}"
    else:
        logger.info("Creating / updating project connection '%s' …", connection_name)
        try:
            connection_id = create_project_connection(
                connection_name=connection_name,
                kb_name=kb_name,
            )
        except Exception as exc:
            logger.error("Failed to create project connection: %s", exc)
            logger.error(
                "If this is a 403, the SP may lack ARM write access at the workspace level.\n"
                "Workaround: create the connection manually in the Foundry portal (Connections tab)\n"
                "then re-run with --skip-connection."
            )
            sys.exit(1)

    # ── Step 2: Create / update agent with MCPTool ────────────────────────────
    logger.info("Registering FoundryIQ agent '%s' with model '%s' …", agent_name, model)
    project = get_project_client()
    try:
        name = create_or_update_agent(
            project=project,
            model=model,
            agent_name=agent_name,
            use_foundry_iq=True,
            kb_connection_id=connection_id,
        )
    except Exception as exc:
        logger.error("Failed to create agent: %s", exc)
        sys.exit(1)

    logger.info("Done.")
    logger.info("")
    logger.info("Add to .env:  FOUNDRY_AGENT_NAME=%s", name)
    logger.info("")
    logger.info("Next: restart Streamlit and enable the 'Use FoundryIQ agent' toggle.")


if __name__ == "__main__":
    main()
