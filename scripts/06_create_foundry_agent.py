"""
Sprint 3 — Task 3.4 (script): Register the Educational Consultant agent in Foundry.

Usage:
  uv run python scripts/06_create_foundry_agent.py
  uv run python scripts/06_create_foundry_agent.py --model gpt-4o
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.foundry.agent import create_or_update_agent, get_project_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register the Educational Consultant agent in Foundry.")
    parser.add_argument("--model", default=None, help="Model deployment name (overrides env var).")
    args = parser.parse_args()

    if not os.environ.get("FOUNDRY_PROJECT_ENDPOINT"):
        logger.error("FOUNDRY_PROJECT_ENDPOINT not set in .env")
        sys.exit(1)

    logger.info("Connecting to Foundry project: %s", os.environ["FOUNDRY_PROJECT_ENDPOINT"])
    project = get_project_client()

    model = args.model or os.environ.get("MODEL_DEPLOYMENT_NAME", "gpt-4o-mini")
    logger.info("Registering agent with model: %s", model)

    name = create_or_update_agent(project=project, model=model)
    logger.info("Agent '%s' ready. Add to .env: FOUNDRY_AGENT_NAME=%s", name, name)


if __name__ == "__main__":
    main()
