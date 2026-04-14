"""
Sprint 3 — Task 3.2 (script): Embed and index all courses into Azure AI Search.

Steps:
  1. Create / update the AI Search index schema
  2. Load data/seed_courses.json
  3. Embed each course via Foundry (text-embedding-3-large)
  4. Enrich with live Fabric analytics (completion_rate, avg_score)
  5. Upload to AI Search in batches

Usage:
  uv run python scripts/05_index_courses_ai_search.py
  uv run python scripts/05_index_courses_ai_search.py --skip-analytics   # skip Fabric enrichment
  uv run python scripts/05_index_courses_ai_search.py --recreate          # drop + recreate index first
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.foundry.agent import get_openai_client
from src.search.index_manager import create_or_update_index, delete_index, get_index_stats
from src.search.indexer import index_courses

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Index courses into Azure AI Search.")
    parser.add_argument("--skip-analytics", action="store_true",
                        help="Skip Fabric SQL enrichment (completion_rate, avg_score).")
    parser.add_argument("--recreate", action="store_true",
                        help="Delete and recreate the index before indexing.")
    args = parser.parse_args()

    # ── Validate env vars ─────────────────────────────────────────────────────
    missing = [k for k in ["FOUNDRY_PROJECT_ENDPOINT", "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_ADMIN_KEY"]
               if not os.environ.get(k)]
    if missing:
        logger.error("Missing env vars: %s", missing)
        sys.exit(1)

    # ── Recreate index if requested ───────────────────────────────────────────
    if args.recreate:
        logger.info("Deleting existing index …")
        try:
            delete_index()
        except Exception as exc:
            logger.warning("Could not delete index (may not exist): %s", exc)

    # ── Create / update index schema ──────────────────────────────────────────
    logger.info("Creating / updating AI Search index …")
    create_or_update_index()

    # ── Load courses ──────────────────────────────────────────────────────────
    courses_path = REPO_ROOT / "data" / "seed_courses.json"
    if not courses_path.exists():
        logger.error("data/seed_courses.json not found. Run scripts/01_generate_synthetic_data.py first.")
        sys.exit(1)

    courses = json.loads(courses_path.read_text(encoding="utf-8"))
    logger.info("Loaded %d courses from %s", len(courses), courses_path)

    # ── Foundry OpenAI client (cognitiveservices scope — needs Cognitive Services OpenAI User role) ──
    openai_client = get_openai_client()
    embedding_model = os.environ.get("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large")
    logger.info("OpenAI client ready. Embedding model: %s", embedding_model)

    # ── Optional: Fabric analytics enrichment ────────────────────────────────
    analytics = None
    if not args.skip_analytics:
        try:
            from src.analytics.queries import FabricAnalytics
            analytics = FabricAnalytics.from_env()
            logger.info("Fabric analytics connected — will enrich completion_rate and avg_score.")
        except Exception as exc:
            logger.warning("Could not connect to Fabric SQL (%s). Skipping analytics enrichment.", exc)

    # ── Index ─────────────────────────────────────────────────────────────────
    total = index_courses(
        courses=courses,
        openai_client=openai_client,
        embedding_model=embedding_model,
        search_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        search_key=os.environ["AZURE_SEARCH_ADMIN_KEY"],
        index_name=os.environ.get("AZURE_SEARCH_INDEX_NAME", "course-catalog"),
        analytics=analytics,
    )

    # ── Final stats ───────────────────────────────────────────────────────────
    try:
        stats = get_index_stats()
        logger.info("Index stats: %d documents, %.1f KB storage",
                    stats["document_count"], stats["storage_size"] / 1024)
    except Exception:
        pass

    logger.info("Done. %d/%d courses indexed.", total, len(courses))


if __name__ == "__main__":
    main()
