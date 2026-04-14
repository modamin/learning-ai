"""
Sprint 3 — Task 3.2: Embed courses and upload to Azure AI Search.

Flow per course:
  1. Build a rich text blob: title + department + level + skills + description
  2. Embed via Foundry (text-embedding-3-large, 3072-dim)
  3. Enrich with live Fabric analytics: completion_rate, avg_score
  4. Upload to AI Search in batches of 50
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

if TYPE_CHECKING:
    from openai import OpenAI
    from src.analytics.queries import FabricAnalytics

logger = logging.getLogger(__name__)

BATCH_SIZE = 50
EMBED_RETRY_DELAY = 2   # seconds between retries on rate-limit


def _build_embed_text(course: dict) -> str:
    """Concatenate course fields into the string we'll embed."""
    skills = ", ".join(course.get("skills_taught") or [])
    return (
        f"Title: {course['title']}\n"
        f"Department: {course['department']}\n"
        f"Level: {course['skill_level']}\n"
        f"Format: {course['format']}\n"
        f"Duration: {course['duration_hours']} hours\n"
        f"Skills: {skills}\n"
        f"Description: {course['description']}"
    )


def embed_texts(texts: list[str], openai_client: "OpenAI", model: str) -> list[list[float]]:
    """Embed a batch of texts. Retries once on rate-limit."""
    for attempt in range(2):
        try:
            response = openai_client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in response.data]
        except Exception as exc:
            if attempt == 0 and "rate" in str(exc).lower():
                logger.warning("Rate limit hit, retrying in %ds …", EMBED_RETRY_DELAY)
                time.sleep(EMBED_RETRY_DELAY)
            else:
                raise


def _to_search_doc(course: dict, vector: list[float], stats: dict | None) -> dict:
    """Convert a course dict + embedding to an AI Search document."""
    doc = {
        "course_id":       course["course_id"],
        "title":           course["title"],
        "description":     course.get("description", ""),
        "department":      course["department"],
        "skill_level":     course["skill_level"],
        "format":          course["format"],
        "duration_hours":  float(course.get("duration_hours") or 0),
        "skills_taught":   course.get("skills_taught") or [],
        "num_modules":     int(course.get("num_modules") or 0),
        "completion_rate": float(stats["completion_rate_pct"]) if stats and stats.get("completion_rate_pct") is not None else None,
        "avg_score":       float(stats["avg_score"]) if stats and stats.get("avg_score") is not None else None,
        "content_vector":  vector,
    }
    return doc


def index_courses(
    courses: list[dict],
    openai_client: "OpenAI",
    embedding_model: str,
    search_endpoint: str,
    search_key: str,
    index_name: str,
    analytics: "FabricAnalytics | None" = None,
) -> int:
    """
    Embed all courses and upload to Azure AI Search.

    Parameters
    ----------
    courses         : list of course dicts (from seed_courses.json)
    openai_client   : authenticated OpenAI client from AIProjectClient.get_openai_client()
    embedding_model : deployment name, e.g. 'text-embedding-3-large'
    analytics       : optional FabricAnalytics for live completion_rate / avg_score enrichment
    Returns         : number of documents successfully indexed
    """
    search_client = SearchClient(
        search_endpoint, index_name, AzureKeyCredential(search_key)
    )

    total_indexed = 0

    for batch_start in range(0, len(courses), BATCH_SIZE):
        batch = courses[batch_start: batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        logger.info(
            "Batch %d/%d — embedding %d courses …",
            batch_num, (len(courses) + BATCH_SIZE - 1) // BATCH_SIZE, len(batch),
        )

        # Embed
        texts = [_build_embed_text(c) for c in batch]
        vectors = embed_texts(texts, openai_client, embedding_model)

        # Build search documents
        docs = []
        for course, vector in zip(batch, vectors):
            stats = None
            if analytics:
                try:
                    stats = analytics.get_course_stats(course["course_id"])
                except Exception:
                    pass
            docs.append(_to_search_doc(course, vector, stats))

        # Upload batch
        result = search_client.upload_documents(documents=docs)
        succeeded = sum(1 for r in result if r.succeeded)
        failed = len(result) - succeeded
        if failed:
            logger.warning("Batch %d: %d failed uploads.", batch_num, failed)
        total_indexed += succeeded
        logger.info("Batch %d: %d/%d uploaded.", batch_num, succeeded, len(batch))

    logger.info("Indexing complete. %d/%d courses indexed.", total_indexed, len(courses))
    return total_indexed
