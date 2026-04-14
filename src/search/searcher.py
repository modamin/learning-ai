"""
Sprint 3 — Task 3.3: Hybrid search over the course catalog.

Supports three modes (auto-selected based on arguments):
  1. Pure vector search    — embed query, find nearest neighbours
  2. Hybrid               — vector + full-text keyword search (default)
  3. Semantic re-rank     — hybrid + Azure AI Search semantic ranker

Usage
-----
from src.search.searcher import CourseSearcher
searcher = CourseSearcher.from_env(openai_client)
results = searcher.search("Python for data engineers", department="Engineering", top=5)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from dotenv import load_dotenv

if TYPE_CHECKING:
    from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

logger = logging.getLogger(__name__)

# Fields returned in every search result (content_vector excluded — large)
_SELECT_FIELDS = [
    "course_id", "title", "description", "department",
    "skill_level", "format", "duration_hours", "skills_taught",
    "completion_rate", "avg_score", "num_modules",
]


class CourseSearcher:
    """
    Hybrid vector + keyword search over the AI Search course-catalog index.

    Parameters
    ----------
    search_client   : Azure AI Search SearchClient
    openai_client   : authenticated OpenAI client (for query embedding)
    embedding_model : deployment name (text-embedding-3-large)
    """

    def __init__(
        self,
        search_client: SearchClient,
        openai_client: "OpenAI",
        embedding_model: str,
    ) -> None:
        self._search = search_client
        self._openai = openai_client
        self._model = embedding_model

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls, openai_client: "OpenAI") -> "CourseSearcher":
        endpoint   = os.environ["AZURE_SEARCH_ENDPOINT"]
        key        = os.environ["AZURE_SEARCH_ADMIN_KEY"]
        index_name = os.environ.get("AZURE_SEARCH_INDEX_NAME", "course-catalog")
        model      = os.environ.get("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large")
        client     = SearchClient(endpoint, index_name, AzureKeyCredential(key))
        return cls(client, openai_client, model)

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        response = self._openai.embeddings.create(input=[text], model=self._model)
        return response.data[0].embedding

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        department: str | None = None,
        skill_level: str | None = None,
        format_filter: str | None = None,
        max_duration_hours: float | None = None,
        top: int = 5,
        semantic: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search: vector similarity + full-text keyword scoring.

        Parameters
        ----------
        query               : natural-language search query
        department          : OData filter e.g. "Engineering"
        skill_level         : "beginner" | "intermediate" | "advanced"
        format_filter       : "self-paced" | "blended" | "instructor-led"
        max_duration_hours  : upper bound filter
        top                 : number of results to return
        semantic            : apply semantic re-ranking (requires S1+ tier or semantic plan)

        Returns
        -------
        list of dicts, each with all _SELECT_FIELDS plus '@search.score'
        """
        # Build OData filter
        filters = []
        if department:
            filters.append(f"department eq '{department}'")
        if skill_level:
            filters.append(f"skill_level eq '{skill_level}'")
        if format_filter:
            filters.append(f"format eq '{format_filter}'")
        if max_duration_hours is not None:
            filters.append(f"duration_hours le {max_duration_hours}")
        odata_filter = " and ".join(filters) if filters else None

        # Embed query for vector search
        query_vector = self._embed(query)
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=top * 2,    # over-fetch, semantic reranks
            fields="content_vector",
        )

        # Search kwargs
        kwargs: dict[str, Any] = dict(
            search_text=query,              # full-text component
            vector_queries=[vector_query],  # vector component
            select=_SELECT_FIELDS,
            filter=odata_filter,
            top=top,
        )

        if semantic:
            kwargs["query_type"] = "semantic"
            kwargs["semantic_configuration_name"] = "semantic-config"

        results = self._search.search(**kwargs)
        return [dict(r) for r in results]

    def search_by_skills(
        self,
        skills: list[str],
        *,
        department: str | None = None,
        top: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Find courses that teach a given set of skills.
        Builds a query from the skill list and runs hybrid search.
        """
        query = "Courses teaching: " + ", ".join(skills)
        return self.search(query, department=department, top=top)

    def get_course(self, course_id: str) -> dict[str, Any] | None:
        """Fetch a single course document by ID."""
        try:
            return self._search.get_document(key=course_id, selected_fields=_SELECT_FIELDS)
        except Exception:
            return None
