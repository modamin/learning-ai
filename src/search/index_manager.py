"""
Sprint 3 — Task 3.1: Azure AI Search index schema for the course catalog.

Creates (or updates) the 'course-catalog' index with:
  - Full-text fields: title, description, skills_taught
  - Filter/facet fields: department, skill_level, format
  - Numeric fields: duration_hours, completion_rate, avg_score
  - Vector field: content_vector (3072-dim, text-embedding-3-large)
  - Semantic configuration: title + description prioritised
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

logger = logging.getLogger(__name__)

INDEX_NAME = os.environ.get("AZURE_SEARCH_INDEX_NAME", "course-catalog")


def _build_index(name: str) -> SearchIndex:
    fields = [
        # ── Key ──────────────────────────────────────────────────────────────
        SimpleField(
            name="course_id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),
        # ── Full-text searchable ──────────────────────────────────────────────
        SearchField(
            name="title",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            sortable=True,
        ),
        SearchField(
            name="description",
            type=SearchFieldDataType.String,
            searchable=True,
        ),
        SearchField(
            name="skills_taught",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            searchable=True,
            filterable=True,
            facetable=True,
        ),
        # ── Filter / facet ────────────────────────────────────────────────────
        SimpleField(
            name="department",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="skill_level",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        SimpleField(
            name="format",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),
        # ── Numeric ───────────────────────────────────────────────────────────
        SimpleField(
            name="duration_hours",
            type=SearchFieldDataType.Double,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="completion_rate",
            type=SearchFieldDataType.Double,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="avg_score",
            type=SearchFieldDataType.Double,
            sortable=True,
        ),
        SimpleField(
            name="num_modules",
            type=SearchFieldDataType.Int32,
            filterable=True,
        ),
        # ── Vector ────────────────────────────────────────────────────────────
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=3072,          # text-embedding-3-large
            vector_search_profile_name="hnsw-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-config",
                parameters={"m": 4, "efConstruction": 400, "efSearch": 500, "metric": "cosine"},
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw-config",
            )
        ],
    )

    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="title"),
                    content_fields=[SemanticField(field_name="description")],
                    keywords_fields=[SemanticField(field_name="skills_taught")],
                ),
            )
        ]
    )

    return SearchIndex(
        name=name,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )


def create_or_update_index(
    endpoint: str | None = None,
    key: str | None = None,
    index_name: str | None = None,
) -> SearchIndex:
    """
    Create or update the course-catalog index in Azure AI Search.

    Falls back to env vars if endpoint/key/index_name not provided.
    """
    endpoint = endpoint or os.environ["AZURE_SEARCH_ENDPOINT"]
    key = key or os.environ["AZURE_SEARCH_ADMIN_KEY"]
    index_name = index_name or INDEX_NAME

    client = SearchIndexClient(endpoint, AzureKeyCredential(key))
    index = _build_index(index_name)
    result = client.create_or_update_index(index)
    logger.info("Index '%s' ready (%d fields).", result.name, len(result.fields))
    return result


def delete_index(
    endpoint: str | None = None,
    key: str | None = None,
    index_name: str | None = None,
) -> None:
    endpoint = endpoint or os.environ["AZURE_SEARCH_ENDPOINT"]
    key = key or os.environ["AZURE_SEARCH_ADMIN_KEY"]
    index_name = index_name or INDEX_NAME
    client = SearchIndexClient(endpoint, AzureKeyCredential(key))
    client.delete_index(index_name)
    logger.info("Index '%s' deleted.", index_name)


def get_index_stats(
    endpoint: str | None = None,
    key: str | None = None,
    index_name: str | None = None,
) -> dict:
    endpoint = endpoint or os.environ["AZURE_SEARCH_ENDPOINT"]
    key = key or os.environ["AZURE_SEARCH_ADMIN_KEY"]
    index_name = index_name or INDEX_NAME
    client = SearchIndexClient(endpoint, AzureKeyCredential(key))
    stats = client.get_index_statistics(index_name)
    return {"document_count": stats.document_count, "storage_size": stats.storage_size}
