"""Central configuration — loads from .env and exposes typed settings."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root (works regardless of where script is invoked from)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _require(key: str) -> str:
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Copy .env.example to .env and fill in your values."
        )
    return value


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default)


# ── Foundry ───────────────────────────────────────────────────────────────────
FOUNDRY_PROJECT_ENDPOINT: str = _require("FOUNDRY_PROJECT_ENDPOINT")
MODEL_DEPLOYMENT_NAME: str = _optional("MODEL_DEPLOYMENT_NAME", "gpt-4o-mini")
EMBEDDING_DEPLOYMENT_NAME: str = _optional("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large")

# ── Azure AI Search ───────────────────────────────────────────────────────────
AZURE_SEARCH_ENDPOINT: str = _require("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_ADMIN_KEY: str = _require("AZURE_SEARCH_ADMIN_KEY")
AZURE_SEARCH_INDEX_NAME: str = _optional("AZURE_SEARCH_INDEX_NAME", "course-catalog")

# ── Fabric / OneLake ──────────────────────────────────────────────────────────
FABRIC_WORKSPACE_ID: str = _require("FABRIC_WORKSPACE_ID")
FABRIC_LAKEHOUSE_ID: str = _require("FABRIC_LAKEHOUSE_ID")
FABRIC_SQL_ENDPOINT: str = _require("FABRIC_SQL_ENDPOINT")
FABRIC_LAKEHOUSE_NAME: str = _optional("FABRIC_LAKEHOUSE_NAME", "learning_lakehouse")

ONELAKE_ACCOUNT_URL: str = "https://onelake.dfs.fabric.microsoft.com"

# ── Data paths ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def fabric_odbc_connection_string() -> str:
    """Build the pyodbc connection string for the Fabric SQL analytics endpoint."""
    return (
        f"Driver={{ODBC Driver 18 for SQL Server}};"
        f"Server={FABRIC_SQL_ENDPOINT},1433;"
        f"Database={FABRIC_LAKEHOUSE_NAME};"
        f"Authentication=ActiveDirectoryInteractive;"
        f"Encrypt=yes;TrustServerCertificate=no;"
    )
