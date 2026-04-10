"""
Thin pyodbc wrapper for the Fabric Lakehouse SQL analytics endpoint.

Authentication strategy
-----------------------
Service principal (when AZURE_CLIENT_ID + AZURE_CLIENT_SECRET + AZURE_TENANT_ID are set):
  Uses azure-identity ClientSecretCredential to fetch an AAD bearer token for
  https://database.windows.net/, then passes it to pyodbc via the
  SQL_COPT_SS_ACCESS_TOKEN pre-connect attribute (attribute ID 1256).
  This is the most reliable path for Fabric SQL with a service principal and
  avoids the ODBC driver's own Authentication=ActiveDirectoryServicePrincipal
  flow, which requires additional driver-level MSAL configuration.

Interactive (local dev, no service principal):
  Uses Authentication=ActiveDirectoryInteractive (browser/device-code pop-up).
  Skipped when CI=true or NO_INTERACTIVE=true.

Default credential (CI / managed identity):
  Falls back to Authentication=ActiveDirectoryDefault.

.env is loaded at module-import time so callers (Streamlit, scripts) don't
need to import src.config before env vars are populated.
"""
from __future__ import annotations

import logging
import os
import struct
from pathlib import Path
from typing import Any
from contextlib import contextmanager

import pandas as pd
import pyodbc
from dotenv import load_dotenv

# Scope required for Fabric / Azure SQL
_SQL_TOKEN_SCOPE = "https://database.windows.net/.default"
# pyodbc pre-connect attribute ID for passing a raw AAD access token
_SQL_COPT_SS_ACCESS_TOKEN = 1256

# Load .env from repo root on module import
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

logger = logging.getLogger(__name__)


# ── Driver detection ──────────────────────────────────────────────────────────

_PREFERRED_DRIVERS = ["ODBC Driver 18 for SQL Server", "ODBC Driver 17 for SQL Server"]


def _detect_driver() -> str:
    installed = pyodbc.drivers()
    for driver in _PREFERRED_DRIVERS:
        if driver in installed:
            return driver
    raise RuntimeError(
        "No supported ODBC driver found for Fabric SQL.\n"
        "Installed: " + ", ".join(installed) + "\n\n"
        "Install with:\n"
        "  winget install Microsoft.msodbcsql.18"
    )


# ── Token helper ──────────────────────────────────────────────────────────────

def _get_sp_token_bytes(tenant_id: str, client_id: str, client_secret: str) -> bytes:
    """
    Acquire an AAD access token for the SQL scope and return it encoded as
    the byte structure expected by SQL_COPT_SS_ACCESS_TOKEN:
      4-byte little-endian length prefix  +  UTF-16-LE encoded token string
    """
    from azure.identity import ClientSecretCredential  # lazy import

    credential = ClientSecretCredential(
        tenant_id=tenant_id,
        client_id=client_id,
        client_secret=client_secret,
    )
    token = credential.get_token(_SQL_TOKEN_SCOPE).token
    token_bytes = token.encode("utf-16-le")
    return struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)


# ── Connection string builders ────────────────────────────────────────────────

def _base_conn_str(endpoint: str, database: str, driver: str) -> str:
    return (
        f"Driver={{{driver}}};"
        f"Server={endpoint},1433;"
        f"Database={database};"
        f"Encrypt=yes;"
        f"TrustServerCertificate=no;"
        f"Connection Timeout=30;"
    )


# ── Client ────────────────────────────────────────────────────────────────────

class FabricSQLClient:
    """
    Manages a single pyodbc connection to the Fabric Lakehouse SQL analytics endpoint.

    Usage
    -----
    client = FabricSQLClient.from_env()
    df = client.query("SELECT * FROM gold_course_completion")
    """

    def __init__(
        self,
        conn_str: str,
        token_bytes: bytes | None = None,
    ) -> None:
        self._conn_str = conn_str
        self._token_bytes = token_bytes   # None → use conn_str-embedded auth
        self._conn: pyodbc.Connection | None = None

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "FabricSQLClient":
        """
        Build from environment variables.

        Required
        --------
        FABRIC_SQL_ENDPOINT     — <guid>.datawarehouse.fabric.microsoft.com
        FABRIC_LAKEHOUSE_NAME   — lakehouse database name

        Service principal (recommended)
        --------------------------------
        AZURE_TENANT_ID
        AZURE_CLIENT_ID
        AZURE_CLIENT_SECRET
        """
        endpoint = os.environ["FABRIC_SQL_ENDPOINT"]
        database = os.environ.get("FABRIC_LAKEHOUSE_NAME", "learning_lakehouse")
        driver   = _detect_driver()
        base     = _base_conn_str(endpoint, database, driver)

        tenant_id     = os.environ.get("AZURE_TENANT_ID")
        client_id     = os.environ.get("AZURE_CLIENT_ID")
        client_secret = os.environ.get("AZURE_CLIENT_SECRET")
        is_ci         = bool(os.environ.get("CI") or os.environ.get("NO_INTERACTIVE"))

        has_sp = bool(tenant_id and client_id and client_secret)

        if has_sp:
            logger.info(
                "Fabric SQL auth: service principal token  tenant=%s  client=%s  db=%s",
                tenant_id, client_id, database,
            )
            token_bytes = _get_sp_token_bytes(tenant_id, client_id, client_secret)
            return cls(base, token_bytes=token_bytes)

        if not is_ci:
            logger.info("Fabric SQL auth: interactive  db=%s", database)
            return cls(base + "Authentication=ActiveDirectoryInteractive;")

        logger.info("Fabric SQL auth: default credential  db=%s", database)
        return cls(base + "Authentication=ActiveDirectoryDefault;")

    # ── Connection lifecycle ──────────────────────────────────────────────────

    def _open(self) -> pyodbc.Connection:
        if self._token_bytes is not None:
            return pyodbc.connect(
                self._conn_str,
                attrs_before={_SQL_COPT_SS_ACCESS_TOKEN: self._token_bytes},
                autocommit=True,
            )
        return pyodbc.connect(self._conn_str, autocommit=True)

    def connect(self) -> None:
        if self._conn is None or self._is_closed():
            logger.info("Opening Fabric SQL connection ...")
            try:
                self._conn = self._open()
                logger.info("Connected to Fabric SQL analytics endpoint.")
            except pyodbc.Error as exc:
                logger.error("Failed to connect: %s", exc)
                raise

    def _is_closed(self) -> bool:
        try:
            if self._conn is None:
                return True
            self._conn.cursor().execute("SELECT 1")
            return False
        except Exception:
            return True

    def close(self) -> None:
        if self._conn and not self._is_closed():
            self._conn.close()
            self._conn = None

    @contextmanager
    def cursor(self):
        self.connect()
        cur = self._conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

    # ── Query helpers ─────────────────────────────────────────────────────────

    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        with self.cursor() as cur:
            cur.execute(sql, params or [])
            cols = [col[0] for col in cur.description]
            rows = cur.fetchall()
        return pd.DataFrame.from_records(rows, columns=cols)

    def query_one(self, sql: str, params: list | None = None) -> dict[str, Any] | None:
        df = self.query(sql, params)
        if df.empty:
            return None
        return df.iloc[0].to_dict()
