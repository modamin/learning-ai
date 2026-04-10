"""
OneLake ADLS Gen2 client.

Wraps azure-storage-file-datalake to upload/download files to the
Fabric Lakehouse Files section via:
  https://onelake.dfs.fabric.microsoft.com/<workspace_id>/<lakehouse_id>/Files/...
"""
from __future__ import annotations

import logging
from pathlib import Path

from azure.core.exceptions import ResourceExistsError
from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient, FileSystemClient

logger = logging.getLogger(__name__)


class OneLakeClient:
    """
    Upload/download files to a Fabric Lakehouse via the OneLake ADLS Gen2 API.

    Parameters
    ----------
    workspace_id : str
        Fabric workspace GUID (FABRIC_WORKSPACE_ID).
    lakehouse_id : str
        Fabric lakehouse GUID (FABRIC_LAKEHOUSE_ID).
    credential : optional
        Any azure-identity credential. Defaults to DefaultAzureCredential.
    account_url : str
        OneLake ADLS Gen2 endpoint. Defaults to the public endpoint.
    """

    ONELAKE_URL = "https://onelake.dfs.fabric.microsoft.com"

    def __init__(
        self,
        workspace_id: str,
        lakehouse_id: str,
        credential=None,
        account_url: str | None = None,
    ) -> None:
        self.workspace_id = workspace_id
        self.lakehouse_id = lakehouse_id
        self._credential = credential or DefaultAzureCredential()
        self._account_url = account_url or self.ONELAKE_URL

        self._service: DataLakeServiceClient = DataLakeServiceClient(
            account_url=self._account_url,
            credential=self._credential,
        )
        # Each Fabric workspace maps to one ADLS Gen2 file system
        self._fs: FileSystemClient = self._service.get_file_system_client(
            file_system=self.workspace_id
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_directory(self, path: str) -> None:
        """Create a directory path in OneLake (no-op if it already exists)."""
        dir_client = self._fs.get_directory_client(path)
        try:
            dir_client.create_directory()
            logger.debug("Created directory: %s", path)
        except ResourceExistsError:
            pass

    def _lakehouse_path(self, remote_path: str) -> str:
        """Prefix remote_path with the lakehouse ID so files land in the right lakehouse."""
        remote_path = remote_path.lstrip("/")
        return f"{self.lakehouse_id}/{remote_path}"

    # ── Public API ────────────────────────────────────────────────────────────

    def upload_file(
        self,
        local_path: str | Path,
        remote_path: str,
        overwrite: bool = True,
        chunk_size: int = 4 * 1024 * 1024,  # 4 MB
    ) -> str:
        """
        Upload a local file to OneLake.

        Parameters
        ----------
        local_path : str | Path
            Local file to upload.
        remote_path : str
            Destination path inside the lakehouse, e.g. "Files/raw/events.json".
        overwrite : bool
            Whether to overwrite an existing file (default True).

        Returns
        -------
        str
            The full OneLake path of the uploaded file.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        full_path = self._lakehouse_path(remote_path)
        parent_dir = str(Path(full_path).parent)
        self._ensure_directory(parent_dir)

        file_client = self._fs.get_file_client(full_path)
        file_size = local_path.stat().st_size
        logger.info("Uploading %s → %s (%.1f MB)", local_path.name, full_path, file_size / 1e6)

        with local_path.open("rb") as fh:
            file_client.upload_data(fh, overwrite=overwrite, chunk_size=chunk_size)

        logger.info("  Upload complete: %s", full_path)
        return full_path

    def upload_bytes(
        self,
        data: bytes,
        remote_path: str,
        overwrite: bool = True,
    ) -> str:
        """Upload raw bytes to OneLake."""
        full_path = self._lakehouse_path(remote_path)
        parent_dir = str(Path(full_path).parent)
        self._ensure_directory(parent_dir)

        file_client = self._fs.get_file_client(full_path)
        file_client.upload_data(data, overwrite=overwrite)
        return full_path

    def download_file(
        self,
        remote_path: str,
        local_path: str | Path,
    ) -> Path:
        """
        Download a file from OneLake to a local path.

        Returns
        -------
        Path
            Resolved local path of the downloaded file.
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        full_path = self._lakehouse_path(remote_path)
        file_client = self._fs.get_file_client(full_path)

        logger.info("Downloading %s → %s", full_path, local_path)
        download = file_client.download_file()
        with local_path.open("wb") as fh:
            download.readinto(fh)
        return local_path

    def list_files(self, directory: str = "Files") -> list[str]:
        """
        List files under a directory in the lakehouse.

        Parameters
        ----------
        directory : str
            Path inside the lakehouse, e.g. "Files/raw".
        """
        prefix = self._lakehouse_path(directory)
        paths = self._fs.get_paths(path=prefix, recursive=True)
        return [p.name for p in paths if not p.is_directory]

    def file_exists(self, remote_path: str) -> bool:
        """Return True if the remote file exists in OneLake."""
        full_path = self._lakehouse_path(remote_path)
        try:
            self._fs.get_file_client(full_path).get_file_properties()
            return True
        except Exception:
            return False
