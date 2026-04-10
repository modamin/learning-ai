"""
Sprint 1 — Task 1.2: Upload seed data to Fabric Lakehouse via OneLake ADLS Gen2 API.

Uploads:
  data/seed_courses.json      → Files/raw/seed_courses.json
  data/seed_learners.json     → Files/raw/seed_learners.json
  data/seed_xapi_events.json  → Files/raw/seed_xapi_events.json

Auth: DefaultAzureCredential (picks up 'az login' or service principal env vars).

Usage:
  python scripts/02_upload_to_fabric.py
  python scripts/02_upload_to_fabric.py --dry-run    # validate files exist, skip upload
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running from repo root or scripts/ directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402 — registers env vars as side-effect
from src.fabric.onelake import OneLakeClient  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Files to upload: (local relative path, remote path inside lakehouse)
UPLOAD_MANIFEST: list[tuple[str, str]] = [
    ("data/seed_courses.json", "Files/raw/seed_courses.json"),
    ("data/seed_learners.json", "Files/raw/seed_learners.json"),
    ("data/seed_xapi_events.json", "Files/raw/seed_xapi_events.json"),
]


def _check_data_files(repo_root: Path) -> bool:
    """Verify all local data files exist before attempting upload."""
    all_ok = True
    for local_rel, _ in UPLOAD_MANIFEST:
        p = repo_root / local_rel
        if p.exists():
            size_mb = p.stat().st_size / 1e6
            logger.info("  ✓ Found %-40s (%.1f MB)", local_rel, size_mb)
        else:
            logger.error("  ✗ Missing %s — run scripts/01_generate_synthetic_data.py first", local_rel)
            all_ok = False
    return all_ok


def upload_all(dry_run: bool = False) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    logger.info("=== Fabric OneLake Upload ===")
    logger.info("Workspace: %s", config.FABRIC_WORKSPACE_ID)
    logger.info("Lakehouse: %s", config.FABRIC_LAKEHOUSE_ID)
    logger.info("")

    logger.info("Checking local data files …")
    if not _check_data_files(repo_root):
        sys.exit(1)

    if dry_run:
        logger.info("\n[DRY RUN] Skipping upload. All files present.")
        return

    logger.info("\nInitialising OneLake client …")
    client = OneLakeClient(
        workspace_id=config.FABRIC_WORKSPACE_ID,
        lakehouse_id=config.FABRIC_LAKEHOUSE_ID,
    )

    results: list[dict] = []
    total_bytes = 0
    t_start = time.monotonic()

    for local_rel, remote_path in UPLOAD_MANIFEST:
        local_path = repo_root / local_rel
        file_size = local_path.stat().st_size
        total_bytes += file_size

        logger.info("Uploading %-40s → %s", local_rel, remote_path)
        t0 = time.monotonic()
        try:
            full_path = client.upload_file(local_path, remote_path, overwrite=True)
            elapsed = time.monotonic() - t0
            throughput_mbps = (file_size / 1e6) / max(elapsed, 0.001)
            logger.info(
                "  ✓ Done in %.1fs  (%.1f MB/s)  →  %s",
                elapsed, throughput_mbps, full_path,
            )
            results.append({"file": local_rel, "status": "ok", "path": full_path})
        except Exception as exc:
            logger.error("  ✗ Failed: %s", exc)
            results.append({"file": local_rel, "status": "error", "error": str(exc)})

    elapsed_total = time.monotonic() - t_start
    logger.info("\n=== Summary ===")
    logger.info("Total data uploaded: %.1f MB in %.1fs", total_bytes / 1e6, elapsed_total)
    for r in results:
        status_icon = "✓" if r["status"] == "ok" else "✗"
        logger.info("  %s %s", status_icon, r["file"])

    errors = [r for r in results if r["status"] == "error"]
    if errors:
        logger.error("%d upload(s) failed.", len(errors))
        sys.exit(1)

    # Verify files are visible in OneLake
    logger.info("\nVerifying files in OneLake …")
    remote_files = client.list_files("Files/raw")
    logger.info("Files found under Files/raw:")
    for f in remote_files:
        logger.info("  %s", f)

    logger.info("\n✓ All files uploaded successfully.")
    logger.info("Next step: run notebooks/01_xapi_ingestion.ipynb in Fabric to create Delta tables.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload seed data to Fabric Lakehouse via OneLake.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local files exist but skip the actual upload.",
    )
    args = parser.parse_args()
    upload_all(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
