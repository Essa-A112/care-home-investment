"""
CQC Care Home Directory downloader.

Uses the CQC public REST API v1 to download all registered care home
locations in England. The API returns paginated JSON. All pages are
fetched and written to a single Newline-Delimited JSON (NDJSON) file
so large datasets can be streamed during processing without holding the
full dataset in memory.

CQC API documentation:
  https://api.cqc.org.uk/public/v1

Refresh logic:
  On first run, downloads all pages and stores the total record count
  in a JSON metadata sidecar. On subsequent runs, queries page 1 to
  obtain the current API total; if it matches the cached count the
  existing file is used unchanged.
"""

import json
import logging
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_API_BASE = "https://api.cqc.org.uk/public/v1"
_LOCATIONS_ENDPOINT = f"{_API_BASE}/locations"
_PAGE_SIZE = 1_000
_REQUEST_DELAY_S = 0.2   # polite rate-limiting between pages
_META_FILENAME = "cqc_directory_meta.json"
_DATA_FILENAME = "cqc_care_homes.ndjson"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _meta_path(raw_dir: Path) -> Path:
    return raw_dir / _META_FILENAME


def _data_path(raw_dir: Path) -> Path:
    return raw_dir / _DATA_FILENAME


def _load_meta(raw_dir: Path) -> dict:
    p = _meta_path(raw_dir)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_meta(raw_dir: Path, meta: dict) -> None:
    _meta_path(raw_dir).write_text(json.dumps(meta, indent=2))


def _fetch_page(client: httpx.Client, page: int) -> dict:
    """Fetch a single page of care home locations from the CQC API."""
    resp = client.get(
        _LOCATIONS_ENDPOINT,
        params={
            "careHomeService": "yes",
            "page": page,
            "perPage": _PAGE_SIZE,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def _current_api_total(client: httpx.Client) -> int:
    """Return the total number of care home locations reported by the API."""
    data = _fetch_page(client, page=1)
    return data.get("total", 0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_cqc_directory(raw_dir: Path) -> Path:
    """
    Download all CQC-registered care home locations to an NDJSON file.

    Each line of the output file is a JSON object for one location.
    Refresh check: if the API's current total matches the cached total,
    the existing file is used without re-downloading.

    Returns the path to the NDJSON file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = _data_path(raw_dir)
    meta = _load_meta(raw_dir)

    with httpx.Client(headers={"User-Agent": "care-home-investment/1.0"}) as client:
        current_total = _current_api_total(client)
        logger.info("CQC API reports %d care home locations.", current_total)

        if dest.exists() and meta.get("total") == current_total:
            logger.info(
                "CQC directory up to date (total=%d); skipping download.", current_total
            )
            return dest

        # Compute number of pages
        first_page = _fetch_page(client, page=1)
        total_pages = first_page.get("totalPages", 1)
        logger.info(
            "Downloading CQC directory: %d locations across %d pages.",
            current_total,
            total_pages,
        )

        tmp = dest.with_suffix(".tmp")
        written = 0
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                # Write first page (already fetched)
                for loc in first_page.get("locations", []):
                    fh.write(json.dumps(loc) + "\n")
                    written += 1

                # Fetch remaining pages
                for page in range(2, total_pages + 1):
                    time.sleep(_REQUEST_DELAY_S)
                    data = _fetch_page(client, page)
                    for loc in data.get("locations", []):
                        fh.write(json.dumps(loc) + "\n")
                        written += 1

                    if page % 10 == 0:
                        logger.info(
                            "CQC download progress: page %d/%d (%d locations so far).",
                            page,
                            total_pages,
                            written,
                        )

            tmp.rename(dest)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    meta["total"] = current_total
    meta["written"] = written
    _save_meta(raw_dir, meta)

    logger.info("CQC directory saved: %d locations → %s.", written, dest.name)
    return dest
