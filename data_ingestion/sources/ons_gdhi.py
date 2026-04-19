"""
ONS Gross Disposable Household Income (GDHI) by Local Authority downloader.

Downloads the ONS GDHI dataset which provides GDHI per head of population
at local authority district level for England. This is published annually
as an Excel workbook.

Publication page:
  https://www.ons.gov.uk/economy/regionalaccounts/grossdisposablehouseholdincome/
  datasets/regionalgrossdisposablehouseholdincomegdhibylocalauthorityunitedkingdom

Refresh logic:
  Performs a HEAD request to the ONS file URL and compares the
  Last-Modified response header against the value stored in a JSON
  metadata sidecar. Re-downloads only when the server reports a newer
  modification date.
"""

import json
import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# ONS uses a stable `/current/` redirect that always points to the latest
# published version of the dataset.
_GDHI_URL = (
    "https://www.ons.gov.uk/file?uri="
    "/economy/regionalaccounts/grossdisposablehouseholdincome"
    "/datasets/regionalgrossdisposablehouseholdincomegdhibylocalauthorityunitedkingdom"
    "/current/regionalgrossdisposablehouseholdincomegdhibylocalauthority.xlsx"
)

_FILENAME = "ons_gdhi_lad.xlsx"
_META_FILENAME = "ons_gdhi_meta.json"
_MIN_BYTES = 200 * 1024  # 200 KB – plausible minimum for the Excel file


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dest_path(raw_dir: Path) -> Path:
    return raw_dir / _FILENAME


def _meta_path(raw_dir: Path) -> Path:
    return raw_dir / _META_FILENAME


def _load_meta(raw_dir: Path) -> dict:
    p = _meta_path(raw_dir)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_meta(raw_dir: Path, meta: dict) -> None:
    _meta_path(raw_dir).write_text(json.dumps(meta, indent=2))


def _server_last_modified(client: httpx.Client) -> str | None:
    """Return the Last-Modified header value from the ONS file server."""
    try:
        resp = client.head(_GDHI_URL, follow_redirects=True, timeout=30)
        if resp.status_code == 200:
            lm = resp.headers.get("last-modified") or resp.headers.get("Last-Modified")
            logger.debug("ONS GDHI Last-Modified: %s", lm)
            return lm
    except httpx.RequestError as exc:
        logger.warning("ONS GDHI HEAD probe failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_gdhi(raw_dir: Path) -> Path:
    """
    Download the ONS GDHI per-head by LAD Excel file.

    Skips the download when the server's Last-Modified header matches
    the previously cached value, indicating no update has been published.

    Returns the local path to the Excel file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = _dest_path(raw_dir)
    meta = _load_meta(raw_dir)

    with httpx.Client(follow_redirects=True) as client:
        server_lm = _server_last_modified(client)

        if dest.exists() and server_lm and server_lm == meta.get("last_modified"):
            logger.info("ONS GDHI file unchanged (Last-Modified: %s); skipping.", server_lm)
            return dest

        if dest.exists() and server_lm is None:
            logger.warning(
                "ONS GDHI: could not determine server modification date; "
                "using cached file."
            )
            return dest

        logger.info("Downloading ONS GDHI Excel from ONS.")
        tmp = dest.with_suffix(".tmp")
        try:
            with client.stream("GET", _GDHI_URL, timeout=120) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=65_536):
                        fh.write(chunk)
            tmp.rename(dest)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    size_kb = dest.stat().st_size / 1024
    logger.info("ONS GDHI saved: %s (%.0f KB).", dest.name, size_kb)

    if size_kb * 1024 < _MIN_BYTES:
        logger.warning(
            "ONS GDHI file is unexpectedly small (%.0f KB). "
            "Check that the URL still resolves correctly.",
            size_kb,
        )

    meta["last_modified"] = server_lm
    _save_meta(raw_dir, meta)
    return dest
