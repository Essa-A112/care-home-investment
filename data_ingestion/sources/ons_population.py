"""
ONS Mid-Year Population Estimates downloader.

Downloads the ONS "Population Estimates for UK, England and Wales, Scotland
and Northern Ireland" Excel file for the latest available mid-year period.
The file contains population by Local Authority District and single year of
age, from which we extract total population and population aged 65 and over.

Publication page:
  https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/
  populationestimates/datasets/
  populationestimatesforukenglandandwalesscotlandandnorthernireland

Refresh logic:
  Probes candidate years from (current_year - 1) downwards. Stores the
  accepted year in a small JSON sidecar file so subsequent runs can skip
  the network probe when nothing has changed.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# URL template for ONS full-year Excel release.
# ONS keeps the filename stable across geography vintages within a year.
_URL_TEMPLATE = (
    "https://www.ons.gov.uk/file?uri="
    "/peoplepopulationandcommunity/populationandmigration/populationestimates"
    "/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland"
    "/mid{year}/ukpopestimatesmid{year}on{year}geographyfinal.xlsx"
)

# Minimum plausible file size – the Excel is typically > 5 MB
_MIN_BYTES = 1 * 1024 * 1024  # 1 MB

_META_FILENAME = "ons_population_meta.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _meta_path(raw_dir: Path) -> Path:
    return raw_dir / _META_FILENAME


def _excel_path(year: int, raw_dir: Path) -> Path:
    return raw_dir / f"ons_mye_{year}.xlsx"


def _probe_year(year: int) -> bool:
    """Return True if the ONS Excel file for *year* is available and non-trivial."""
    url = _URL_TEMPLATE.format(year=year)
    try:
        resp = httpx.head(url, follow_redirects=True, timeout=30)
        if resp.status_code != 200:
            logger.debug("ONS MYE HEAD %s → HTTP %s", year, resp.status_code)
            return False
        content_length = int(resp.headers.get("content-length", 0))
        ok = content_length >= _MIN_BYTES
        logger.debug("ONS MYE HEAD %s → %s bytes (%s)", year, content_length, "ok" if ok else "too small")
        return ok
    except httpx.RequestError as exc:
        logger.warning("Could not probe ONS MYE year %d: %s", year, exc)
        return False


def _load_meta(raw_dir: Path) -> dict:
    p = _meta_path(raw_dir)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_meta(raw_dir: Path, meta: dict) -> None:
    _meta_path(raw_dir).write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_latest_mye_year(raw_dir: Path) -> int:
    """
    Return the most recent mid-year estimate year available from ONS.

    Starts from (current_calendar_year - 1) and counts downwards up to
    three years back. ONS typically publishes mid-year estimates with a
    ~12-month lag, so current_year - 1 is usually the latest.

    If a cached year is found in the metadata sidecar and the cached year
    equals (current_year - 1), assumes no new release and skips network
    probing. Otherwise re-probes to detect a new release.
    """
    current_year = datetime.now().year
    expected_latest = current_year - 1

    meta = _load_meta(raw_dir)
    cached_year = meta.get("year")

    if cached_year == expected_latest:
        logger.info("ONS MYE metadata current (year=%d); skipping probe.", cached_year)
        return cached_year

    # Probe most-recent first, stop at the first hit
    for candidate in range(expected_latest, expected_latest - 3, -1):
        if _probe_year(candidate):
            logger.info("ONS MYE latest year detected: %d", candidate)
            meta["year"] = candidate
            raw_dir.mkdir(parents=True, exist_ok=True)
            _save_meta(raw_dir, meta)
            return candidate

    # If all probes fail, fall back to the cached year or a safe default
    fallback = cached_year or (expected_latest - 1)
    logger.warning(
        "ONS MYE probes failed; using fallback year %d.", fallback
    )
    return fallback


def download_mye(raw_dir: Path, year: int | None = None) -> Path:
    """
    Download the ONS mid-year population estimates Excel file.

    Skips the download if the file is already present on disk.
    Returns the local path to the Excel file.
    """
    if year is None:
        year = detect_latest_mye_year(raw_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = _excel_path(year, raw_dir)

    if dest.exists():
        logger.info("ONS MYE %d already on disk (%s) – skipping.", year, dest.name)
        return dest

    url = _URL_TEMPLATE.format(year=year)
    logger.info("Downloading ONS MYE %d from %s", year, url)

    tmp = dest.with_suffix(".tmp")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=300) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=65_536):
                    fh.write(chunk)
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    logger.info("Saved ONS MYE %d → %s (%.1f MB).", year, dest.name, dest.stat().st_size / 1e6)
    return dest
