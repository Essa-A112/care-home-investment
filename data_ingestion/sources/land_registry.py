"""
Land Registry Price Paid Data downloader.

Downloads yearly CSV files from the Land Registry public S3 bucket.
Skips files that are already present on disk. Probes for newer years
beyond the configured base end-year so the pipeline automatically
advances when fresh data is published.

Public data index:
  https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads
"""

import logging
from datetime import datetime
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# Land Registry public S3 endpoint for full-year Price Paid CSV files
_BASE_URL = (
    "http://prod.publicdata.landregistry.gov.uk"
    ".s3-website-eu-west-1.amazonaws.com"
)

START_YEAR = 2020
_BASE_END_YEAR = 2024

# Files smaller than this are treated as incomplete / unavailable.
# A full year's Price Paid CSV is typically 500 MB – 1 GB.
_MIN_BYTES_FOR_COMPLETE_YEAR = 50 * 1024 * 1024  # 50 MB


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _year_url(year: int) -> str:
    return f"{_BASE_URL}/pp-{year}.csv"


def _raw_path(year: int, raw_dir: Path) -> Path:
    return raw_dir / f"pp-{year}.csv"


def _probe_year(year: int) -> bool:
    """Return True if Land Registry has published usable data for *year*."""
    url = _year_url(year)
    try:
        resp = httpx.head(url, follow_redirects=True, timeout=20)
        if resp.status_code != 200:
            logger.debug("HEAD %s → HTTP %s", url, resp.status_code)
            return False
        content_length = int(resp.headers.get("content-length", 0))
        available = content_length >= _MIN_BYTES_FOR_COMPLETE_YEAR
        logger.debug(
            "HEAD %s → %s bytes (%s)",
            url,
            content_length,
            "available" if available else "too small",
        )
        return available
    except httpx.RequestError as exc:
        logger.warning("Could not probe %d data: %s", year, exc)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_end_year() -> int:
    """
    Probe Land Registry for years beyond the base end-year.

    Iterates from _BASE_END_YEAR + 1 up to (and including) the current
    calendar year. Returns the highest year whose file passes the size
    threshold, so the pipeline automatically picks up newly released data.
    """
    current_year = datetime.now().year
    end_year = _BASE_END_YEAR

    for candidate in range(_BASE_END_YEAR + 1, current_year + 1):
        if _probe_year(candidate):
            logger.info("Newer Land Registry data detected: %d", candidate)
            end_year = candidate
        else:
            # Stop at the first year with no usable file rather than
            # skipping a gap (Land Registry publishes sequentially).
            break

    if end_year == _BASE_END_YEAR:
        logger.info("No data found beyond %d; using base end-year.", _BASE_END_YEAR)

    return end_year


def download_year(year: int, raw_dir: Path) -> Path:
    """
    Download one year's Price Paid CSV into *raw_dir*.

    Skips the download if the file already exists on disk (idempotent).
    Streams the response to avoid loading the whole file into memory.

    Returns the local path to the downloaded (or pre-existing) file.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = _raw_path(year, raw_dir)

    if dest.exists():
        size_mb = dest.stat().st_size / 1e6
        logger.info("%s already exists (%.1f MB) – skipping download.", dest.name, size_mb)
        return dest

    url = _year_url(year)
    logger.info("Downloading %s → %s", url, dest)

    tmp = dest.with_suffix(".tmp")
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=600) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=65_536):
                    fh.write(chunk)
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise

    logger.info("Saved %s (%.1f MB).", dest.name, dest.stat().st_size / 1e6)
    return dest


def download_all(raw_dir: Path, end_year: int | None = None) -> dict[int, Path]:
    """
    Download all years from START_YEAR to *end_year* (inclusive).

    If *end_year* is None, calls detect_end_year() first.
    Returns a mapping of {year: local_path}.
    """
    if end_year is None:
        end_year = detect_end_year()

    logger.info("Downloading Price Paid Data %d–%d.", START_YEAR, end_year)
    paths: dict[int, Path] = {}
    for year in range(START_YEAR, end_year + 1):
        paths[year] = download_year(year, raw_dir)

    return paths
