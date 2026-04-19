"""
ONS Local Authority District (LAD) code lookup downloader.

Fetches a lightweight name → code reference table from the ONS Open
Geography Portal ArcGIS REST service so that Price Paid Data district
names can be mapped to official nine-character LAD codes (e.g. E06000001).

The service returns ~374 rows for the full UK; England-only rows start
with 'E'.

Multiple candidate service URLs are tried in order so the lookup
degrades gracefully if ONS updates a service name between boundary
releases. All queries use f=json (universally supported) rather than
f=csv (which some services reject).
"""

import logging
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

_CACHE_FILENAME = "ons_lad_lookup.csv"

# Each entry: (query URL, code field name, name field name).
# Tried newest-first; first successful response wins.
# Service names verified against the ONS ArcGIS catalog at:
#   https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services?f=json
_CANDIDATES = [
    (
        "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services"
        "/LAD_APR_2025_UK_NC_v2/FeatureServer/0/query",
        "LAD25CD", "LAD25NM",
    ),
    (
        "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services"
        "/LAD_DEC_2024_UK_NC/FeatureServer/0/query",
        "LAD24CD", "LAD24NM",
    ),
    (
        "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services"
        "/LAD_APR_2023_UK_NC/FeatureServer/0/query",
        "LAD23CD", "LAD23NM",
    ),
]

_BASE_PARAMS = {
    "where": "1=1",
    "returnGeometry": "false",
    "f": "json",
}


def _try_candidate(url: str, code_field: str, name_field: str) -> pd.DataFrame | None:
    """
    Attempt one ArcGIS FeatureServer query.

    Returns a two-column DataFrame on success, None on HTTP error or
    missing fields. Uses JSON response to avoid the CSV-format 400 errors
    some ONS services return.
    """
    params = {**_BASE_PARAMS, "outFields": f"{code_field},{name_field}"}
    try:
        resp = httpx.get(url, params=params, timeout=60, follow_redirects=True)
        if resp.status_code != 200:
            logger.debug("ArcGIS %s → HTTP %s", url, resp.status_code)
            return None
        data = resp.json()
        if "error" in data:
            logger.debug("ArcGIS %s → error body: %s", url, data["error"])
            return None
        features = data.get("features", [])
        if not features:
            logger.debug("ArcGIS %s → 0 features", url)
            return None
        rows = [f["attributes"] for f in features]
        df = pd.DataFrame(rows)
        if code_field not in df.columns or name_field not in df.columns:
            logger.debug("ArcGIS %s → fields %s not found", url, (code_field, name_field))
            return None
        df = df.rename(columns={code_field: "LAD_code", name_field: "LAD_name"})
        df = df[["LAD_code", "LAD_name"]].dropna()
        logger.info("ONS LAD lookup: %d entries from %s.", len(df), url)
        return df
    except Exception as exc:
        logger.debug("ArcGIS candidate failed (%s): %s", url, exc)
        return None


def _fetch_from_service() -> pd.DataFrame:
    """Try each candidate service URL in order and return the first success."""
    for url, code_field, name_field in _CANDIDATES:
        df = _try_candidate(url, code_field, name_field)
        if df is not None and not df.empty:
            return df
    raise RuntimeError(
        "All ONS ArcGIS candidates failed. Check network access or update "
        "_CANDIDATES in data_ingestion/sources/ons_lad.py."
    )


def load_lad_lookup(raw_dir: Path) -> pd.DataFrame:
    """
    Return a DataFrame with columns [LAD_code, LAD_name].

    Caches the result in *raw_dir* so repeat runs don't re-fetch.
    Falls back to an empty DataFrame (with a warning) if the service
    is unreachable, so the rest of the pipeline can still run without
    LAD codes.
    """
    cache_path = raw_dir / _CACHE_FILENAME

    if cache_path.exists():
        logger.info("Loading ONS LAD lookup from cache: %s", cache_path.name)
        return pd.read_csv(cache_path, dtype=str)

    try:
        df = _fetch_from_service()
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info("ONS LAD lookup cached at %s.", cache_path)
        return df
    except Exception as exc:
        logger.warning(
            "Could not fetch ONS LAD lookup (%s). LAD_code will be empty.", exc
        )
        return pd.DataFrame(columns=["LAD_code", "LAD_name"])
