"""
ONS Local Authority District (LAD) code lookup downloader.

Fetches a lightweight name → code reference table from the ONS Open
Geography Portal ArcGIS REST service so that Price Paid Data district
names can be mapped to official nine-character LAD codes (e.g. E06000001).

The service returns ~374 rows for the full UK; England-only rows start
with 'E'.
"""

import logging
from io import StringIO
from pathlib import Path

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

# ONS Open Geography Portal – LAD December 2023 boundaries
# Field names: LAD23CD (code), LAD23NM (name)
_ONS_LAD_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services"
    "/LAD_DEC_2023_UK_NC/FeatureServer/0/query"
)
_ONS_LAD_PARAMS = {
    "where": "1=1",
    "outFields": "LAD23CD,LAD23NM",
    "returnGeometry": "false",
    "f": "csv",
    "resultRecordCount": "500",
}

_CACHE_FILENAME = "ons_lad_lookup.csv"


def _fetch_from_service() -> pd.DataFrame:
    """Download the LAD reference table from ONS ArcGIS and return a DataFrame."""
    logger.info("Fetching ONS LAD lookup from ArcGIS REST service.")
    resp = httpx.get(_ONS_LAD_URL, params=_ONS_LAD_PARAMS, timeout=60, follow_redirects=True)
    resp.raise_for_status()

    df = pd.read_csv(StringIO(resp.text))

    # Normalise column names – the service may vary LAD year suffix
    code_col = next(c for c in df.columns if c.endswith("CD"))
    name_col = next(c for c in df.columns if c.endswith("NM"))
    df = df.rename(columns={code_col: "LAD_code", name_col: "LAD_name"})

    df = df[["LAD_code", "LAD_name"]].dropna()
    logger.info("ONS LAD lookup: %d entries fetched.", len(df))
    return df


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
