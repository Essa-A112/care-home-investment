"""
House Price Growth Pipeline
===========================

Entry point that orchestrates:

  1. Detect the latest available Land Registry year (auto-advances beyond 2024).
  2. Download raw Price Paid CSVs (idempotent – skips files already on disk).
  3. Download the ONS LAD code lookup (cached locally after first fetch).
  4. Compute per-LAD median house prices for each year.
  5. Compute growth between the start year and the detected end year.
  6. Write ``data/processed/house_price_growth.csv``.

Usage
-----
    python -m pipelines.house_price_pipeline

Environment
-----------
No secrets are required for this pipeline; all data is public.
The .env file is loaded automatically if present (for other pipeline use).
"""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path resolution – works whether the module is run as a script or imported.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(_REPO_ROOT / ".env")

# ---------------------------------------------------------------------------
# Logging – configure before importing project modules so all loggers inherit
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project imports (after logging setup)
# ---------------------------------------------------------------------------
from data_ingestion.sources.land_registry import (  # noqa: E402
    START_YEAR,
    detect_end_year,
    download_all,
)
from data_ingestion.sources.ons_lad import load_lad_lookup  # noqa: E402
from data_processing.house_prices import (  # noqa: E402
    compute_growth,
    compute_lad_medians,
    save_output,
)

# ---------------------------------------------------------------------------
# Directory constants – relative to repo root, no hardcoded absolute paths
# ---------------------------------------------------------------------------
RAW_DIR = _REPO_ROOT / "data" / "raw" / "land_registry"
PROCESSED_DIR = _REPO_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "house_price_growth.csv"


def run() -> None:
    logger.info("=== House Price Growth Pipeline starting ===")

    # Step 1 – detect latest available year
    logger.info("Step 1: Detecting latest available Land Registry year.")
    end_year = detect_end_year()
    logger.info("End year resolved to: %d", end_year)

    # Step 2 – download raw Price Paid Data
    logger.info("Step 2: Downloading Price Paid Data %d–%d.", START_YEAR, end_year)
    raw_paths = download_all(raw_dir=RAW_DIR, end_year=end_year)

    # Step 3 – fetch ONS LAD lookup
    logger.info("Step 3: Loading ONS LAD code lookup.")
    lad_lookup = load_lad_lookup(raw_dir=RAW_DIR)

    # Step 4 – compute per-LAD medians across all years
    logger.info("Step 4: Computing LAD-level median house prices.")
    medians = compute_lad_medians(raw_paths)

    # Step 5 – compute growth (start_year=2020, end_year=detected)
    logger.info(
        "Step 5: Computing house price growth %d → %d.", START_YEAR, end_year
    )
    growth_df = compute_growth(
        medians=medians,
        lad_lookup=lad_lookup,
        start_year=START_YEAR,
        end_year=end_year,
    )

    # Step 6 – persist output
    logger.info("Step 6: Saving output to %s.", OUTPUT_PATH)
    save_output(growth_df, OUTPUT_PATH)

    logger.info("=== Pipeline complete. ===")
    logger.info(
        "Rows written: %d  |  Insufficient data: %d",
        len(growth_df),
        growth_df["insufficient_data"].sum(),
    )


if __name__ == "__main__":
    run()
