"""
Master LAD Dataset Pipeline
============================

Orchestrates the end-to-end ingestion, processing, and merge of four
data sources to produce data/processed/master_lad_dataset.csv.

Sources
-------
  1. ONS Mid-Year Population Estimates   (auto-detects latest year)
  2. CQC Care Home Directory             (auto-detects new records via API count)
  3. ONS GDHI per head by Local Authority (auto-detects update via Last-Modified)
  4. Land Registry House Price Growth    (pre-built by house_price_pipeline.py)

Each download is idempotent: already-current files are skipped. The
pipeline re-checks each source for updates every run and only downloads
when the upstream data has changed.

Usage
-----
    python -m pipelines.master_dataset_pipeline

Prerequisites
-------------
    data/processed/house_price_growth.csv must exist.
    Run pipelines/house_price_pipeline.py first if it does not.
"""

import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap – resolve repo root before any relative imports
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_REPO_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from data_ingestion.sources.cqc_directory import download_cqc_directory  # noqa: E402
from data_ingestion.sources.ons_gdhi import download_gdhi  # noqa: E402
from data_ingestion.sources.ons_lad import load_lad_lookup  # noqa: E402
from data_ingestion.sources.ons_population import (  # noqa: E402
    detect_latest_mye_year,
    download_mye,
)
from data_processing.cqc import process_cqc  # noqa: E402
from data_processing.gdhi import load_gdhi  # noqa: E402
from data_processing.merge import merge_all, save_master  # noqa: E402
from data_processing.population import load_population  # noqa: E402

# ---------------------------------------------------------------------------
# Directory / path constants
# ---------------------------------------------------------------------------
_RAW_ONS = _REPO_ROOT / "data" / "raw" / "ons"
_RAW_CQC = _REPO_ROOT / "data" / "raw" / "cqc"
_PROCESSED = _REPO_ROOT / "data" / "processed"

HOUSE_PRICES_CSV = _PROCESSED / "house_price_growth.csv"
OUTPUT_PATH = _PROCESSED / "master_lad_dataset.csv"


def run() -> None:
    logger.info("=== Master LAD Dataset Pipeline starting ===")

    # ------------------------------------------------------------------
    # Step 1 – Download / refresh data sources
    # ------------------------------------------------------------------
    logger.info("Step 1a: ONS Mid-Year Population Estimates.")
    mye_year = detect_latest_mye_year(_RAW_ONS)
    mye_path = download_mye(_RAW_ONS, year=mye_year)

    logger.info("Step 1b: CQC Care Home Directory.")
    cqc_path = download_cqc_directory(_RAW_CQC)

    logger.info("Step 1c: ONS GDHI per head by Local Authority.")
    gdhi_path = download_gdhi(_RAW_ONS)

    logger.info("Step 1d: ONS LAD code lookup (shared).")
    lad_lookup = load_lad_lookup(_RAW_ONS)

    # ------------------------------------------------------------------
    # Step 2 – Process each source to a clean per-LAD DataFrame
    # ------------------------------------------------------------------
    logger.info("Step 2a: Processing population estimates.")
    population_df = load_population(mye_path, year=mye_year)

    logger.info("Step 2b: Processing CQC care home directory.")
    cqc_df = process_cqc(cqc_path, lad_lookup)

    logger.info("Step 2c: Processing GDHI data.")
    gdhi_df = load_gdhi(gdhi_path)

    # ------------------------------------------------------------------
    # Step 3 – Merge all sources into the master dataset
    # ------------------------------------------------------------------
    logger.info("Step 3: Merging all sources on LAD_code.")
    master = merge_all(
        population=population_df,
        cqc=cqc_df,
        gdhi=gdhi_df,
        house_prices_path=HOUSE_PRICES_CSV,
    )

    # ------------------------------------------------------------------
    # Step 4 – Save output
    # ------------------------------------------------------------------
    logger.info("Step 4: Saving master dataset.")
    save_master(master, OUTPUT_PATH)

    logger.info("=== Pipeline complete. ===")
    logger.info(
        "Output: %d LADs | Columns: %d | Missing any source: %d",
        len(master),
        len(master.columns),
        master["missing_any_source"].sum(),
    )


if __name__ == "__main__":
    run()
