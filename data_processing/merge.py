"""
Master LAD dataset merger.

Joins four processed datasets on LAD_code to produce the project's
master feature table:

  1. ONS mid-year population estimates   (population.py output)
  2. CQC care home directory             (cqc.py output)
  3. ONS GDHI per head                   (gdhi.py output)
  4. House price growth                  (house_price_pipeline.py output)

For each source, a boolean flag column records whether data is present:
  has_population, has_cqc, has_gdhi, has_house_prices

A summary flag ``missing_any_source`` is True when any of the four are
absent, making it easy for downstream models to filter or impute.

After joining, beds_per_1000_65plus is computed using the merged
population column (this requires both CQC and population to be present).

Output: data/processed/master_lad_dataset.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Sources expected in the master join
_SOURCE_FLAGS = ["has_population", "has_cqc", "has_gdhi", "has_house_prices"]


def _load_house_prices(path: Path) -> pd.DataFrame:
    """Load house_price_growth.csv produced by the house-price pipeline."""
    if not path.exists():
        raise FileNotFoundError(
            f"House price growth file not found at {path}. "
            "Run pipelines/house_price_pipeline.py first."
        )
    df = pd.read_csv(path, dtype={"LAD_code": str})
    logger.info("House price data loaded: %d rows.", len(df))
    return df


def merge_all(
    population: pd.DataFrame,
    cqc: pd.DataFrame,
    gdhi: pd.DataFrame,
    house_prices_path: Path,
) -> pd.DataFrame:
    """
    Outer-join all four sources on LAD_code.

    The outer join ensures every LAD that appears in any source is
    represented in the master dataset, even if some sources have no data
    for that LAD (those cells become NaN and the corresponding flag is False).

    beds_per_1000_65plus is computed here once pop_65_plus is available
    from the population join.
    """
    hp = _load_house_prices(house_prices_path)

    # Standardise: keep only the LAD_code column as the join key; rename
    # LAD_name columns to avoid collisions (we keep population's name as canonical).
    def _prep(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        df = df.copy()
        # If there's a LAD_name column, suffix it uniquely
        if "LAD_name" in df.columns and prefix != "pop":
            df = df.rename(columns={"LAD_name": f"LAD_name_{prefix}"})
        return df

    pop = _prep(population, "pop")
    cqc_df = _prep(cqc, "cqc")
    gdhi_df = _prep(gdhi, "gdhi")
    hp_df = _prep(hp, "hp")

    logger.info(
        "Merging: pop=%d, cqc=%d, gdhi=%d, hp=%d LADs.",
        len(pop), len(cqc_df), len(gdhi_df), len(hp_df),
    )

    # Sequential outer joins – LAD_code is the universal key
    master = pop.merge(cqc_df, on="LAD_code", how="outer")
    master = master.merge(gdhi_df, on="LAD_code", how="outer")
    master = master.merge(hp_df, on="LAD_code", how="outer")

    # Determine the canonical LAD_name: prefer population's name, then others
    name_cols = ["LAD_name", "LAD_name_cqc", "LAD_name_gdhi", "LAD_name_hp"]
    existing_name_cols = [c for c in name_cols if c in master.columns]
    if existing_name_cols:
        master["LAD_name"] = master[existing_name_cols[0]]
        for col in existing_name_cols[1:]:
            master["LAD_name"] = master["LAD_name"].fillna(master[col])
        # Drop redundant name columns
        drop_cols = [c for c in existing_name_cols[1:] if c in master.columns]
        master = master.drop(columns=drop_cols)

    # Source-presence flags
    master["has_population"] = master["total_population"].notna()
    master["has_cqc"] = master["care_home_count"].notna()
    master["has_gdhi"] = master["gdhi_per_head"].notna()

    # House price growth presence: sufficient data = not flagged insufficient
    hp_code_set = set(hp["LAD_code"].dropna())
    master["has_house_prices"] = master["LAD_code"].isin(hp_code_set) & (
        ~master["insufficient_data"].fillna(True)
    )

    master["missing_any_source"] = ~master[_SOURCE_FLAGS].all(axis=1)

    # Compute beds_per_1000_65plus now that pop_65_plus is in the merged frame
    can_compute = master["total_beds"].notna() & master["pop_65_plus"].notna() & (master["pop_65_plus"] > 0)
    master["beds_per_1000_65plus"] = np.nan
    master.loc[can_compute, "beds_per_1000_65plus"] = (
        master.loc[can_compute, "total_beds"] / (master.loc[can_compute, "pop_65_plus"] / 1000)
    ).round(2)

    # Round floating-point columns for readability
    for col in ["pct_65_plus", "gdhi_per_head", "house_price_growth_pct",
                "beds_per_1000_65plus"]:
        if col in master.columns:
            master[col] = master[col].round(4)

    # Column ordering: identity columns first, then each source's features
    priority = [
        "LAD_code", "LAD_name",
        # Population
        "total_population", "pop_65_plus", "pct_65_plus", "mye_year",
        # CQC
        "care_home_count", "total_beds", "beds_per_1000_65plus",
        "pct_outstanding", "pct_good", "pct_requires_improvement",
        "pct_inadequate", "pct_not_yet_rated",
        # GDHI
        "gdhi_per_head", "gdhi_year",
        # House prices
        "median_2020", "house_price_growth_pct", "insufficient_data",
        # Flags
        "has_population", "has_cqc", "has_gdhi", "has_house_prices",
        "missing_any_source",
    ]
    # Dynamic: find median_{end_year} column that might not be median_2024
    extra_median_cols = [c for c in master.columns if c.startswith("median_") and c != "median_2020"]
    all_ordered = priority + extra_median_cols
    final_cols = [c for c in all_ordered if c in master.columns]
    remaining = [c for c in master.columns if c not in final_cols]
    master = master[final_cols + remaining]

    master = master.sort_values("LAD_code", na_position="last").reset_index(drop=True)

    n_missing = master["missing_any_source"].sum()
    logger.info(
        "Master dataset: %d LADs total, %d with all sources present, %d missing at least one.",
        len(master),
        len(master) - n_missing,
        n_missing,
    )
    return master


def save_master(df: pd.DataFrame, output_path: Path) -> None:
    """Write the master DataFrame to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Master dataset saved: %d rows, %d columns → %s.", len(df), len(df.columns), output_path)
