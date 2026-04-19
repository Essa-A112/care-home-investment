"""
House price processing module.

Loads raw Land Registry Price Paid CSVs, computes per-LAD median prices
for each year, and calculates growth between a start and end year.

Output schema (house_price_growth.csv):
    LAD_code              – ONS nine-character LAD code (e.g. E06000001)
    LAD_name              – ONS official LAD name
    median_2020           – Median transaction price in the start year
    median_<end_year>     – Median transaction price in the end year
    house_price_growth_pct – ((median_end - median_start) / median_start) × 100
    insufficient_data     – True where either year has < MIN_TRANSACTIONS
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Price Paid Data CSVs have no header row.
# Only "price" and "district" are loaded to minimise memory usage.
_PP_ALL_COLUMNS = [
    "transaction_id", "price", "date_of_transfer", "postcode",
    "property_type", "old_new", "duration", "paon", "saon",
    "street", "locality", "town_city", "district", "county",
    "ppd_category_type", "record_status",
]
_PP_USE_COLUMNS = ["price", "district"]
_PP_USE_INDICES = [_PP_ALL_COLUMNS.index(c) for c in _PP_USE_COLUMNS]

MIN_TRANSACTIONS = 30

# Characters stripped when normalising district names for matching
_NORMALISE_RE = re.compile(r"[^a-z0-9 ]")
_WHITESPACE_RE = re.compile(r"\s+")

# Land Registry district names that don't normalise to a direct match against
# the ONS LAD name. Keys are the normalised Land Registry form; values are the
# normalised ONS form. Covers:
#   • "City of X" word-order inversions (LR: "CITY OF BRISTOL", ONS: "Bristol, City of")
#   • Single-word truncations (LR: "WREKIN", ONS: "Telford and Wrekin")
#   • Possessive / county differences (LR: "HEREFORDSHIRE", ONS: "Herefordshire, County of")
_NAME_ALIASES: dict[str, str] = {
    # City of X inversions
    "city of bristol":              "bristol city of",
    "city of derby":                "derby",
    "city of kingston upon hull":   "kingston upon hull city of",
    "city of london":               "city of london",
    "city of nottingham":           "nottingham",
    "city of peterborough":         "peterborough",
    "city of plymouth":             "plymouth",
    "city of westminster":          "westminster",
    "city of york":                 "york",
    # Truncated / renamed
    "wrekin":                       "telford and wrekin",
    "herefordshire":                "herefordshire county of",
    "durham":                       "county durham",
    "st helens":                    "st helens",
    "kings lynn and west norfolk":  "kings lynn and west norfolk",
}


def _normalise_name(name: str) -> str:
    """Lower-case, remove punctuation, collapse whitespace."""
    cleaned = _NORMALISE_RE.sub("", name.lower())
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def _normalise_with_alias(name: str) -> str:
    """Normalise *name* then apply the alias table if a match exists."""
    key = _normalise_name(name)
    return _NAME_ALIASES.get(key, key)


def _load_year(path: Path, year: int) -> pd.DataFrame:
    """
    Read one year's Price Paid CSV and return a two-column DataFrame
    [district, price] with basic cleaning applied.
    """
    logger.info("Loading %s (year=%d).", path.name, year)

    df = pd.read_csv(
        path,
        header=None,
        usecols=_PP_USE_INDICES,
        names=_PP_USE_COLUMNS,
        dtype={"price": "float32", "district": str},
        low_memory=False,
        on_bad_lines="skip",
    )

    before = len(df)
    df = df.dropna(subset=["price", "district"])
    df = df[df["price"] > 0]
    dropped = before - len(df)
    if dropped:
        logger.warning("%s: dropped %d invalid rows.", path.name, dropped)

    df["district"] = df["district"].str.strip().str.upper()
    df["year"] = year

    logger.info("%s: %d valid transactions.", path.name, len(df))
    return df


def compute_lad_medians(paths: dict[int, Path]) -> pd.DataFrame:
    """
    Load all year files and return a DataFrame with columns:
        [district, year, median_price, tx_count]
    """
    frames = [_load_year(path, year) for year, path in sorted(paths.items())]
    all_data = pd.concat(frames, ignore_index=True)

    medians = (
        all_data.groupby(["district", "year"], observed=True)
        .agg(
            median_price=("price", "median"),
            tx_count=("price", "count"),
        )
        .reset_index()
    )

    logger.info(
        "Computed medians for %d LAD-year combinations.",
        len(medians),
    )
    return medians


def attach_lad_codes(
    growth: pd.DataFrame,
    lad_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join official LAD codes and names from the ONS lookup onto *growth*.

    The Land Registry district name (uppercase) and the ONS LAD name
    (mixed case) are both normalised before matching. Rows that cannot
    be matched retain an empty LAD_code.
    """
    if lad_lookup.empty:
        logger.warning("LAD lookup is empty – LAD_code and LAD_name columns will be blank.")
        growth["LAD_code"] = np.nan
        growth["LAD_name"] = np.nan
        return growth

    lookup = lad_lookup.copy()
    lookup["_key"] = lookup["LAD_name"].map(_normalise_name)

    # Use alias-aware normalisation so known Land Registry name variants
    # (e.g. "CITY OF BRISTOL" → "bristol city of") resolve to the correct
    # ONS key even when the word order or format differs.
    growth["_key"] = growth["district"].str.lower().map(
        lambda s: _normalise_with_alias(s) if isinstance(s, str) else ""
    )

    merged = growth.merge(lookup, on="_key", how="left", suffixes=("", "_ons"))

    unmatched = merged["LAD_code"].isna().sum()
    if unmatched:
        bad = merged.loc[merged["LAD_code"].isna(), "district"].unique()
        logger.warning(
            "%d LAD(s) could not be matched to ONS codes: %s",
            unmatched,
            list(bad[:10]),
        )

    merged = merged.drop(columns=["_key"])
    return merged


def compute_growth(
    medians: pd.DataFrame,
    lad_lookup: pd.DataFrame,
    start_year: int,
    end_year: int,
    min_tx: int = MIN_TRANSACTIONS,
) -> pd.DataFrame:
    """
    Compute LAD-level house price growth from *start_year* to *end_year*.

    Rules
    -----
    - A LAD must have >= *min_tx* transactions in **both** years to receive
      a growth figure.
    - LADs that fail this threshold get ``insufficient_data = True`` and
      NaN for the growth percentage.
    - LADs present in one year but not the other are included but also
      flagged as insufficient.

    Returns a DataFrame ready for CSV export with columns:
        LAD_code, LAD_name, median_{start_year}, median_{end_year},
        house_price_growth_pct, insufficient_data
    """
    start_col = f"median_{start_year}"
    end_col = f"median_{end_year}"

    start_df = (
        medians[medians["year"] == start_year]
        .set_index("district")
        .rename(columns={"median_price": start_col, "tx_count": f"tx_{start_year}"})
    )
    end_df = (
        medians[medians["year"] == end_year]
        .set_index("district")
        .rename(columns={"median_price": end_col, "tx_count": f"tx_{end_year}"})
    )

    result = (
        start_df[[start_col, f"tx_{start_year}"]]
        .join(end_df[[end_col, f"tx_{end_year}"]], how="outer")
        .reset_index()
        .rename(columns={"district": "district"})
    )

    # Flag rows that don't meet the transaction minimum in either year
    tx_start = result[f"tx_{start_year}"]
    tx_end = result[f"tx_{end_year}"]

    insufficient = (
        tx_start.isna() | tx_end.isna()
        | (tx_start < min_tx) | (tx_end < min_tx)
    )
    result["insufficient_data"] = insufficient

    # Only compute growth where both years have sufficient coverage
    mask = ~insufficient & result[start_col].notna() & result[end_col].notna()
    result["house_price_growth_pct"] = np.nan
    result.loc[mask, "house_price_growth_pct"] = (
        (result.loc[mask, end_col] - result.loc[mask, start_col])
        / result.loc[mask, start_col]
        * 100
    ).round(4)

    logger.info(
        "Growth computed: %d LADs sufficient, %d flagged insufficient.",
        mask.sum(),
        insufficient.sum(),
    )

    # Attach official LAD codes/names
    result = attach_lad_codes(result, lad_lookup)

    # Build final column order; use LAD_name from ONS if available, else district
    result["LAD_name"] = result["LAD_name"].fillna(result["district"].str.title())

    output_cols = [
        "LAD_code", "LAD_name",
        start_col, end_col,
        "house_price_growth_pct",
        "insufficient_data",
    ]
    return result[output_cols].sort_values("LAD_name").reset_index(drop=True)


def save_output(df: pd.DataFrame, output_path: Path) -> None:
    """Write the processed DataFrame to CSV, creating parent directories."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(
        "Saved %d rows → %s", len(df), output_path
    )
