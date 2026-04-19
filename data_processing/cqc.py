"""
CQC Care Home Directory processor.

Reads the raw NDJSON file produced by the CQC downloader, filters to
England care homes, and aggregates to Local Authority District level.

Per-LAD output columns
----------------------
LAD_code                 : ONS LAD code (from ONS lookup via name matching)
LAD_name                 : ONS LAD name
care_home_count          : Total registered care homes in the LAD
total_beds               : Sum of registered bed capacity
beds_per_1000_65plus     : total_beds / (pop_65_plus / 1000)  [computed in merge]
pct_outstanding          : % of homes rated Outstanding
pct_good                 : % of homes rated Good
pct_requires_improvement : % of homes rated Requires Improvement
pct_inadequate           : % of homes rated Inadequate
pct_not_yet_rated        : % of homes without a published rating
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# CQC rating values as returned by the API (case-normalised during processing)
_RATING_MAP = {
    "outstanding": "outstanding",
    "good": "good",
    "requires improvement": "requires_improvement",
    "inadequate": "inadequate",
}

_VALID_ENGLAND_PREFIXES = ("E06", "E07", "E08", "E09")


def _extract_fields(raw: dict) -> dict:
    """
    Pull the fields we need from a single CQC location JSON object.

    The CQC API nests ratings under currentRatings → overall → rating.
    Beds are at the top level as numberOfBeds.
    LAD identification comes from localAuthority (name) and, when
    present, localAuthorityCode.
    """
    overall_rating = None
    ratings = raw.get("currentRatings") or {}
    overall = ratings.get("overall") or {}
    if overall:
        overall_rating = (overall.get("rating") or "").strip() or None

    return {
        "name": raw.get("name", ""),
        "postal_code": raw.get("postalCode", ""),
        "la_code": raw.get("localAuthorityCode", ""),
        "la_name": raw.get("localAuthority") or raw.get("localAuthorityName", ""),
        "region": raw.get("region", ""),
        "beds": raw.get("numberOfBeds") or 0,
        "rating": overall_rating,
        "registration_status": raw.get("registrationStatus", ""),
    }


def load_cqc_raw(path: Path) -> pd.DataFrame:
    """Parse the NDJSON file into a flat DataFrame of individual care homes."""
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(_extract_fields(json.loads(line)))
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Skipping malformed line %d: %s", line_no, exc)

    df = pd.DataFrame(records)
    logger.info("CQC raw: %d care home records loaded.", len(df))
    return df


def _filter_england_active(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retain only active English care homes.

    Active = registrationStatus == "Registered".
    England = la_code starts with E06/E07/E08/E09 OR (if la_code is missing)
    region is not Scotland/Wales/Northern Ireland.
    """
    # Keep only registered homes
    status_mask = df["registration_status"].str.lower() == "registered"
    df = df[status_mask].copy()

    # Identify England by LAD code when available
    has_code = df["la_code"].str.match(r"^E0[6-9]").fillna(False)

    # Fallback: exclude non-English regions by name
    non_england = df["region"].str.lower().isin(
        ["wales", "scotland", "northern ireland"]
    )
    england_mask = has_code | (~non_england & ~df["la_code"].astype(bool))

    result = df[england_mask].copy()
    logger.info(
        "CQC filter: %d England active care homes (from %d total).",
        len(result),
        len(df),
    )
    return result


def _attach_lad_codes(df: pd.DataFrame, lad_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich *df* with ONS LAD codes where la_code is missing or non-standard,
    using the same normalised name matching as the house-price pipeline.
    """
    if lad_lookup.empty:
        logger.warning("LAD lookup empty – using raw la_code values for CQC data.")
        df["LAD_code"] = df["la_code"]
        df["LAD_name"] = df["la_name"]
        return df

    # Use la_code directly if it looks like an ONS code
    valid_code_mask = df["la_code"].str.match(r"^E0[6-9]\d{6}$").fillna(False)

    # For rows with a valid code, join on code; for others, join by normalised name
    import re
    _nonalpha = re.compile(r"[^a-z0-9 ]")
    _whitespace = re.compile(r"\s+")

    def normalise(s: str) -> str:
        return _whitespace.sub(" ", _nonalpha.sub("", s.lower())).strip()

    lookup = lad_lookup.copy()
    lookup["_key"] = lookup["LAD_name"].apply(normalise)

    df["_la_name_norm"] = df["la_name"].apply(
        lambda s: normalise(s) if isinstance(s, str) else ""
    )

    # Build code → (code, name) map and name → (code, name) map
    code_map = lookup.set_index("LAD_code")[["LAD_name"]].to_dict(orient="index")
    name_map = lookup.set_index("_key")[["LAD_code", "LAD_name"]].to_dict(orient="index")

    def resolve_row(row):
        if valid_code_mask.at[row.name] and row["la_code"] in code_map:
            return pd.Series(
                {"LAD_code": row["la_code"], "LAD_name": code_map[row["la_code"]]["LAD_name"]}
            )
        hit = name_map.get(row["_la_name_norm"])
        if hit:
            return pd.Series({"LAD_code": hit["LAD_code"], "LAD_name": hit["LAD_name"]})
        return pd.Series({"LAD_code": row["la_code"] or None, "LAD_name": row["la_name"]})

    resolved = df.apply(resolve_row, axis=1)
    df = pd.concat([df.drop(columns=["_la_name_norm"]), resolved], axis=1)

    unmatched = df["LAD_code"].isna().sum()
    if unmatched:
        logger.warning(
            "CQC: %d homes could not be matched to an ONS LAD code.", unmatched
        )
    return df


def compute_lad_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate individual care home records to produce one row per LAD.

    Rating percentages are computed from homes that are registered;
    homes without a published rating contribute to pct_not_yet_rated.
    beds_per_1000_65plus is left as NaN here and filled in the merge step
    once population data is available.
    """
    df = df.copy()
    df["beds"] = pd.to_numeric(df["beds"], errors="coerce").fillna(0).astype(int)

    # Normalise rating strings
    df["rating_norm"] = (
        df["rating"]
        .fillna("not yet rated")
        .str.lower()
        .str.strip()
    )

    # One-hot encode rating categories
    for category in ["outstanding", "good", "requires_improvement", "inadequate", "not_yet_rated"]:
        api_label = category.replace("_", " ")
        df[f"is_{category}"] = df["rating_norm"] == api_label

    grouped = df.groupby(["LAD_code", "LAD_name"], dropna=False)

    agg = grouped.agg(
        care_home_count=("beds", "count"),
        total_beds=("beds", "sum"),
        n_outstanding=("is_outstanding", "sum"),
        n_good=("is_good", "sum"),
        n_requires_improvement=("is_requires_improvement", "sum"),
        n_inadequate=("is_inadequate", "sum"),
        n_not_yet_rated=("is_not_yet_rated", "sum"),
    ).reset_index()

    total = agg["care_home_count"]
    agg["pct_outstanding"] = (agg["n_outstanding"] / total * 100).round(2)
    agg["pct_good"] = (agg["n_good"] / total * 100).round(2)
    agg["pct_requires_improvement"] = (agg["n_requires_improvement"] / total * 100).round(2)
    agg["pct_inadequate"] = (agg["n_inadequate"] / total * 100).round(2)
    agg["pct_not_yet_rated"] = (agg["n_not_yet_rated"] / total * 100).round(2)

    # beds_per_1000_65plus filled downstream once population is joined
    agg["beds_per_1000_65plus"] = float("nan")

    output_cols = [
        "LAD_code", "LAD_name",
        "care_home_count", "total_beds", "beds_per_1000_65plus",
        "pct_outstanding", "pct_good",
        "pct_requires_improvement", "pct_inadequate", "pct_not_yet_rated",
    ]
    result = agg[output_cols].sort_values("LAD_code").reset_index(drop=True)
    logger.info("CQC aggregated: %d LADs.", len(result))
    return result


def process_cqc(path: Path, lad_lookup: pd.DataFrame) -> pd.DataFrame:
    """Full CQC processing pipeline: load → filter → enrich → aggregate."""
    df = load_cqc_raw(path)
    df = _filter_england_active(df)
    df = _attach_lad_codes(df, lad_lookup)
    return compute_lad_metrics(df)
