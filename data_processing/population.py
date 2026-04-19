"""
ONS Mid-Year Population Estimates processor.

Parses the ONS MYE Excel workbook and produces a clean per-LAD DataFrame
with total population, population aged 65 and over, and the share of
residents in that age band.

Expected output columns
-----------------------
LAD_code        : ONS nine-character LAD code (e.g. E06000001)
LAD_name        : ONS LAD name
total_population: Integer total population
pop_65_plus     : Integer count of residents aged 65 and over
pct_65_plus     : Float percentage (pop_65_plus / total_population * 100)
mye_year        : Integer year of the mid-year estimate used
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Candidate sheet names for the "persons by single year of age" tab.
# ONS has varied the label across publication years.
_SHEET_CANDIDATES = [
    "MYE2 - Persons",
    "MYE2-Persons",
    "MYE 2 - Persons",
    "Persons",
    "MYE2",
]

# Columns in the sheet that hold the LAD code and name
_CODE_CANDIDATES = ["Code", "Area code", "Area Code", "code"]
_NAME_CANDIDATES = ["Name", "Area name", "Area Name", "name"]

# Pattern to identify age columns (integers or "90+")
_AGE_COL_RE = re.compile(r"^\d+$")
_AGE_PLUS_RE = re.compile(r"^(\d+)\+$")   # e.g. "90+"


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Return the first matching column name from *candidates*, case-insensitively."""
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    raise KeyError(
        f"None of {candidates} found in columns: {list(df.columns[:20])}"
    )


def _open_mye_sheet(path: Path) -> pd.DataFrame:
    """
    Open the ONS MYE Excel workbook and return the persons-by-age sheet
    as a raw DataFrame. Tries each candidate sheet name in order.
    """
    import openpyxl  # lazy import – only required for Excel parsing

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    available = wb.sheetnames
    wb.close()

    for sheet in _SHEET_CANDIDATES:
        if sheet in available:
            logger.info("Reading ONS MYE sheet '%s' from %s.", sheet, path.name)
            return pd.read_excel(path, sheet_name=sheet, header=None)

    raise ValueError(
        f"Could not find a persons-by-age sheet in {path.name}. "
        f"Available sheets: {available}. "
        f"Tried: {_SHEET_CANDIDATES}"
    )


def _locate_header_row(raw: pd.DataFrame) -> int:
    """
    Find the row index that contains the column headers (Code, Name, age cols).
    ONS files typically have 3–8 rows of title/notes before the data table.
    """
    for idx, row in raw.iterrows():
        vals = [str(v).strip() for v in row if pd.notna(v)]
        if any(v.lower() in {"code", "area code"} for v in vals):
            return idx
    raise ValueError("Could not locate header row in ONS MYE sheet.")


def _parse_age_columns(columns: pd.Index) -> dict[int, str]:
    """
    Return a mapping {age_int: column_label} for all age columns found.
    Handles both plain integers ('0', '1', ..., '90') and annotated
    strings ('90+', '90 and over').
    """
    age_map: dict[int, str] = {}
    for col in columns:
        col_str = str(col).strip()
        if _AGE_COL_RE.match(col_str):
            age_map[int(col_str)] = col_str
        else:
            m = _AGE_PLUS_RE.match(col_str)
            if m:
                age_map[int(m.group(1))] = col_str
            elif "and over" in col_str.lower():
                # e.g. "90 and over"
                digits = re.search(r"\d+", col_str)
                if digits:
                    age_map[int(digits.group())] = col_str
    return age_map


def load_population(path: Path, year: int) -> pd.DataFrame:
    """
    Parse the ONS MYE Excel file and return a clean LAD-level DataFrame.

    Steps
    -----
    1. Locate the persons-by-single-year-of-age sheet.
    2. Find the header row and set it as column labels.
    3. Filter to England LAD rows (code starts with 'E').
    4. Identify age columns, sum ages 65+ for each LAD.
    5. Return the tidy output DataFrame.
    """
    raw = _open_mye_sheet(path)
    header_row = _locate_header_row(raw)

    # Re-read with the correct header row
    df = pd.read_excel(
        path,
        sheet_name=None,  # We'll re-open below with the correct sheet
        header=header_row,
    )
    # Re-open specific sheet with known header position
    for sheet in _SHEET_CANDIDATES:
        try:
            df = pd.read_excel(path, sheet_name=sheet, header=header_row)
            break
        except Exception:
            continue

    # Normalise column names (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]

    code_col = _find_column(df, _CODE_CANDIDATES)
    name_col = _find_column(df, _NAME_CANDIDATES)

    # Keep only England LAD rows (ONS codes E06xxxxxx, E07xxxxxx, E08xxxxxx, E09xxxxxx)
    code_series = df[code_col].astype(str).str.strip()
    england_lad_mask = code_series.str.match(r"^E0[6-9]\d{6}$")
    df = df.loc[england_lad_mask].copy()

    if df.empty:
        raise ValueError(
            f"No England LAD rows found in {path.name}. "
            f"Sample codes: {df[code_col].dropna().head(5).tolist()}"
        )

    age_map = _parse_age_columns(df.columns)
    if not age_map:
        raise ValueError(f"No age columns detected in {path.name}. Columns: {list(df.columns[:30])}")

    logger.info(
        "Found %d age columns (range %d–%d) in %s.",
        len(age_map),
        min(age_map),
        max(age_map),
        path.name,
    )

    # Total population – prefer an explicit "All Ages" column; otherwise sum all ages
    all_ages_candidates = ["All Ages", "All ages", "all ages", "Total"]
    try:
        total_col = _find_column(df, all_ages_candidates)
        total_pop = pd.to_numeric(df[total_col], errors="coerce")
    except KeyError:
        logger.warning("'All Ages' column not found – computing total by summing all age columns.")
        age_cols = [age_map[a] for a in age_map]
        total_pop = df[age_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    # 65+ population: sum all age columns where age >= 65
    cols_65_plus = [age_map[a] for a in age_map if a >= 65]
    if not cols_65_plus:
        raise ValueError("No age columns >= 65 found. Cannot compute pop_65_plus.")

    pop_65_plus = df[cols_65_plus].apply(pd.to_numeric, errors="coerce").sum(axis=1)

    result = pd.DataFrame(
        {
            "LAD_code": code_series.loc[df.index].values,
            "LAD_name": df[name_col].astype(str).str.strip().values,
            "total_population": total_pop.values,
            "pop_65_plus": pop_65_plus.values,
        }
    )

    result["total_population"] = result["total_population"].round(0).astype("Int64")
    result["pop_65_plus"] = result["pop_65_plus"].round(0).astype("Int64")
    result["pct_65_plus"] = (
        result["pop_65_plus"] / result["total_population"] * 100
    ).round(2)
    result["mye_year"] = year

    before = len(result)
    result = result.dropna(subset=["total_population", "pop_65_plus"])
    if before - len(result):
        logger.warning(
            "Dropped %d rows with null population values.", before - len(result)
        )

    logger.info(
        "Population data loaded: %d England LADs for MYE %d.", len(result), year
    )
    return result.sort_values("LAD_code").reset_index(drop=True)
