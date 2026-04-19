"""
ONS GDHI (Gross Disposable Household Income) per head processor.

Parses the ONS GDHI by Local Authority Excel workbook and extracts the
most recent available year of GDHI per head of population for each LAD.

Expected output columns
-----------------------
LAD_code      : ONS nine-character LAD code (e.g. E06000001)
LAD_name      : ONS LAD name
gdhi_per_head : GDHI per head (£, current prices) for the latest year
gdhi_year     : Integer year of the GDHI estimate

Notes
-----
The ONS GDHI Excel file contains multiple sheets. The sheet providing
GDHI per head at local authority level is typically named "Table 3",
"Per head", "GDHI per head", or a year-suffixed variant. The processor
tries several candidate names and raises a descriptive error if none match.

The geographic codes in the ONS GDHI dataset use the standard nine-
character ONS codes (E06xxxxxx etc.) so they join directly to the
LAD_code key used throughout this project.
"""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Candidate sheet names for the GDHI-per-head table (tried in order)
_SHEET_CANDIDATES = [
    "Table 3",
    "Table3",
    "Per head",
    "GDHI per head",
    "Per Head",
    "per head",
    "LA per head",
    "3",
]

# Column or row label patterns that indicate a geographic code column
_CODE_COL_PATTERNS = [
    re.compile(r"^(la|lad|nuts|area|geography|geo|local.authority).*(code|cd)$", re.I),
    re.compile(r"^code$", re.I),
    re.compile(r"^[A-Z]{2,4}(22|23|24)?CD$"),
]

# Column or row label patterns for the name column
_NAME_COL_PATTERNS = [
    re.compile(r"^(la|lad|nuts|area|geography|local.authority).*(name|nm)$", re.I),
    re.compile(r"^name$", re.I),
    re.compile(r"^[A-Z]{2,4}(22|23|24)?NM$"),
]

# Pattern identifying a year column (four-digit year 1990–2099)
_YEAR_COL_RE = re.compile(r"^(19|20)\d{2}$")


def _match_column(columns: list[str], patterns: list[re.Pattern]) -> str | None:
    """Return the first column that matches any of *patterns*, or None."""
    for col in columns:
        for pat in patterns:
            if pat.match(str(col).strip()):
                return col
    return None


def _find_sheet(path: Path) -> str:
    """Return the name of the GDHI-per-head sheet, or raise ValueError."""
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    available = wb.sheetnames
    wb.close()

    for candidate in _SHEET_CANDIDATES:
        if candidate in available:
            return candidate

    # Fuzzy fallback: any sheet whose name contains "per head"
    for name in available:
        if "per head" in name.lower() or "perhead" in name.lower():
            return name

    raise ValueError(
        f"Could not locate GDHI per-head sheet in {path.name}. "
        f"Available: {available}. Tried: {_SHEET_CANDIDATES}"
    )


def _find_header_row(raw: pd.DataFrame) -> int:
    """
    Find the first row that contains a year-like column header.
    ONS Excel files often have multiple title rows before the data table.
    """
    for idx, row in raw.iterrows():
        vals = [str(v).strip() for v in row if pd.notna(v)]
        if any(_YEAR_COL_RE.match(v) for v in vals):
            return idx
    raise ValueError("Could not locate header row with year columns in GDHI sheet.")


def load_gdhi(path: Path) -> pd.DataFrame:
    """
    Parse the ONS GDHI per-head Excel file and return a tidy LAD DataFrame.

    Steps
    -----
    1. Identify the correct sheet.
    2. Find the header row (first row with year-like column labels).
    3. Filter to England LAD rows (code starts with E06/E07/E08/E09).
    4. Select the most recent year column.
    5. Return the tidy output DataFrame.
    """
    sheet_name = _find_sheet(path)
    logger.info("Reading GDHI sheet '%s' from %s.", sheet_name, path.name)

    # First pass: find header row
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    header_row = _find_header_row(raw)

    # Second pass: read with correct header
    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    # Identify code and name columns
    cols = list(df.columns)
    code_col = _match_column(cols, _CODE_COL_PATTERNS)
    name_col = _match_column(cols, _NAME_COL_PATTERNS)

    if code_col is None:
        # Last resort: look for a column whose values start with 'E0'
        for col in cols:
            sample = df[col].dropna().astype(str).head(20)
            if sample.str.match(r"^E0[6-9]\d{6}$").any():
                code_col = col
                break

    if code_col is None:
        raise ValueError(
            f"Cannot identify LAD code column in GDHI sheet. Columns: {cols[:20]}"
        )

    logger.debug("GDHI code column: '%s', name column: '%s'.", code_col, name_col)

    # Filter to England LAD rows
    code_series = df[code_col].astype(str).str.strip()
    england_mask = code_series.str.match(r"^E0[6-9]\d{6}$")
    df = df.loc[england_mask].copy()

    if df.empty:
        raise ValueError(
            f"No England LAD rows found after filtering GDHI sheet '{sheet_name}'. "
            f"Sample codes: {code_series.head(10).tolist()}"
        )

    # Identify year columns and pick the latest
    year_cols = [c for c in df.columns if _YEAR_COL_RE.match(str(c))]
    if not year_cols:
        raise ValueError(f"No year columns found in GDHI sheet. Columns: {cols}")

    latest_year_col = sorted(year_cols, key=lambda c: int(c))[-1]
    latest_year = int(latest_year_col)
    logger.info("GDHI latest year: %d (from %d available).", latest_year, len(year_cols))

    gdhi_values = pd.to_numeric(df[latest_year_col], errors="coerce")

    result = pd.DataFrame(
        {
            "LAD_code": code_series.loc[df.index].values,
            "LAD_name": (
                df[name_col].astype(str).str.strip().values
                if name_col else np.nan
            ),
            "gdhi_per_head": gdhi_values.values,
            "gdhi_year": latest_year,
        }
    )

    before = len(result)
    result = result.dropna(subset=["gdhi_per_head"])
    if before - len(result):
        logger.warning(
            "GDHI: dropped %d rows with null per-head values.", before - len(result)
        )

    logger.info(
        "GDHI loaded: %d England LADs, year %d.", len(result), latest_year
    )
    return result.sort_values("LAD_code").reset_index(drop=True)
