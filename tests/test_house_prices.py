"""
Unit tests for data_processing.house_prices.

All tests use in-memory DataFrames – no file I/O, no network calls.
"""

import numpy as np
import pandas as pd
import pytest

from data_processing.house_prices import (
    MIN_TRANSACTIONS,
    _normalise_name,
    attach_lad_codes,
    compute_growth,
)


def _make_medians(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def _empty_lookup() -> pd.DataFrame:
    return pd.DataFrame(columns=["LAD_code", "LAD_name"])


# ---------------------------------------------------------------------------
# _normalise_name
# ---------------------------------------------------------------------------

class TestNormaliseName:
    def test_strips_punctuation(self):
        assert _normalise_name("Bristol, City of") == "bristol city of"

    def test_lowercases(self):
        assert _normalise_name("WANDSWORTH") == "wandsworth"

    def test_collapses_whitespace(self):
        assert _normalise_name("  Isle  of Wight  ") == "isle of wight"


# ---------------------------------------------------------------------------
# compute_growth – sufficient LADs
# ---------------------------------------------------------------------------

class TestComputeGrowthSufficient:
    def _medians(self):
        rows = []
        for district in ["WANDSWORTH", "BRISTOL"]:
            for year in [2020, 2024]:
                rows.append({
                    "district": district,
                    "year": year,
                    "median_price": 300_000.0 if year == 2020 else 360_000.0,
                    "tx_count": 100,
                })
        return pd.DataFrame(rows)

    def test_growth_pct_correct(self):
        df = compute_growth(self._medians(), _empty_lookup(), 2020, 2024)
        assert df["house_price_growth_pct"].tolist() == pytest.approx([20.0, 20.0], rel=1e-3)

    def test_insufficient_data_false(self):
        df = compute_growth(self._medians(), _empty_lookup(), 2020, 2024)
        assert (~df["insufficient_data"]).all()

    def test_output_columns_present(self):
        df = compute_growth(self._medians(), _empty_lookup(), 2020, 2024)
        assert set(df.columns) >= {
            "LAD_code", "LAD_name", "median_2020", "median_2024",
            "house_price_growth_pct", "insufficient_data",
        }


# ---------------------------------------------------------------------------
# compute_growth – insufficient LADs
# ---------------------------------------------------------------------------

class TestComputeGrowthInsufficient:
    def _medians_low_tx(self):
        return pd.DataFrame([
            {"district": "TINYTOWN", "year": 2020, "median_price": 200_000.0, "tx_count": 5},
            {"district": "TINYTOWN", "year": 2024, "median_price": 250_000.0, "tx_count": 5},
        ])

    def test_flagged_insufficient(self):
        df = compute_growth(self._medians_low_tx(), _empty_lookup(), 2020, 2024)
        row = df[df["LAD_name"].str.upper() == "TINYTOWN"]
        assert row["insufficient_data"].all()

    def test_growth_pct_is_nan_for_insufficient(self):
        df = compute_growth(self._medians_low_tx(), _empty_lookup(), 2020, 2024)
        row = df[df["LAD_name"].str.upper() == "TINYTOWN"]
        assert row["house_price_growth_pct"].isna().all()

    def test_missing_end_year_flagged(self):
        medians = pd.DataFrame([
            {"district": "GHOSTTOWN", "year": 2020, "median_price": 150_000.0, "tx_count": 50},
        ])
        df = compute_growth(medians, _empty_lookup(), 2020, 2024)
        row = df[df["LAD_name"].str.upper() == "GHOSTTOWN"]
        assert row["insufficient_data"].all()


# ---------------------------------------------------------------------------
# attach_lad_codes
# ---------------------------------------------------------------------------

class TestAttachLadCodes:
    def _base_growth(self):
        return pd.DataFrame([
            {"district": "WANDSWORTH", "house_price_growth_pct": 20.0, "insufficient_data": False},
            {"district": "NOWHERESVILE", "house_price_growth_pct": np.nan, "insufficient_data": True},
        ])

    def _lookup(self):
        return pd.DataFrame([
            {"LAD_code": "E09000032", "LAD_name": "Wandsworth"},
        ])

    def test_matched_code_assigned(self):
        result = attach_lad_codes(self._base_growth(), self._lookup())
        wandsworth = result[result["district"] == "WANDSWORTH"]
        assert wandsworth["LAD_code"].iloc[0] == "E09000032"

    def test_unmatched_code_is_nan(self):
        result = attach_lad_codes(self._base_growth(), self._lookup())
        nowhere = result[result["district"] == "NOWHERESVILE"]
        assert pd.isna(nowhere["LAD_code"].iloc[0])

    def test_empty_lookup_does_not_raise(self):
        result = attach_lad_codes(self._base_growth(), _empty_lookup())
        assert "LAD_code" in result.columns
