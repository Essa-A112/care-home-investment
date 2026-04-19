"""Unit tests for data_processing.merge – in-memory DataFrames and tmp CSV."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_processing.merge import merge_all


def _population():
    return pd.DataFrame([
        {"LAD_code": "E09000032", "LAD_name": "Wandsworth",
         "total_population": 330_000, "pop_65_plus": 40_000,
         "pct_65_plus": 12.1, "mye_year": 2023},
        {"LAD_code": "E06000001", "LAD_name": "Hartlepool",
         "total_population": 92_000, "pop_65_plus": 20_000,
         "pct_65_plus": 21.7, "mye_year": 2023},
    ])


def _cqc():
    return pd.DataFrame([
        {"LAD_code": "E09000032", "LAD_name": "Wandsworth",
         "care_home_count": 40, "total_beds": 1_200, "beds_per_1000_65plus": np.nan,
         "pct_outstanding": 5.0, "pct_good": 70.0,
         "pct_requires_improvement": 20.0, "pct_inadequate": 5.0,
         "pct_not_yet_rated": 0.0},
    ])


def _gdhi():
    return pd.DataFrame([
        {"LAD_code": "E09000032", "LAD_name": "Wandsworth",
         "gdhi_per_head": 28_500, "gdhi_year": 2021},
        {"LAD_code": "E06000001", "LAD_name": "Hartlepool",
         "gdhi_per_head": 15_200, "gdhi_year": 2021},
    ])


def _house_prices_csv(tmp_path: Path) -> Path:
    hp = pd.DataFrame([
        {"LAD_code": "E09000032", "LAD_name": "Wandsworth",
         "median_2020": 450_000.0, "median_2024": 540_000.0,
         "house_price_growth_pct": 20.0, "insufficient_data": False},
        {"LAD_code": "E06000001", "LAD_name": "Hartlepool",
         "median_2020": 110_000.0, "median_2024": 130_000.0,
         "house_price_growth_pct": 18.18, "insufficient_data": False},
    ])
    p = tmp_path / "house_price_growth.csv"
    hp.to_csv(p, index=False)
    return p


class TestMergeAll:
    def test_output_has_expected_columns(self, tmp_path):
        hp_path = _house_prices_csv(tmp_path)
        result = merge_all(_population(), _cqc(), _gdhi(), hp_path)
        for col in ["LAD_code", "LAD_name", "total_population", "pop_65_plus",
                    "care_home_count", "gdhi_per_head", "house_price_growth_pct",
                    "missing_any_source"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_beds_per_1000_computed(self, tmp_path):
        hp_path = _house_prices_csv(tmp_path)
        result = merge_all(_population(), _cqc(), _gdhi(), hp_path)
        wands = result[result["LAD_code"] == "E09000032"].iloc[0]
        expected = 1_200 / (40_000 / 1000)
        assert abs(wands["beds_per_1000_65plus"] - expected) < 0.01

    def test_missing_any_source_flagged(self, tmp_path):
        hp_path = _house_prices_csv(tmp_path)
        result = merge_all(_population(), _cqc(), _gdhi(), hp_path)
        # Hartlepool has population + gdhi + house prices but no CQC data
        hart = result[result["LAD_code"] == "E06000001"].iloc[0]
        assert hart["missing_any_source"] is True or hart["missing_any_source"] == True  # noqa: E712

    def test_has_population_flag(self, tmp_path):
        hp_path = _house_prices_csv(tmp_path)
        result = merge_all(_population(), _cqc(), _gdhi(), hp_path)
        assert result["has_population"].all()

    def test_has_cqc_false_for_lad_without_cqc(self, tmp_path):
        hp_path = _house_prices_csv(tmp_path)
        result = merge_all(_population(), _cqc(), _gdhi(), hp_path)
        hart = result[result["LAD_code"] == "E06000001"].iloc[0]
        assert not hart["has_cqc"]

    def test_raises_when_house_prices_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            merge_all(_population(), _cqc(), _gdhi(), tmp_path / "nonexistent.csv")

    def test_sorted_by_lad_code(self, tmp_path):
        hp_path = _house_prices_csv(tmp_path)
        result = merge_all(_population(), _cqc(), _gdhi(), hp_path)
        codes = result["LAD_code"].dropna().tolist()
        assert codes == sorted(codes)
