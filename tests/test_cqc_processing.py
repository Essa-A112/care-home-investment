"""Unit tests for data_processing.cqc – all in-memory, no file I/O or network."""

import json
import pytest
import pandas as pd
import numpy as np
from data_processing.cqc import (
    _extract_fields,
    _filter_england_active,
    compute_lad_metrics,
)


def _make_location(
    name="Test Home",
    beds=30,
    rating="Good",
    la_code="E09000032",
    la_name="Wandsworth",
    region="London",
    status="Registered",
) -> dict:
    return {
        "name": name,
        "postalCode": "SW1A 1AA",
        "localAuthorityCode": la_code,
        "localAuthority": la_name,
        "region": region,
        "numberOfBeds": beds,
        "registrationStatus": status,
        "currentRatings": {"overall": {"rating": rating}},
    }


class TestExtractFields:
    def test_extracts_basic_fields(self):
        raw = _make_location(beds=50, rating="Outstanding", la_code="E06000001")
        result = _extract_fields(raw)
        assert result["beds"] == 50
        assert result["rating"] == "Outstanding"
        assert result["la_code"] == "E06000001"

    def test_missing_rating_is_none(self):
        raw = _make_location()
        raw["currentRatings"] = {}
        result = _extract_fields(raw)
        assert result["rating"] is None

    def test_null_beds_defaults_to_zero(self):
        raw = _make_location()
        raw["numberOfBeds"] = None
        result = _extract_fields(raw)
        assert result["beds"] == 0


class TestFilterEnglandActive:
    def _df(self, rows):
        return pd.DataFrame([_extract_fields(r) for r in rows])

    def test_keeps_england_registered(self):
        df = self._df([_make_location(la_code="E09000032", status="Registered")])
        result = _filter_england_active(df)
        assert len(result) == 1

    def test_removes_deregistered(self):
        df = self._df([_make_location(status="Deregistered")])
        result = _filter_england_active(df)
        assert len(result) == 0

    def test_removes_wales(self):
        df = self._df([
            _make_location(la_code="W06000001", region="Wales", status="Registered"),
        ])
        result = _filter_england_active(df)
        assert len(result) == 0


class TestComputeLadMetrics:
    def _base_df(self):
        # Supply a raw `rating` column – compute_lad_metrics derives rating_norm itself
        return pd.DataFrame([
            {"LAD_code": "E09000032", "LAD_name": "Wandsworth", "beds": 40, "rating": "Good"},
            {"LAD_code": "E09000032", "LAD_name": "Wandsworth", "beds": 20, "rating": "Outstanding"},
            {"LAD_code": "E06000001", "LAD_name": "Hartlepool", "beds": 0,  "rating": "Requires Improvement"},
        ])

    def test_care_home_count(self):
        result = compute_lad_metrics(self._base_df())
        wands = result[result["LAD_code"] == "E09000032"]
        assert wands["care_home_count"].iloc[0] == 2

    def test_total_beds(self):
        result = compute_lad_metrics(self._base_df())
        wands = result[result["LAD_code"] == "E09000032"]
        assert wands["total_beds"].iloc[0] == 60

    def test_pct_sums_to_100(self):
        result = compute_lad_metrics(self._base_df())
        pct_cols = ["pct_outstanding", "pct_good", "pct_requires_improvement",
                    "pct_inadequate", "pct_not_yet_rated"]
        for _, row in result.iterrows():
            total = sum(row[c] for c in pct_cols)
            assert abs(total - 100.0) < 0.1, f"Percentages don't sum to 100 for {row['LAD_code']}"

    def test_beds_per_1000_is_nan(self):
        result = compute_lad_metrics(self._base_df())
        # beds_per_1000 should be NaN until population data is merged in the master join
        assert result["beds_per_1000_65plus"].isna().all()
