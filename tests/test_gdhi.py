"""Unit tests for data_processing.gdhi helper functions – no file I/O."""

import re
import pytest
import pandas as pd
from data_processing.gdhi import (
    _match_column,
    _find_header_row,
    _YEAR_COL_RE,
    _CODE_COL_PATTERNS,
    _NAME_COL_PATTERNS,
)


class TestMatchColumn:
    def test_finds_code_column(self):
        cols = ["LAD23CD", "LAD23NM", "2020", "2021"]
        result = _match_column(cols, _CODE_COL_PATTERNS)
        assert result == "LAD23CD"

    def test_finds_name_column(self):
        cols = ["LAD23CD", "LAD23NM", "2020"]
        result = _match_column(cols, _NAME_COL_PATTERNS)
        assert result == "LAD23NM"

    def test_returns_none_when_not_found(self):
        assert _match_column(["foo", "bar"], _CODE_COL_PATTERNS) is None


class TestYearColRe:
    def test_matches_valid_years(self):
        for year in ["1997", "2023", "2099"]:
            assert _YEAR_COL_RE.match(year), f"{year} should match"

    def test_rejects_non_years(self):
        for val in ["200", "20234", "Name", "code"]:
            assert not _YEAR_COL_RE.match(val), f"{val} should not match"


class TestFindHeaderRow:
    def test_finds_row_with_year(self):
        raw = pd.DataFrame([
            ["Title", None],
            ["code", "name", "2020", "2021"],
        ])
        # Row index 1 has "2020" which matches the year pattern
        assert _find_header_row(raw) == 1

    def test_raises_when_no_year_row(self):
        raw = pd.DataFrame([["foo", "bar"], ["baz", "qux"]])
        with pytest.raises(ValueError):
            _find_header_row(raw)
