"""Unit tests for data_processing.population – all in-memory, no file I/O."""

import re
import pytest
from data_processing.population import (
    _find_column,
    _locate_header_row,
    _parse_age_columns,
)
import pandas as pd


class TestFindColumn:
    def test_exact_match(self):
        df = pd.DataFrame(columns=["Code", "Name", "0", "1"])
        assert _find_column(df, ["Code"]) == "Code"

    def test_case_insensitive_match(self):
        df = pd.DataFrame(columns=["area code", "name"])
        assert _find_column(df, ["Area Code", "Code"]) == "area code"

    def test_raises_when_not_found(self):
        df = pd.DataFrame(columns=["X", "Y"])
        with pytest.raises(KeyError):
            _find_column(df, ["Code", "area code"])


class TestLocateHeaderRow:
    def test_finds_row_with_code(self):
        raw = pd.DataFrame([
            ["Title row", None, None],
            ["Code", "Name", 0],
        ])
        assert _locate_header_row(raw) == 1

    def test_raises_when_no_header(self):
        raw = pd.DataFrame([["foo", "bar"], ["baz", "qux"]])
        with pytest.raises(ValueError):
            _locate_header_row(raw)


class TestParseAgeColumns:
    def test_integer_columns(self):
        cols = pd.Index(["Code", "Name", "0", "1", "65", "90"])
        result = _parse_age_columns(cols)
        assert 65 in result
        assert 90 in result
        assert result[0] == "0"

    def test_plus_suffix(self):
        cols = pd.Index(["Code", "Name", "64", "90+"])
        result = _parse_age_columns(cols)
        assert 90 in result
        assert result[90] == "90+"

    def test_and_over_suffix(self):
        cols = pd.Index(["90 and over"])
        result = _parse_age_columns(cols)
        assert 90 in result

    def test_ignores_non_age_columns(self):
        cols = pd.Index(["Code", "Name", "All Ages", "Geography"])
        result = _parse_age_columns(cols)
        assert len(result) == 0
