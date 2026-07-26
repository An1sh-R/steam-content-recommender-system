"""Pins the raw dataset's header defect.

If these fail, the upstream layout changed and every downstream feature is
suspect until RAW_COLUMNS is re-verified.
"""

import csv

import pytest

from recommender import config, schema


def test_raw_columns_declares_forty_fields():
    assert len(schema.RAW_COLUMNS) == 40
    assert len(set(schema.RAW_COLUMNS)) == 40, "column names must be unique"


def test_published_header_is_malformed():
    """The header declares 39 names while rows carry 40 fields."""
    with open(config.SAMPLE_CSV, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        widths = {len(row) for row in reader}

    assert len(header) == 39
    assert widths == {40}
    assert schema.MERGED_HEADER_FIELD in header, "the two merged columns should still be merged"


def test_merged_field_is_the_documented_one():
    with open(config.SAMPLE_CSV, encoding="utf-8", newline="") as fh:
        header = next(csv.reader(fh))

    assert header[7] == "DiscountDLC count"
    # RAW_COLUMNS splits exactly that field into two.
    assert schema.RAW_COLUMNS[7] == "Discount"
    assert schema.RAW_COLUMNS[8] == "DLC count"


def test_validate_raw_shape_accepts_the_sample():
    schema.validate_raw_shape(config.SAMPLE_CSV)


def test_validate_raw_shape_rejects_a_changed_layout(tmp_path):
    bad = tmp_path / "games.csv"
    bad.write_text("a,b,c\n1,2,3\n", encoding="utf-8")

    with pytest.raises(schema.RawSchemaError):
        schema.validate_raw_shape(bad)
