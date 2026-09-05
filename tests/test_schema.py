import pytest

from streaming_video_understanding.data.schema import parse_timestamp


def test_parse_timestamp_accepts_hh_mm_ss():
    assert parse_timestamp("00:01:22") == 82.0


def test_parse_timestamp_accepts_decimal_seconds():
    assert parse_timestamp("01:02.5") == 62.5


def test_parse_timestamp_rejects_malformed_value():
    with pytest.raises(ValueError, match="timestamp"):
        parse_timestamp("one minute")
