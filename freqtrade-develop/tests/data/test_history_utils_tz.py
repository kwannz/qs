import pandas as pd

from freqtrade.data.history.history_utils import _normalize_ohlcv_timezone_and_boundary


def test_normalize_ohlcv_timezone_and_boundary():
    # Build a naive datetime series with misaligned entries for 5m
    dates = pd.to_datetime(
        [
            "2024-02-01 00:00:00",
            "2024-02-01 00:03:00",  # misaligned for 5m
            "2024-02-01 00:10:00",
        ]
    )
    df = pd.DataFrame(
        {
            "date": dates,  # tz-naive
            "open": [1, 1, 1],
            "high": [1, 1, 1],
            "low": [1, 1, 1],
            "close": [1, 1, 1],
            "volume": [0, 0, 0],
        }
    )

    norm_df, stats = _normalize_ohlcv_timezone_and_boundary(df, "5m")
    # Should have localized to UTC
    assert str(norm_df["date"].dt.tz) == "UTC"
    # One misaligned row
    assert stats["misaligned"] == 1
    # All rows were tz-naive originally
    assert stats["tz_naive"] == len(df)

