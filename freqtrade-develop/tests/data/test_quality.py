import pandas as pd

from freqtrade.data.quality import detect_ohlcv_gaps_df


def test_detect_ohlcv_gaps_df_basic():
    # Build a 5m series with a gap and a duplicate
    base = pd.to_datetime(
        [
            "2024-01-01 00:00:00",
            "2024-01-01 00:05:00",
            "2024-01-01 00:10:00",
            "2024-01-01 00:20:00",  # gap of 5m (one missing candle between 00:10 and 00:20)
            "2024-01-01 00:20:00",  # duplicate timestamp
        ]
    )
    df = pd.DataFrame(
        {
            "date": base,
            "open": [1, 2, 3, 4, 4],
            "high": [1, 2, 3, 4, 4],
            "low": [1, 2, 3, 4, 4],
            "close": [1, 2, 3, 4, 4],
            "volume": [10, 10, 10, 10, 10],
        }
    )
    res = detect_ohlcv_gaps_df(df, "5m")
    assert res["duplicate_count"] >= 1
    assert len(res["gaps"]) == 1
    gap = res["gaps"][0]
    # One missing candle (15min delta -> missing 2? Wait: 00:10 -> 00:20 is 600s, step=300 -> missing 1)
    assert gap["missing_candles"] == 1

