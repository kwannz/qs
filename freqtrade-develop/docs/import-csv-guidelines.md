## Importing Third‑Party CSV into Historical Folder

This document outlines recommended formats when importing OHLCV/trades data from third‑party sources into your historical folder for local backtesting.

### OHLCV format

- Required columns: `date`, `open`, `high`, `low`, `close`, `volume`
- Timestamp: UTC (timezone aware or naive UTC). For naive timestamps, the importer assumes UTC.
- Boundary: Candle timestamps must align to the timeframe boundary (e.g., 5m candles at mm=00/05/10/...).
- Types: Numeric columns should be floats; `date` should be ISO datetime or epoch seconds converted to datetime

Example (pandas conversion):

```python
import pandas as pd

df = pd.read_csv("thirdparty_5m.csv")
# Convert epoch seconds -> UTC datetime
df["date"] = pd.to_datetime(df["date"], unit="s", utc=True)
df = df[["date", "open", "high", "low", "close", "volume"]]
df.to_csv("normalized_5m.csv", index=False)
```

### Trades format

- Typical columns: `timestamp` (UTC), `price`, `amount`, `side` (buy/sell)
- You can import trades and then convert to OHLCV via `trades-to-ohlcv`.

### Directory & Filenames

- Freqtrade data handlers search for files by pair/timeframe.
- Recommended naming: `PAIR-TF.ext` (e.g., `BTC_USDT-5m.feather`) within the exchange folder.
- Use Freqtrade's converters to migrate formats:

```
freqtrade convert-data \
  --config user_data/config.json \
  --datadir "historical data" \
  --format-from json \
  --format-to feather \
  --timeframes 5m 1h \
  --erase
```

### Common pitfalls

- Non‑UTC timestamps → convert to UTC
- Misaligned candle timestamps → resample with pandas before import
- Missing columns or wrong types → rename/cast columns to required names and types

### Validation

Use the `data-quality` subcommand to validate and fix issues:

```
freqtrade data-quality \
  --config user_data/config.json \
  --datadir "historical data" \
  --pairs BTC/USDT \
  --timeframes 5m 1h \
  --timerange 20240101-20240701 \
  --fix
```

