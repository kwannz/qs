usage: freqtrade data-quality [-h] [-v] [--no-color] [--logfile FILE] [-V]
                              [--config PATH] [--datadir PATH]
                              [--userdir PATH] [--exchange EXCHANGE]
                              [--data-format-ohlcv {json,jsongz,feather,parquet}]
                              [--data-format-trades {json,jsongz,feather,parquet}]
                              [--trades] [--pairs PAIR [PAIR ...]]
                              [--trading-mode {spot,margin,futures}] [--timeframes TF [TF ...]]
                              [--timerange RANGE] [--fix] [--jobs INT]
                              [--print-json] [--fill-gaps]

Scan local OHLCV/Trades data for duplicates and gaps, optionally remove duplicates.
Supports limiting the scan to a specific timerange and filling detected OHLCV gaps.

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Verbose mode (-vv for more, -vvv to get all messages).
  --no-color            Disable colorization of output.
  --logfile FILE, --log-file FILE
                        Log to the file specified.
  -V, --version         show program's version number and exit
  -c PATH, --config PATH
                        Specify configuration file.
  -d PATH, --datadir PATH, --data-dir PATH
                        Path to base directory containing historical data.
  --userdir PATH, --user-data-dir PATH
                        Path to userdata directory.
  --exchange EXCHANGE   Exchange name. Only valid if no config is provided.
  --data-format-ohlcv {json,jsongz,feather,parquet}
                        Storage format for OHLCV data.
  --data-format-trades {json,jsongz,feather,parquet}
                        Storage format for Trades data.
  --trades              Work on Trades data (default: OHLCV).
  -p PAIR [PAIR ...], --pairs PAIR [PAIR ...]
                        Limit command to these pairs.
  --trading-mode {spot,margin,futures}
                        Select Trading mode
  -t TF [TF ...], --timeframes TF [TF ...]
                        Timeframes to scan (OHLCV only).
  --timerange RANGE     Limit scan to a specific timerange (e.g. 20240101-20240201).
  --fix                 Attempt to fix detected issues (currently: deduplicate only).
  -j INT, --jobs INT    Number of parallel jobs to use.
  --print-json          Print report in JSON format.
  --fill-gaps           Attempt to fill OHLCV gaps (requires --exchange).

Examples:

Scan OHLCV for duplicates/gaps on 5m and 1h:

    freqtrade data-quality --timeframes 5m 1h

Scan and deduplicate OHLCV for specific pairs:

    freqtrade data-quality --pairs BTC/USDT ETH/USDT -t 5m --fix

Scan and deduplicate Trades:

    freqtrade data-quality --trades --pairs XRP/ETH --fix
