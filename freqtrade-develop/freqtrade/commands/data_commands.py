import logging
import sys
from collections import defaultdict
from typing import Any

from freqtrade.constants import DATETIME_PRINT_FORMAT, DL_DATA_TIMEFRAMES, Config
from freqtrade.enums import CandleType, RunMode, TradingMode
from freqtrade.exceptions import ConfigurationError
from freqtrade.plugins.pairlist.pairlist_helpers import dynamic_expand_pairlist, expand_pairlist


logger = logging.getLogger(__name__)


def _check_data_config_download_sanity(config: Config) -> None:
    if "days" in config and "timerange" in config:
        raise ConfigurationError(
            "--days and --timerange are mutually exclusive. You can only specify one or the other."
        )

    if "pairs" not in config:
        raise ConfigurationError(
            "Downloading data requires a list of pairs. "
            "Please check the documentation on how to configure this."
        )


def start_download_data(args: dict[str, Any]) -> None:
    """
    Download data (former download_backtest_data.py script)
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.data.history import download_data_main

    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)

    _check_data_config_download_sanity(config)

    try:
        download_data_main(config)

    except KeyboardInterrupt:
        sys.exit("SIGINT received, aborting ...")


def start_convert_trades(args: dict[str, Any]) -> None:
    from freqtrade.configuration import TimeRange, setup_utils_configuration
    from freqtrade.data.converter import convert_trades_to_ohlcv
    from freqtrade.resolvers import ExchangeResolver

    config = setup_utils_configuration(args, RunMode.UTIL_EXCHANGE)

    timerange = TimeRange()

    # Remove stake-currency to skip checks which are not relevant for datadownload
    config["stake_currency"] = ""

    if "timeframes" not in config:
        config["timeframes"] = DL_DATA_TIMEFRAMES

    # Init exchange
    exchange = ExchangeResolver.load_exchange(config, validate=False)
    # Manual validations of relevant settings

    for timeframe in config["timeframes"]:
        exchange.validate_timeframes(timeframe)
    available_pairs = [
        p
        for p in exchange.get_markets(
            tradable_only=True, active_only=not config.get("include_inactive")
        ).keys()
    ]

    expanded_pairs = dynamic_expand_pairlist(config, available_pairs)

    # Convert downloaded trade data to different timeframes
    convert_trades_to_ohlcv(
        pairs=expanded_pairs,
        timeframes=config["timeframes"],
        datadir=config["datadir"],
        timerange=timerange,
        erase=bool(config.get("erase")),
        data_format_ohlcv=config["dataformat_ohlcv"],
        data_format_trades=config["dataformat_trades"],
        candle_type=config.get("candle_type_def", CandleType.SPOT),
    )


def start_convert_data(args: dict[str, Any], ohlcv: bool = True) -> None:
    """
    Convert data from one format to another
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.data.converter import convert_ohlcv_format, convert_trades_format
    from freqtrade.util.migrations import migrate_data

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)
    if ohlcv:
        migrate_data(config)
        convert_ohlcv_format(
            config,
            convert_from=args["format_from"],
            convert_to=args["format_to"],
            erase=args["erase"],
        )
    else:
        convert_trades_format(
            config,
            convert_from=args["format_from_trades"],
            convert_to=args["format_to"],
            erase=args["erase"],
        )


def start_list_data(args: dict[str, Any]) -> None:
    """
    List available OHLCV data
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.exchange import timeframe_to_minutes
    from freqtrade.util import print_rich_table

    if args["trades"]:
        start_list_trades_data(args)
        return

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    from freqtrade.data.history import get_datahandler

    dhc = get_datahandler(config["datadir"], config["dataformat_ohlcv"])

    paircombs = dhc.ohlcv_get_available_data(
        config["datadir"], config.get("trading_mode", TradingMode.SPOT)
    )
    if args["pairs"]:
        pl = expand_pairlist(args["pairs"], [p[0] for p in paircombs], keep_invalid=True)
        paircombs = [comb for comb in paircombs if comb[0] in pl]
    title = f"Found {len(paircombs)} pair / timeframe combinations."
    if not config.get("show_timerange"):
        groupedpair = defaultdict(list)
        for pair, timeframe, candle_type in sorted(
            paircombs, key=lambda x: (x[0], timeframe_to_minutes(x[1]), x[2])
        ):
            groupedpair[(pair, candle_type)].append(timeframe)

        if groupedpair:
            print_rich_table(
                [
                    (pair, ", ".join(timeframes), candle_type)
                    for (pair, candle_type), timeframes in groupedpair.items()
                ],
                ("Pair", "Timeframe", "Type"),
                title,
                table_kwargs={"min_width": 50},
            )
    else:
        paircombs1 = [
            (pair, timeframe, candle_type, *dhc.ohlcv_data_min_max(pair, timeframe, candle_type))
            for pair, timeframe, candle_type in paircombs
        ]
        print_rich_table(
            [
                (
                    pair,
                    timeframe,
                    candle_type,
                    start.strftime(DATETIME_PRINT_FORMAT),
                    end.strftime(DATETIME_PRINT_FORMAT),
                    str(length),
                )
                for pair, timeframe, candle_type, start, end, length in sorted(
                    paircombs1, key=lambda x: (x[0], timeframe_to_minutes(x[1]), x[2])
                )
            ],
            ("Pair", "Timeframe", "Type", "From", "To", "Candles"),
            summary=title,
            table_kwargs={"min_width": 50},
        )


def start_list_trades_data(args: dict[str, Any]) -> None:
    """
    List available Trades data
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.misc import plural
    from freqtrade.util import print_rich_table

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    from freqtrade.data.history import get_datahandler

    dhc = get_datahandler(config["datadir"], config["dataformat_trades"])

    paircombs = dhc.trades_get_available_data(
        config["datadir"], config.get("trading_mode", TradingMode.SPOT)
    )

    if args["pairs"]:
        pl = expand_pairlist(args["pairs"], [p for p in paircombs], keep_invalid=True)
        paircombs = [comb for comb in paircombs if comb in pl]

    title = f"Found trades data for {len(paircombs)} {plural(len(paircombs), 'pair')}."
    if not config.get("show_timerange"):
        print_rich_table(
            [(pair, config.get("candle_type_def", CandleType.SPOT)) for pair in sorted(paircombs)],
            ("Pair", "Type"),
            title,
            table_kwargs={"min_width": 50},
        )
    else:
        paircombs1 = [
            (pair, *dhc.trades_data_min_max(pair, config.get("trading_mode", TradingMode.SPOT)))
            for pair in paircombs
        ]
        print_rich_table(
            [
                (
                    pair,
                    config.get("candle_type_def", CandleType.SPOT),
                    start.strftime(DATETIME_PRINT_FORMAT),
                    end.strftime(DATETIME_PRINT_FORMAT),
                    str(length),
                )
                for pair, start, end, length in sorted(paircombs1, key=lambda x: (x[0]))
            ],
            ("Pair", "Type", "From", "To", "Trades"),
            summary=title,
            table_kwargs={"min_width": 50},
        )


def start_data_quality(args: dict[str, Any]) -> None:
    """
    Scan local data for gaps and duplicates. Optionally deduplicate.
    """
    from freqtrade.configuration import setup_utils_configuration
    from freqtrade.data.quality import scan_ohlcv_quality, scan_trades_quality
    from freqtrade.util import print_rich_table

    config = setup_utils_configuration(args, RunMode.UTIL_NO_EXCHANGE)

    pairs = args.get("pairs")
    fix = bool(args.get("fix"))
    jobs = args.get("jobs")
    if args.get("trades"):
        # Trades quality
        report = scan_trades_quality(
            config["datadir"],
            config["dataformat_trades"],
            config.get("trading_mode", TradingMode.SPOT),
            pairs=pairs,
            fix=fix,
            jobs=jobs,
        )
        if args.get("print_json"):
            import orjson

            print(orjson.dumps(report).decode())
        else:
            rows = [
                (pair, str(item.get("duplicate_count", 0))) for pair, item in sorted(report.items())
            ]
            print_rich_table(rows, ("Pair", "Duplicates"), "Trades quality report")
    else:
        # OHLCV quality
        timeframes = args.get("timeframes")
        report = scan_ohlcv_quality(
            config["datadir"],
            config["dataformat_ohlcv"],
            config.get("trading_mode", TradingMode.SPOT),
            pairs=pairs,
            timeframes=timeframes,
            fix=fix,
            jobs=jobs,
            timerange=args.get("timerange"),
        )
        # Optional gap filling (requires exchange)
        if args.get("fill_gaps"):
            if not args.get("exchange") and not config.get("exchange", {}).get("name"):
                logger.error("--fill-gaps requires --exchange or exchange.name in config.")
            else:
                # Build gaps list
                gaps_to_fill: dict[tuple[str, str], list[tuple[int, int]]] = {}
                for p, tf_dict in report.items():
                    for tf, res in tf_dict.items():
                        for gap in res.get("gaps", []):
                            gaps_to_fill.setdefault((p, tf), []).append((gap["start"], gap["end"]))
                if gaps_to_fill:
                    from freqtrade.resolvers.exchange_resolver import ExchangeResolver
                    from freqtrade.data.history.datahandlers import get_datahandler
                    from freqtrade.data.converter import clean_ohlcv_dataframe
                    from freqtrade.enums import CandleType
                    import pandas as pd

                    exchange = ExchangeResolver.load_exchange(config, validate=False)
                    dh = get_datahandler(config["datadir"], config["dataformat_ohlcv"])
                    candle_type = CandleType.get_default(config.get("trading_mode", TradingMode.SPOT))
                    for (p, tf), ranges in gaps_to_fill.items():
                        # Merge ranges (just iterate; fetch each range to avoid huge downloads)
                        for start_s, end_s in ranges:
                            try:
                                df_new = exchange.get_historic_ohlcv(
                                    pair=p,
                                    timeframe=tf,
                                    since_ms=(start_s + 1) * 1000,
                                    is_new_pair=False,
                                    candle_type=candle_type,
                                    until_ms=end_s * 1000,
                                )
                                df_old = dh.ohlcv_load(
                                    p,
                                    timeframe=tf,
                                    timerange=None,
                                    fill_missing=False,
                                    drop_incomplete=False,
                                    warn_no_data=False,
                                    candle_type=candle_type,
                                )
                                if df_new is not None and not df_new.empty:
                                    df_merged = clean_ohlcv_dataframe(
                                        pd.concat([df_old, df_new], axis=0),
                                        tf,
                                        p,
                                        fill_missing=False,
                                        drop_incomplete=False,
                                    )
                                    dh.ohlcv_store(p, tf, data=df_merged, candle_type=candle_type)
                                    logger.info(
                                        f"Filled gap for {p} {tf}: {start_s} - {end_s} ({len(df_new)} candles)"
                                    )
                            except Exception as e:
                                logger.warning(f"Failed to fill gap for {p} {tf}: {e}")
        if args.get("print_json"):
            import orjson

            print(orjson.dumps(report).decode())
        else:
            # Flatten for display
            rows = []
            for pair in sorted(report.keys()):
                for tf, res in sorted(report[pair].items()):
                    gap_count = len(res.get("gaps", []))
                    dups = res.get("duplicate_count", 0)
                    rows.append((pair, tf, str(dups), str(gap_count)))
            print_rich_table(
                rows, ("Pair", "Timeframe", "Duplicates", "Gaps"), "OHLCV quality report"
            )
