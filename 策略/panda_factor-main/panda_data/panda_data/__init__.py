# panda_data/__init__.py
import logging
from typing import Optional, List, Union, Dict, Any

import pandas as pd

from panda_common.config import get_config
from panda_data.factor.factor_reader import FactorReader
from panda_data.factor.crypto_basic_factors import compute_basic_factors_daily
from panda_data.market_data.market_data_reader import MarketDataReader
from panda_data.market_data.market_crypto_reader import MarketCryptoReader
from panda_data.market_data.market_data_reader import MarketDataReader
from panda_data.market_data.market_crypto_minute_reader import MarketCryptoMinReader
from panda_data.market_data.market_stock_cn_minute_reader import MarketStockCnMinReaderV3
from datetime import timezone

_config = None
_factor = None
_market_data = None
_market_crypto_data: MarketCryptoReader = None
_market_crypto_min_data: MarketCryptoMinReader = None
_market_min_data: MarketStockCnMinReaderV3 = None


def init(configPath: Optional[str] = None) -> None:
    """
    Initialize the panda_data package with configuration

    Args:
        config_path: Path to the config file. If None, will use default config from panda_common.config
    """
    global _config, _factor, _market_data, _market_min_data, _market_crypto_data, _market_crypto_min_data

    try:
        # 使用panda_common中的配置
        _config = get_config()

        if not _config:
            raise RuntimeError("Failed to load configuration from panda_common")

        _factor = FactorReader(_config)
        _market_data = MarketDataReader(_config)
        _market_min_data = MarketStockCnMinReaderV3(_config)
        _market_crypto_data = MarketCryptoReader(_config)
        _market_crypto_min_data = MarketCryptoMinReader(_config)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize panda_data: {str(e)}")


def get_all_symbols() -> pd.DataFrame:
    if _market_min_data is None:
        raise RuntimeError("Please call init() before using any functions")
    return _market_min_data.get_all_symbols()


def get_factor(
        factors: Union[str, List[str]],
        start_date: str,
        end_date: str,
        symbols: Optional[Union[str, List[str]]] = None,
        index_component: Optional[str] = None,
        type: Optional[str] = 'stock'
) -> Optional[pd.DataFrame]:
    """
    Get factor data for given symbols and date range

    Args:
        factors: List of factor names to retrieve or single factor name
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        symbols: Optional list of symbols or single symbol. If None, returns all symbols

    Returns:
        pandas DataFrame with factor data, or None if no data found
    """
    if type == 'crypto':
        # 加密路径：直接从 crypto 市场数据计算基础因子（最小实现）
        # 若请求的是基础字段，直接返回 OHLCV；否则计算基础技术因子
        base_fields = {"open", "high", "low", "close", "volume"}
        req = [factors] if isinstance(factors, str) else (factors or [])
        req_lower = set([str(x).lower() for x in req])

        # 获取日线数据
        ohlcv = get_crypto_market_data(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            fields=["date", "symbol", "open", "high", "low", "close", "volume"],
        )
        if ohlcv is None or ohlcv.empty:
            return None

        # 若仅请求基础字段
        if req_lower and req_lower.issubset(base_fields):
            keep = ["date", "symbol"] + [f for f in ["open","high","low","close","volume"] if f in req_lower]
            return ohlcv[keep]

        # 计算基础因子（RSI/MACD/动量/波动等）
        fac = compute_basic_factors_daily(ohlcv, list(req_lower))
        return fac

    # 默认股票/期货路径
    if _factor is None:
        raise RuntimeError("Please call init() before using any functions")
    return _factor.get_factor(symbols, factors, start_date, end_date, index_component, type)


def get_custom_factor(
        factor_logger: logging.Logger,
        user_id: int,
        factor_name: str,
        start_date: str,
        end_date: str,
        symbol_type: Optional[str] = 'stock'
) -> Optional[pd.DataFrame]:
    """
    Get factor data for given symbols and date range

    Args:
        factors: List of factor names to retrieve or single factor name
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        symbols: Optional list of symbols or single symbol. If None, returns all symbols

    Returns:
        pandas DataFrame with factor data, or None if no data found
    """
    if _factor is None:
        raise RuntimeError("Please call init() before using any functions")

    return _factor.get_custom_factor(factor_logger, user_id, factor_name, start_date, end_date, symbol_type)

def get_factor_by_name(factor_name, start_date, end_date):
    if _factor is None:
        raise RuntimeError("Please call init() before using any functions")
    return _factor.get_factor_by_name(factor_name, start_date, end_date)


"""获取所有股票代码"""


def get_stock_instruments() -> pd.DataFrame:
    stocks = _market_min_data.get_stock_instruments()
    return pd.DataFrame(stocks)


def get_market_min_data(
        start_date: str,
        end_date: str,
        symbol: Optional[str] = None,
        fields: Optional[Union[str, List[str]]] = None
) -> Optional[pd.DataFrame]:
    """
    Get market data for given symbols and date range

    Args:
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        symbols: Optional list of symbols or single symbol. If None, returns all symbols
        fields: Optional list of fields to retrieve (e.g., ['open', 'close', 'volume']).
               If None, returns all available fields

    Returns:
        pandas DataFrame with market data, or None if no data found
    """
    if _market_min_data is None:
        raise RuntimeError("Please call init() before using any functions")

    return _market_min_data.get_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        fields=fields
    )


def get_market_data(
        start_date: str,
        end_date: str,
        indicator="000985",
        st=True,
        symbols: Optional[Union[str, List[str]]] = None,
        fields: Optional[Union[str, List[str]]] = None,
        type: Optional[str] = 'stock'
) -> Optional[pd.DataFrame]:
    """
    Get market data for given symbols and date range

    Args:
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        symbols: Optional list of symbols or single symbol. If None, returns all symbols
        fields: Optional list of fields to retrieve (e.g., ['open', 'close', 'volume']).
               If None, returns all available fields

    Returns:
        pandas DataFrame with market data, or None if no data found
    """
    if _market_data is None:
        raise RuntimeError("Please call init() before using any functions")

    return _market_data.get_market_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        indicator=indicator,
        st=st,
        fields=fields,
        type=type
    )


def get_crypto_market_data(
        start_date: str,
        end_date: str,
        symbols: Optional[Union[str, List[str]]] = None,
        fields: Optional[Union[str, List[str]]] = None,
) -> Optional[pd.DataFrame]:
    """
    Get crypto market OHLCV for given symbols and date range.

    Notes:
        - Assumes Mongo collection "crypto_market" with fields at least
          ['date','symbol','open','high','low','close','volume'].
        - Dates are inclusive in YYYYMMDD.
    """
    if _market_crypto_data is None:
        raise RuntimeError("Please call init() before using any functions")

    return _market_crypto_data.get_market_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        fields=fields
    )


def get_crypto_min_data(
        start_date: str,
        end_date: str,
        symbol: Optional[str] = None,
        fields: Optional[Union[str, List[str]]] = None
) -> Optional[pd.DataFrame]:
    """
    获取加密货币分钟级行情（按自然日范围，UTC）。

    参数:
        start_date/end_date: YYYYMMDD（含首尾日）
        symbol: 例如 BINANCE:BTCUSDT
        fields: 需要的列名；不填返回全部
    """
    if _market_crypto_min_data is None:
        raise RuntimeError("Please call init() before using any functions")

    return _market_crypto_min_data.get_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        fields=fields
    )


def get_available_market_fields() -> List[str]:
    """
    Get all available fields in the stock_market collection

    Returns:
        List of available field names
    """
    if _market_data is None:
        raise RuntimeError("Please call init() before using any functions")

    return _market_data.get_available_fields()

# Add more public functions as needed


def get_crypto_instruments(collection: str = "crypto_market") -> List[str]:
    """
    列出 Mongo 中已入库的加密货币符号（默认从日线集合 crypto_market 读取）。

    Args:
        collection: 集合名，默认 "crypto_market"；可切换为 "crypto_market_1m"。

    Returns:
        符号列表，例如 ["BINANCE:BTCUSDT", "BINANCE:ETHUSDT", ...]
    """
    if _market_crypto_data is None:
        raise RuntimeError("Please call init() before using any functions")

    coll = _market_crypto_data.db_handler.get_mongo_collection(_config["MONGO_DB"], collection)
    try:
        return sorted(coll.distinct("symbol"))
    except Exception:
        return []


def get_crypto_min_data_resampled(
        start_date: str,
        end_date: str,
        symbol: str,
        rule: str = "5min",
        closed: str = "right",
        label: str = "right",
        tz_aware: bool = True,
        fields: Optional[Union[str, List[str]]] = None
) -> Optional[pd.DataFrame]:
    """
    按分钟数据重采样为更粗粒度（如 5min/15min/1h/1d）。

    约定：
    - 输入数据列包含 ['datetime','open','high','low','close','volume']，UTC 时间。
    - 聚合：open=first, high=max, low=min, close=last, volume=sum。

    Args:
        start_date/end_date: YYYYMMDD 包含端点
        symbol: 例如 BINANCE:BTCUSDT
        rule: pandas 频率字符串，如 '5min','15min','1h','1d'
        closed/label: 重采样区间闭合与标签，默认右闭右标
        tz_aware: True 则确保 datetime 为 timezone-aware(UTC)
        fields: 需要的列名，默认全部

    Returns:
        DataFrame，包含 ['datetime','symbol','open','high','low','close','volume']
    """
    df = get_crypto_min_data(start_date, end_date, symbol=symbol, fields=fields)
    if df is None or df.empty:
        return None

    df = df.copy()
    # 确保 UTC aware
    dt = pd.to_datetime(df['datetime'])
    if tz_aware:
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize("UTC")
    df['datetime'] = dt

    df = df.set_index('datetime')
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    out = df.resample(rule, closed=closed, label=label).agg(agg).dropna(how='any')
    out = out.reset_index()
    out['symbol'] = symbol
    cols = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    return out[cols]
