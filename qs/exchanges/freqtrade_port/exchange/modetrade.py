from __future__ import annotations
import logging

# from qs.exchanges.freqtrade_port.enums import MarginMode, TradingMode
from qs.exchanges.freqtrade_port.exchange import Exchange
from qs.exchanges.freqtrade_port.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Modetrade(Exchange):
    """
    MOdetrade exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.

    Please note that this exchange is not included in the list of exchanges
    officially supported by the Freqtrade development team. So some features
    may still not work as expected.
    """

    _ft_has: FtHas = {
        "always_require_api_keys": True,  # Requires API keys to fetch candles
    }

    # _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
    #     (TradingMode.FUTURES, MarginMode.ISOLATED),
    # ]
