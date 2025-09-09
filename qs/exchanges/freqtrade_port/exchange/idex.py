from __future__ import annotations
"""Idex exchange subclass"""

import logging

from qs.exchanges.freqtrade_port.exchange import Exchange
from qs.exchanges.freqtrade_port.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Idex(Exchange):
    """
    Idex exchange class. Contains adjustments needed for Freqtrade to work
    with this exchange.
    """

    _ft_has: FtHas = {
        "ohlcv_candle_limit": 1000,
    }
