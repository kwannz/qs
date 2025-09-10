from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .position_futures import FuturesPosition


@dataclass
class FuturesExchange:
    """
    Minimal futures exchange for backtest (supports long/short, fee, slippage, funding).
    Not integrated into qlib.executor yet; can be used standalone or via adapter.
    """

    maker_bps: float = 0.0
    taker_bps: float = 5.0
    slippage_bps: float = 2.0
    contract_multiplier: float = 1.0
    im: Optional[float] = None
    mm: Optional[float] = None
    leverage: Optional[float] = None
    trade_unit: Optional[float] = None  # min order size
    min_notional: Optional[float] = None

    funding_rate: float = 0.0

    def _slip_price(self, price: float, side: int) -> float:
        # side: +1 buy, -1 sell
        slip = self.slippage_bps / 10000.0
        return price * (1.0 + slip if side > 0 else 1.0 - slip)

    def _fee(self, price: float, qty: float, taker: bool = True) -> float:
        bps = (self.taker_bps if taker else self.maker_bps) / 10000.0
        return abs(price * qty * self.contract_multiplier) * bps

    def _check_unit(self, qty: float) -> float:
        if self.trade_unit and self.trade_unit > 0:
            steps = round(qty / self.trade_unit)
            return steps * self.trade_unit
        return qty

    def deal(self, *, position: FuturesPosition, symbol: str, side: int, qty: float, price: float,
             taker: bool = True, price_map: Optional[Dict[str, float]] = None) -> None:
        """
        Execute a trade on the position.
        - side: +1 buy (increase), -1 sell (decrease)
        - qty: positive quantity in contracts (or units)
        """
        if qty <= 0:
            return
        qty = self._check_unit(qty)
        if qty == 0:
            return
        # Slippage-adjusted trade price
        trade_px = self._slip_price(price, side)
        trade_amt = qty if side > 0 else -qty
        # Fee as taker or maker
        fee = self._fee(trade_px, qty, taker=taker)

        # Optional margin check (IM)
        if price_map is None:
            price_map = {symbol: price}
        else:
            price_map = dict(price_map)
            price_map[symbol] = price

        # Pre-check margin if opening in same direction
        if position.contract_multiplier != self.contract_multiplier:
            position.contract_multiplier = self.contract_multiplier

        position.update_order(symbol=symbol, trade_price=trade_px, trade_amount=trade_amt, fee=fee)

        # Optional liquidation check
        if position.check_liquidation(price_map):
            # Simplified liquidation: close all
            # Real implementation may partial close to reach MM
            for sym, leg in list(position.legs.items()):
                if leg.amount == 0:
                    continue
                px = price_map.get(sym, trade_px)
                fee_liq = self._fee(px, abs(leg.amount), taker=True)
                position.update_order(symbol=sym, trade_price=px, trade_amount=-leg.amount, fee=fee_liq)

    def settle_funding(self, position: FuturesPosition, price_map: Dict[str, float]) -> float:
        return position.apply_funding(price_map, self.funding_rate)

