from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class FuturesLeg:
    symbol: str
    amount: float = 0.0  # >0 long, <0 short (in contracts or units)
    avg_price: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class FuturesPosition:
    """
    Minimal futures position model for backtest (long/short + margin).

    - Positive amount: long; Negative amount: short
    - PnL = (price - avg_price) * amount * contract_multiplier (note sign)
    - Equity = cash + sum(unrealized) - fee_accum - funding_accum
    - Margin: supports either leverage or IM/MM pair (use one of them)
    """

    contract_multiplier: float = 1.0
    im: Optional[float] = None  # initial margin ratio, e.g. 0.1
    mm: Optional[float] = None  # maintenance margin ratio, e.g. 0.05
    leverage: Optional[float] = None  # alternative to im/mm

    cash: float = 0.0
    legs: Dict[str, FuturesLeg] = field(default_factory=dict)
    fee_accum: float = 0.0
    funding_accum: float = 0.0

    def _ensure_leg(self, symbol: str) -> FuturesLeg:
        leg = self.legs.get(symbol)
        if leg is None:
            leg = FuturesLeg(symbol=symbol)
            self.legs[symbol] = leg
        return leg

    def _notional(self, price: float, amount: float) -> float:
        return abs(price * amount * self.contract_multiplier)

    def _required_im(self, price: float, amount: float) -> float:
        if self.leverage and self.leverage > 0:
            return self._notional(price, amount) / self.leverage
        if self.im and self.im > 0:
            return self._notional(price, amount) * self.im
        # fallback: no margin constraint
        return 0.0

    def update_order(self, *, symbol: str, trade_price: float, trade_amount: float, fee: float) -> None:
        """
        Apply a trade to the position.

        - trade_amount > 0: buy/open long or buy to close short (increase amount)
        - trade_amount < 0: sell/open short or sell to close long (decrease amount)
        - fee charged in cash immediately
        """
        leg = self._ensure_leg(symbol)
        amt0, px0 = leg.amount, leg.avg_price
        amt1 = amt0 + trade_amount

        # Realized PnL if position cross zero or reduce
        realized = 0.0
        if amt0 == 0 or (amt0 > 0 and trade_amount > 0) or (amt0 < 0 and trade_amount < 0):
            # Increasing same direction: average price update only
            pass
        else:
            # Reducing or flipping
            reduce_amt = -trade_amount if abs(trade_amount) < abs(amt0) else abs(amt0)
            # For long reduced by sell (trade_amount<0): pnl = (trade_price - avg_price)*reduced
            # For short reduced by buy (trade_amount>0): pnl = (avg_price - trade_price)*reduced
            if amt0 > 0 and trade_amount < 0:
                realized = (trade_price - px0) * reduce_amt * self.contract_multiplier
            elif amt0 < 0 and trade_amount > 0:
                realized = (px0 - trade_price) * reduce_amt * self.contract_multiplier
            if abs(amt1) < 1e-12:
                amt1 = 0.0

        # Update avg price
        if amt1 == 0:
            leg.avg_price = 0.0
        else:
            if (amt0 >= 0 and trade_amount >= 0) or (amt0 <= 0 and trade_amount <= 0):
                # Same direction add
                total_notional0 = abs(px0 * amt0)
                total_notional1 = abs(trade_price * trade_amount)
                denom = abs(amt0 + trade_amount)
                leg.avg_price = (total_notional0 + total_notional1) / denom if denom != 0 else 0.0
            else:
                # Reduce: avg price unchanged, unless flip (when amt1 and amt0 opposite sign)
                if amt0 * amt1 < 0:  # flipped
                    # New avg price is entry price of flipped remainder
                    leg.avg_price = trade_price

        # Update amount & cash/fee
        leg.amount = amt1
        leg.realized_pnl += realized
        self.cash += realized
        self.fee_accum += fee
        self.cash -= fee

    def unrealized_pnl(self, price_map: Dict[str, float]) -> float:
        pnl = 0.0
        for sym, leg in self.legs.items():
            if leg.amount == 0:
                continue
            px = price_map.get(sym)
            if px is None:
                continue
            # long: (px - avg)*amount; short: (avg - px)*abs(amount)
            pnl += (px - leg.avg_price) * leg.amount * self.contract_multiplier
        return pnl

    def equity(self, price_map: Dict[str, float]) -> float:
        return self.cash + self.unrealized_pnl(price_map) - self.fee_accum - self.funding_accum

    def margin_used(self, price_map: Dict[str, float]) -> float:
        total = 0.0
        for sym, leg in self.legs.items():
            px = price_map.get(sym)
            if px is None or leg.amount == 0:
                continue
            total += self._required_im(px, leg.amount)
        return total

    def check_liquidation(self, price_map: Dict[str, float]) -> bool:
        """Return True if equity < maintenance margin requirement."""
        if self.mm is None and (self.leverage is None or self.leverage <= 0):
            return False
        if self.mm is None and self.leverage:
            # derive mm as im/2 heuristically if not set
            mm = (self.im or (1.0 / self.leverage)) * 0.5
        else:
            mm = self.mm or 0.05
        mm_required = 0.0
        for sym, leg in self.legs.items():
            px = price_map.get(sym)
            if px is None or leg.amount == 0:
                continue
            mm_required += abs(px * leg.amount * self.contract_multiplier) * mm
        return self.equity(price_map) < mm_required

    def apply_funding(self, price_map: Dict[str, float], funding_rate: float) -> float:
        """
        Apply funding payment (long pays short if rate>0 for perpetuals, simplified):
        funding = rate * notional (sign depends on side)
        """
        charge = 0.0
        for sym, leg in self.legs.items():
            px = price_map.get(sym)
            if px is None or leg.amount == 0:
                continue
            notional = px * abs(leg.amount) * self.contract_multiplier
            # Convention: if rate>0, longs pay shorts
            if leg.amount > 0:
                c = funding_rate * notional
            else:
                c = -funding_rate * notional
            charge += c
        self.funding_accum += max(0.0, charge)
        self.cash -= max(0.0, charge)
        # if shorts receive, add to cash
        if charge < 0:
            self.cash += (-charge)
        return charge

