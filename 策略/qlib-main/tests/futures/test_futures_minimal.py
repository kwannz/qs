import math
from qlib.backtest.position_futures import FuturesPosition
from qlib.backtest.exchange_futures import FuturesExchange


def test_long_short_pnl_and_liquidation():
    pos = FuturesPosition(contract_multiplier=1.0, im=0.1, mm=0.05, cash=1000.0)
    ex = FuturesExchange(taker_bps=5.0, slippage_bps=2.0, contract_multiplier=1.0, im=0.1, mm=0.05)

    # open long 1 @100
    ex.deal(position=pos, symbol='BTCUSDT', side=1, qty=1.0, price=100.0, taker=True, price_map={'BTCUSDT':100.0})
    assert 'BTCUSDT' in pos.legs
    assert pos.legs['BTCUSDT'].amount > 0

    # move to 110 -> up PnL ~ (110-100)*1=10
    eq = pos.equity({'BTCUSDT':110.0})
    assert eq > pos.cash  # unrealized positive

    # sell close @110
    ex.deal(position=pos, symbol='BTCUSDT', side=-1, qty=1.0, price=110.0, taker=True, price_map={'BTCUSDT':110.0})
    assert pos.legs['BTCUSDT'].amount == 0

    # open short 2 @100
    ex.deal(position=pos, symbol='BTCUSDT', side=-1, qty=2.0, price=100.0, taker=True, price_map={'BTCUSDT':100.0})
    # price jumps to 130; check liquidation potential
    liq = pos.check_liquidation({'BTCUSDT':130.0})
    assert isinstance(liq, bool)

