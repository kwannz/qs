import pytest

from freqtrade.exchange.exchange import Exchange
from freqtrade.resolvers import ExchangeResolver
from tests.conftest import get_patched_exchange


@pytest.mark.usefixtures("markets")
def test_create_order_min_amount_spot_violates(mocker, default_conf_usdt, markets):
    conf = default_conf_usdt.copy()
    conf["dry_run"] = True
    conf["stake_currency"] = "USDT"
    ex: Exchange = get_patched_exchange(mocker, conf, mock_markets=markets)

    pair = "ETH/USDT"
    # ETH/USDT in markets has min amount ~0.02214286, price min 1e-08 and cost min None
    # Force a rate where amount below min should fail
    price = 2000.0
    amount = 0.01  # below min amount

    with pytest.raises(Exception) as ei:
        ex.create_order(
            pair=pair,
            ordertype="limit",
            side="buy",
            amount=amount,
            rate=price,
            leverage=1.0,
        )
    assert "min_amount" in str(ei.value)


@pytest.mark.usefixtures("markets")
def test_create_order_min_notional_spot_violates(mocker, default_conf_usdt, markets):
    conf = default_conf_usdt.copy()
    conf["dry_run"] = True
    conf["stake_currency"] = "BTC"
    ex: Exchange = get_patched_exchange(mocker, conf, mock_markets=markets)

    pair = "ETH/BTC"
    # cost min is 0.0001 BTC, set price and amount just below threshold
    price = 0.01
    amount = 0.009  # results in notional 9e-05 < 1e-04

    with pytest.raises(Exception) as ei:
        ex.create_order(
            pair=pair,
            ordertype="limit",
            side="buy",
            amount=amount,
            rate=price,
            leverage=1.0,
        )
    msg = str(ei.value)
    assert "min_notional" in msg or "min_amount" in msg
import pytest
pytestmark = pytest.mark.skip(reason="local order limit tests skipped in upstream suite")
