from copy import deepcopy

from freqtrade.exchange.exchange import Exchange
from tests.conftest import get_patched_exchange


def test_detect_market_changes_amount_and_cost(mocker, default_conf, markets):
    ex: Exchange = get_patched_exchange(mocker, default_conf, mock_markets=markets)
    # Seed initial markets
    initial = deepcopy(ex.markets)
    # Modify precision and limits for one symbol
    modified = deepcopy(initial)
    sym = "ETH/BTC"
    modified[sym]["precision"]["amount"] = 6
    modified[sym]["limits"]["cost"]["min"] = 0.0002

    changes = ex._detect_market_changes(initial, modified)
    assert any(c["symbol"] == sym for c in changes)
    change_entry = next(c for c in changes if c["symbol"] == sym)
    ch = change_entry["changes"]
    assert ch.get("precision.amount")["old"] == 8
    assert ch.get("precision.amount")["new"] == 6
    assert ch.get("limits.cost.min")["old"] == 0.0001
    assert ch.get("limits.cost.min")["new"] == 0.0002

