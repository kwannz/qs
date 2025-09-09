import dataclasses


@dataclasses.dataclass
class TradeCollections():
    FUTURE_ACCOUNT = "panda_future_account"
    REAL_TRADE_STRATEGY = "real_trad_strategy_server"
    REAL_TRADE_BINDING = "real_trade_binding"