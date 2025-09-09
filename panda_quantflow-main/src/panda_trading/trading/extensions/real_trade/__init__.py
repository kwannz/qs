def load_extension():
    from panda_trading.trading.extensions.real_trade.main import TradingExtension
    return TradingExtension()