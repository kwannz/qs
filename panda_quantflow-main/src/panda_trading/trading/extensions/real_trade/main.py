from panda_backtest.backtest_common.system.interface.base_extension import BaseExtension
from panda_trading.trading.extensions.real_trade.reverse_event_process import ReverseEventProcess
from panda_trading.trading.extensions.real_trade.reverse_operation_proxy import ReverseOperationProxy


class TradingExtension(BaseExtension):
    def create(self, _context):
        _context.set_event_process(ReverseEventProcess(_context))
        _context.set_operation_proxy(ReverseOperationProxy(_context))


