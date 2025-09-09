from panda_backtest.backtest_common.system.interface.base_extension import BaseExtension
import logging

from panda_backtest.extensions.trade_reverse_future.reverse_event_process import ReverseEventProcess
from panda_backtest.extensions.trade_reverse_future.reverse_operation_proxy import ReverseOperationProxy

class FutureTradingExtension(BaseExtension):

    def create(self, _context):
        _context.set_event_process(ReverseEventProcess(_context))
        _context.set_operation_proxy(ReverseOperationProxy(_context))

