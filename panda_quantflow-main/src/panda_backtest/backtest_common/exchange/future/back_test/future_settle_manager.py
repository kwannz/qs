from collections import defaultdict
import logging

from panda_backtest.backtest_common.system.event.event import ConstantEvent, Event

from panda_backtest.backtest_common.system.context.core_context import CoreContext

class FutureSettleManager(object):
    def __init__(self):
        self.settle_dict = defaultdict(set)
        self.context = CoreContext.get_instance()

    def add_settle_future(self, future_symbol, settle_date):
        self.settle_dict[settle_date].add(future_symbol)

    def handle_settle(self):
        strategy_context = self.context.strategy_context
        event_bus = self.context.event_bus

        for future_account in strategy_context.future_account_dict.values():
            if future_account.total_value <= 0:
                event = Event(ConstantEvent.SYSTEM_FUTURE_BURNED, account=future_account.account)
                event_bus.publish_event(event)

        if strategy_context.trade_date in self.settle_dict.keys():
            for future_symbol in self.settle_dict[strategy_context.trade_date]:
                event = Event(ConstantEvent.SYSTEM_FUTURE_DELIVERY, future_symbol=future_symbol)
                event_bus.publish_event(event)
            del self.settle_dict[strategy_context.trade_date]

        event = Event(ConstantEvent.SYSTEM_FUTURE_SETTLE)
        event_bus.publish_event(event)

