from panda_backtest.backtest_common.system.event.event import EventBus
import logging

class CoreContext(object):
    _context = None

    def __init__(self, strategy_context):
        CoreContext._context = self
        self.event_process = None
        self.operation_proxy = None
        self.event_bus = EventBus()
        self.strategy_context = strategy_context
        self.risk_control_manager = None

    @classmethod
    def get_instance(cls):
        if CoreContext._context is None:
            raise RuntimeError(u"_context未初始化")
        return CoreContext._context

    def set_event_process(self, event_process):
        self.event_process = event_process

    def set_operation_proxy(self, operation_proxy):
        self.operation_proxy = operation_proxy

    def set_risk_control_manager(self, risk_control_manager):
        self.risk_control_manager = risk_control_manager

