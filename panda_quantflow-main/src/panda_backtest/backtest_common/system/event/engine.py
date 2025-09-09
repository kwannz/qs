import time
import logging

class Engine(object):
    def __init__(self, context):
        self._context = context

    def run(self, handle_message):
        self.create_handle_data_event(handle_message)

    def create_handle_data_event(self, handle_message):
        old_ticks = time.time()
        self._context.event_process.init_backtest_params(handle_message)
        self._context.event_process.event_factory()
        new_ticks = time.time() - old_ticks
        print('策略运行时间' + str(new_ticks))
