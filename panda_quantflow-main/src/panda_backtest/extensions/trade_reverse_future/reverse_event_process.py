from panda_backtest.data.quotation.bar_data_source import BarDataSource

from panda_backtest.backtest_common.data.quotation.back_test.bar_map import BarMap

from panda_backtest.backtest_common.exchange.common.back_test.trade_time_manager import TradeTimeManager
from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData
from panda_backtest.backtest_common.system.interface.base_event_process import BaseEventProcess
from common.connector.mongodb_handler import DatabaseHandler
from common.config.config import config

class ReverseEventProcess(BaseEventProcess):
    def __init__(self, context):
        self._context = context
        self.quotation_mongo_db = DatabaseHandler(config=config)
        self.trade_time_manager = TradeTimeManager(self.quotation_mongo_db)

    def init_backtest_params(self, handle_message):
        strategy_context = self._context.strategy_context
        strategy_context.init_run_info(handle_message)
        strategy_context.init_trade_time_manager(self.trade_time_manager)
        bar_map = BarMap(BarDataSource())
        QuotationData.get_instance().init_bar_dict(bar_map)
        self.init_data()

    def init_data(self):
        self._context.operation_proxy.init_data()
        self._context.operation_proxy.init_event()

    def event_factory(self):
        self.trade_time_manager.start_trade_time_play()
