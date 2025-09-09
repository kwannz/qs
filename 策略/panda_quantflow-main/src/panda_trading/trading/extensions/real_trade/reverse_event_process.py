import traceback
import time

from common.connector.mongodb_handler import DatabaseHandler as MongoClient

from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData
from panda_trading.trading.quotation.ctp.tushare.tushare_future_tick_quotation import TushareFutureTickQuotation
from panda_trading.trading.system.trade_time_manager import TradeTimeManager

from common.connector.redis_client import RedisClient
from panda_trading.trading_account_monitor.quotation.real_time_bar_map import RealTimeBarMap
import  common.config as config
class ReverseEventProcess(object):
    def __init__(self, context):
        print("init")
        self._context = context
        self.quotation_mongo_db = MongoClient(config).get_mongo_db()
        self.trade_time_manager = TradeTimeManager(self.quotation_mongo_db)
        self.redis_client = RedisClient()

    def init_backtest_params(self, handle_message):
        strategy_context = self._context.strategy_context
        strategy_context.init_run_info(handle_message)
        tushare_future_tick_quotation = TushareFutureTickQuotation(self.redis_client)
        real_time_bar_map = RealTimeBarMap(None, tushare_future_tick_quotation)
        quotation_data = QuotationData.get_instance()
        quotation_data.init_bar_dict(real_time_bar_map)
        strategy_context.init_trade_time_manager(self.trade_time_manager)
        self.init_data()

    def init_data(self):
        self._context.operation_proxy.init_event()
        self._context.operation_proxy.init_data()

    def event_factory(self):
        self.trade_time_manager.start_trade_time_play()

