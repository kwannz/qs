import pickle

from panda_backtest.backtest_common.system.context.core_context import CoreContext
from common.connector.redis_client import RedisClient

from common.connector.mongodb_handler import DatabaseHandler as MongoClient
from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData
from panda_backtest.backtest_common.model.quotation.dividend import Dividend
from panda_trading.trading.constant.redis_key import real_trade_restore_data, standard_symbol_result
from common.config.config import config

class StandSymbolResult(object):

    def __init__(self):
        self.context = CoreContext.get_instance()
        self.start_capital = 1000000   # 回测时的起始本金（默认100万）

        self.standard_symbol = '603081'  # 基准合约
        self.standard_symbol_start_value = None  # 基准合约开始价值
        self.standard_symbol_position = 0  # 基准合约仓位
        self.standard_symbol_cash = 0  # 基准合约余额
        self.standard_symbol_value = 0  # 基准合约余额

        self.quotation_mongo_db = MongoClient(config).get_mongo_db()  # mongodb客户端连接

        # 当日收益率
        self.standard_portfolio = None

    def init_data(self):
        # self.start_capital = self._context.strategy_context.run_info.stock_starting_cash
        self.standard_symbol = self.context.strategy_context.run_info.benchmark
        self.standard_symbol_value = self.start_capital
        self.standard_symbol_cash = self.start_capital

    def new_date(self):
        pass

    def day_start(self):
        self.set_dividend()

    def refresh_position(self, bar_data):
        # 基准合约初始化价值
        if self.standard_symbol_start_value is None and bar_data.last != 0:
            self.standard_symbol_start_value = bar_data.last
            self.standard_symbol_position = self.start_capital / bar_data.last
            self.standard_symbol_cash = 0

    def end_date(self):
        bar_dict = QuotationData.get_instance().bar_dict
        if bar_dict[self.standard_symbol].last != 0:
            self.standard_symbol_value = self.standard_symbol_cash + \
                self.standard_symbol_position * bar_dict[self.standard_symbol].last

        self.standard_portfolio = self.standard_symbol_value / self.start_capital - 1

    def set_dividend(self):
        bar_dict = QuotationData.get_instance().bar_dict
        strategy_context = self.context.strategy_context

        # 基准除权除息
        collection = self.quotation_mongo_db.stock_dividends
        stand_dividend_dict = collection.find_one(
            {'ex_div_date': str(strategy_context.trade_date), 'symbol': self.standard_symbol})

        if stand_dividend_dict:
            stand_dividend = Dividend()
            stand_dividend.__dict__ = stand_dividend_dict
            self.standard_symbol_cash += stand_dividend.cash_div_tax * \
                self.standard_symbol_position
            self.standard_symbol_position = int(
                self.standard_symbol_position +
                self.standard_symbol_position *
                (
                    stand_dividend.share_trans_ratio +
                    stand_dividend.share_ratio))
            self.standard_symbol_value = bar_dict[
                self.standard_symbol].last * self.standard_symbol_position

    def restore_save(self, mock_id):
        save_run_data_dict = dict()
        save_list = ['standard_portfolio']
        for name, item in self.__dict__.items():
            if name in save_list:
                save_run_data_dict[name] = item

        redis_client = RedisClient()
        redis_client.setHashRedis(real_trade_restore_data + str(mock_id), standard_symbol_result,
                                  pickle.dumps(save_run_data_dict))

    def restore_read(self, mock_id):
        redis_client = RedisClient()
        var_run_data = redis_client.getHashRedis(real_trade_restore_data + str(mock_id), standard_symbol_result)
        if var_run_data:
            var_run_data = pickle.loads(var_run_data)
            for name, value in var_run_data.items():
                self.__setattr__(name, value)
