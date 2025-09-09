import json
import os
import re

from panda_backtest.backtest_common.data.result.real_time.portfolio import Portfolio
from panda_backtest.backtest_common.model.info.run_info import RunInfo
import time
from panda_backtest.backtest_common.data.future.real_time.future_account import FutureAccount
from datetime import datetime
import pickle
from common.connector.redis_client import RedisClient
from panda_trading.trading.constant.redis_key import real_trade_restore_data, restore_strategy_context


class StrategyContext(object):

    def __init__(self):
        self.enable_risk_control = False
        self.all_trade_reverse_result = None
        self.trade_time_manager = None
        self.run_info = RunInfo()
        self.stock_account_dict = dict()
        self.portfolio = Portfolio(self)
        self.future_account_dict = dict()
        # 期货交易时间
        self.pz_dict = dict()
        self.sub_stock_symbol_list = list()
        self.sub_future_symbol_list = list()
        self.sub_strategy_stock_symbol_list = list()
        self.sub_strategy_future_symbol_list = list()

    def init_opz_params(self, params_dict):
        for key, value in params_dict.items():
            self.__setattr__(key, value)

    def init_all_result(self, all_trade_reverse_result):
        self.all_trade_reverse_result = all_trade_reverse_result

    def init_trade_time_manager(self, trade_time_manager):
        self.trade_time_manager = trade_time_manager

    @property
    def now(self):
        return self.trade_time_manager.now

    @property
    def trade_date(self):
        return self.trade_time_manager.trade_date

    @property
    def trade_time(self):
        return self.trade_time_manager.trade_time

    @property
    def hms(self):
        return self.trade_time_manager.hms

    def init_run_info(self, handle_message):
        self.run_info.strategy_id = handle_message['strategy_id']
        # self.run_info.strategy_name = handle_message['strategy_name']
        self.run_info.run_id = str(handle_message['run_id'])
        self.run_info.run_type = handle_message['run_type']
        self.run_info.stock_starting_cash = handle_message['start_capital']
        self.run_info.future_starting_cash = handle_message['start_future_capital']
        self.run_info.benchmark = handle_message['standard_symbol']
        self.run_info.account_type = handle_message['account_type']
        self.run_info.date_type = handle_message.setdefault('date_type', 0)
        self.run_info.stock_account = handle_message.setdefault('stock_account_id', '15032863')
        self.run_info.future_account = handle_message.setdefault('future_account_id', '5588')
        self.run_info.product_id = handle_message.setdefault('product_id', None)
        self.run_info.product_name = handle_message.setdefault('product_name', None)
        standard_info_list = self.run_info.benchmark.split('.')
        if len(standard_info_list) != 2:
            self.run_info.standard_type = 0
        else:
            standard_exchange = self.run_info.benchmark.split('.')[1]
            if standard_exchange == 'SZ' or standard_exchange == 'SH':
                self.run_info.standard_type = 0
            elif standard_exchange == 'OF':
                self.run_info.standard_type = 2
            else:
                self.run_info.standard_type = 1

    def init_future_trade_time_dict(self):
        redis_client = RedisClient()
        res = redis_client.getRedis('future_trade_time_dict')
        if res:
            self.pz_dict = pickle.loads(res)

    def add_future_account(self, future_account):
        self.future_account_dict[future_account] = FutureAccount(self, future_account)  # 股票资金账户信息

    def judge_future_trade(self, symbol):
        code_type = re.sub(r'\d+', '', symbol.split(".")[0])  # 转换成品种名，如AP
        if code_type in self.pz_dict.keys():
            time_stamp = datetime.now()
            now_time = time_stamp.strftime('%H:%M')
            return self.pz_dict[code_type].overlaps_point(time.strptime(now_time, "%H:%M"))

    def is_stock_trade(self):
        return self.trade_time_manager.is_stock_trade()

    def is_future_trade(self):
        return self.trade_time_manager.is_future_trade()

    def sub_stock_symbol(self, symbol_list):
        for symbol in symbol_list:
            if symbol not in self.sub_stock_symbol_list:
                self.sub_stock_symbol_list.append(symbol)

    def sub_strategy_stock_symbol(self, symbol_list):
        for symbol in symbol_list:
            if symbol not in self.sub_stock_symbol_list:
                self.sub_stock_symbol_list.append(symbol)
            if symbol not in self.sub_strategy_stock_symbol_list:
                self.sub_strategy_stock_symbol_list.append(symbol)

    def sub_future_symbol(self, symbol_list):
        for symbol in symbol_list:
            if symbol not in self.sub_future_symbol_list:
                self.sub_future_symbol_list.append(symbol)

    def sub_strategy_future_symbol(self, symbol_list):
        for symbol in symbol_list:
            if symbol not in self.sub_future_symbol_list:
                self.sub_future_symbol_list.append(symbol)
            if symbol not in self.sub_strategy_future_symbol_list:
                self.sub_strategy_future_symbol_list.append(symbol)

    def un_sub_stock_symbol(self, symbol_list):
        for symbol in symbol_list:
            if symbol not in self.sub_stock_symbol_list:
                self.sub_stock_symbol_list.remove(symbol)

    def un_sub_future_symbol(self, symbol_list):
        for symbol in symbol_list:
            if symbol not in self.sub_future_symbol_list:
                self.sub_future_symbol_list.remove(symbol)

    def init_future_account_position_status(self, account):
        self.future_account_dict[account].init_pos_status = True

    def init_stock_account_position_status(self, account):
        self.stock_account_dict[account].init_pos_status = True

    def restore_save(self, mock_id):
        redis_client = RedisClient()
        save_run_data_dict = dict()
        not_save_list = ['all_trade_reverse_result', 'run_info', 'stock_account_dict', 'portfolio',
                         'future_account_dict', 'pz_dict', 'trade_time_manager',
                         'sub_future_symbol_list', 'sub_stock_symbol_list', 'sub_strategy_stock_symbol_list',
                         'sub_strategy_future_symbol_list', 'enable_risk_control']
        for name, item in self.__dict__.items():
            if not name.startswith('_') and name not in not_save_list:
                save_run_data_dict[name] = item
        redis_client.setHashRedis(real_trade_restore_data + str(mock_id), restore_strategy_context,
                                       pickle.dumps(save_run_data_dict))

    def restore_read(self, mock_id):
        redis_client = RedisClient()
        var_run_data = redis_client.getHashRedis(real_trade_restore_data + str(mock_id), restore_strategy_context)
        if var_run_data:
            var_run_data = pickle.loads(var_run_data)
            for name, value in var_run_data.items():
                self.__setattr__(name, value)


if __name__ == '__main__':
    sc = StrategyContext()
    sc.init_future_trade_time_dict()
    res = sc.judge_future_trade('B2009.DCE')
    print(res)
