import pandas as pd
from panda_backtest.backtest_common.data.fund.back_test.fund_account import FundAccount
import logging

from panda_backtest.backtest_common.data.stock.back_test.stock_account import StockAccount
from panda_backtest.backtest_common.data.result.back_test.portfolio import Portfolio
from panda_backtest.backtest_common.model.info.run_info import RunInfo
import time
from panda_backtest.backtest_common.data.future.back_test.future_account import FutureAccount

class StrategyContext(object):

    def __init__(self):
        self.enable_risk_control = False
        self.all_trade_reverse_result = None
        self.trade_time_manager = None
        self.run_info = RunInfo()
        self.portfolio = Portfolio(self)
        self.stock_account_dict = dict()
        self.future_account_dict = dict()
        self.fund_account_dict = dict()

        self.sub_position_stock_symbol_list = set()
        self.sub_position_future_symbol_list = set()
        self.sub_position_fund_symbol_list = set()

        self.sub_order_stock_symbol_list = set()
        self.sub_order_future_symbol_list = set()
        self.sub_order_fund_symbol_list = set()

    def init_opz_params(self, params_dict):
        for key, value in params_dict.items():
            self.__setattr__(key, value)
    def init_factor_params(self, df_factor: pd.DataFrame):
        self.__setattr__("df_factor", df_factor)

    def init_run_info(self, handle_message):
        self.run_info.strategy_id = handle_message['strategy_id']
        self.run_info.run_id = str(handle_message['back_test_id'])
        self.run_info.custom_tag = str(handle_message.setdefault('custom_tag', ''))
        self.run_info.run_type = handle_message['run_type']
        self.run_info.start_date = handle_message['start_date']
        self.run_info.end_date = handle_message['end_date']
        self.run_info.frequency = handle_message['frequency']
        self.run_info.stock_starting_cash = handle_message['start_capital']
        self.run_info.future_starting_cash = handle_message['start_future_capital']
        self.run_info.fund_starting_cash = handle_message['start_fund_capital']
        self.run_info.slippage = handle_message['slippage']
        self.run_info.future_slippage = handle_message.setdefault('future_slippage', 0)
        self.run_info.commission_multiplier = handle_message['commission_rate']
        self.run_info.margin_multiplier = handle_message['margin_rate']
        self.run_info.benchmark = handle_message['standard_symbol']
        self.run_info.matching_type = handle_message['matching_type']
        self.run_info.account_type = handle_message['account_type']
        self.run_info.date_type = handle_message.setdefault('date_type', 0)
        self.run_info.rate_dict_data_str = handle_message.setdefault('rate_dict_data_str', '')
        self.run_info.stock_account = handle_message.setdefault('stock_account', '8888')
        self.run_info.future_account = handle_message.setdefault('future_account', '5588')
        self.run_info.fund_account = handle_message.setdefault('fund_account', '2233')
        self.run_info.run_strategy_type = 0
        self.run_info.start_run_time = time.time()
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

    def add_stock_account(self, account):
        self.stock_account_dict[account] = StockAccount(self, account)  # 股票资金账户信息

    def add_future_account(self, future_account):
        self.future_account_dict[future_account] = FutureAccount(self, future_account)  # 股票资金账户信息

    def add_fund_account(self, fund_account):
        self.fund_account_dict[fund_account] = FundAccount(self, fund_account)  # 股票资金账户信息

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

    @property
    def trade_date_len(self):
        return len(self.trade_time_manager.all_date_list)

    @property
    def trade_date_list(self):
        return self.trade_time_manager.all_date_list

    @property
    def nature_date_len(self):
        return len(self.trade_time_manager.all_nature_date_list)

    def is_stock_trade(self):
        return self.trade_time_manager.is_stock_trade()

    def is_future_trade(self):
        return self.trade_time_manager.is_future_trade()

    def sub_stock_symbol(self, symbol_list, sub_type=0):
        if sub_type == 0:
            sub_stock_symbol_list = self.sub_position_stock_symbol_list
        else:
            sub_stock_symbol_list = self.sub_order_stock_symbol_list
        sub_stock_symbol_list.update(set(symbol_list))

    def sub_future_symbol(self, symbol_list, sub_type=0):

        if sub_type == 0:
            sub_future_symbol_list = self.sub_position_future_symbol_list
        else:
            sub_future_symbol_list = self.sub_order_future_symbol_list
        sub_future_symbol_list.update(set(symbol_list))

    def sub_fund_symbol(self, symbol_list, sub_type=0):
        if sub_type == 0:
            sub_fund_symbol_list = self.sub_position_fund_symbol_list
        else:
            sub_fund_symbol_list = self.sub_order_fund_symbol_list
        sub_fund_symbol_list.update(set(symbol_list))

    def un_sub_stock_symbol(self, symbol_list=None, sub_type=0):
        if sub_type == 0:
            sub_stock_symbol_list = self.sub_position_stock_symbol_list
        else:
            sub_stock_symbol_list = self.sub_order_stock_symbol_list
        if symbol_list is None:
            sub_stock_symbol_list.clear()
            return
        for symbol in symbol_list:
            if symbol in sub_stock_symbol_list:
                sub_stock_symbol_list.remove(symbol)

    def un_sub_future_symbol(self, symbol_list=None, sub_type=0):
        if sub_type == 0:
            sub_future_symbol_list = self.sub_position_future_symbol_list
        else:
            sub_future_symbol_list = self.sub_order_future_symbol_list
        if symbol_list is None:
            sub_future_symbol_list.clear()
            return
        for symbol in symbol_list:
            if symbol in sub_future_symbol_list:
                sub_future_symbol_list.remove(symbol)

    def un_sub_fund_symbol(self, symbol_list=None, sub_type=0):
        if sub_type == 0:
            sub_fund_symbol_list = self.sub_position_fund_symbol_list
        else:
            sub_fund_symbol_list = self.sub_order_fund_symbol_list
        if symbol_list is None:
            sub_fund_symbol_list.clear()
            return
        for symbol in symbol_list:
            if symbol not in sub_fund_symbol_list:
                sub_fund_symbol_list.remove(symbol)

    def is_last_trade_date(self):
        if self.trade_time_manager.trade_date == self.trade_time_manager.all_date_list[-1]:
            return True
        else:
            return False

    def get_next_count_date(self, date, count):
        return self.trade_time_manager.get_next_count_date(date, count)

    def get_next_count_nature_date(self, date, count):
        return self.trade_time_manager.get_next_count_nature_date(date, count)

    def get_date_distance(self, start_date, end_date):
        return self.trade_time_manager.get_date_distance(start_date, end_date)

    def is_trade_date(self):
        return self.trade_time_manager.is_trade_date()

    def is_trade_date_end(self):
        if self.hms == '150000':
            return True
        else:
            return False

