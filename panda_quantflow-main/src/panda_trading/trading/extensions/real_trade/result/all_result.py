import datetime
import time

from panda_backtest.backtest_common.model.result.panda_real_daily_value import PandaRealDailyValue as XbRealDailyValue
from panda_backtest.backtest_common.model.result.panda_real_minute_value import PandaRealMinuteValue as XbRealMinuteValue
from panda_backtest.backtest_common.system.context.core_context import CoreContext

from panda_backtest.backtest_common.model.result.panda_backtest_account import PandaBacktestAccount as XbBacktestAccount
from utils.annotation.singleton_annotation import singleton
from panda_trading.trading.extensions.real_trade.result.trade_reverse_result import TradeReverseResult
from panda_trading.trading.extensions.real_trade.result.future_reverse_result import FutureReverseResult
from panda_trading.trading.extensions.real_trade.result.standard_symbol_result import StandSymbolResult
import json
from common.connector.redis_client import RedisClient
from common.connector.mongodb_handler import DatabaseHandler as MongoClient
from panda_trading.trading.constant.redis_key import *
import common.config as config

@singleton
class AllTradeReverseResult(object):

    def __init__(self):
        self.stock_result_dict = dict()
        self.future_result_dict = dict()
        self.context = CoreContext.get_instance()
        self.standard_symbol_result = StandSymbolResult()
        self._redis_client = RedisClient()
        self._business_mongo = MongoClient(config).get_mongo_db()
        self.all_account = None

    def add_account(self, account):
        self.stock_result_dict[account] = TradeReverseResult(account)

    def add_future_account(self, future_account):
        self.future_result_dict[future_account] = FutureReverseResult(future_account)

    def init_data(self):
        for trade_reverse_result in self.stock_result_dict.values():
            trade_reverse_result.init_data()

        self.standard_symbol_result.init_data()

    def calculate_all_result(self):
        print('测试数据')

    def show_all_result(self, benchmark_name):
        print('结果结算')

    def new_date(self):
        for trade_reverse_result in self.stock_result_dict.values():
            trade_reverse_result.new_date()

        for trade_reverse_result in self.future_result_dict.values():
            trade_reverse_result.new_date()

        self.standard_symbol_result.new_date()

    def day_start(self):
        self.standard_symbol_result.day_start()

    def end_date(self):
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        for trade_reverse_result in self.stock_result_dict.values():
            trade_reverse_result.end_date()

        for future_reverse_result in self.future_result_dict.values():
            future_reverse_result.end_date()

        self.standard_symbol_result.end_date()

    def bar_end(self):
        pass

    def save_daily_result(self):
        self.save_minute_result(end_date=True)
        self.save_account_info(end_date=True)

        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        real_daily_result = XbRealDailyValue()
        real_daily_result.run_id = run_info.run_id
        real_daily_result.trade_date = strategy_context.trade_date
        real_daily_result.date = strategy_context.now
        real_daily_result.total_value = self.all_account.total_profit
        real_daily_result.strategy_profit = self.all_account.total_profit / self.all_account.start_capital
        real_daily_result.add_profit = self.all_account.add_profit
        real_daily_result.daily_pnl = self.all_account.daily_pnl
        collection = self._business_mongo.real_daily_value
        query = {'run_id': run_info.run_id, 'date': real_daily_result.date}
        # collection.update_one(query, real_daily_result.__dict__, upsert=True)
        collection.update_one(query, {"$set": real_daily_result.__dict__}, upsert=True)

    def save_minute_result(self, end_date=False):
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        real_minute_value = XbRealMinuteValue()
        real_minute_value.run_id = run_info.run_id
        real_minute_value.total_value = self.all_account.total_profit
        real_minute_value.add_profit = self.all_account.add_profit
        real_minute_value.daily_pnl = self.all_account.daily_pnl
        next_minute_time = datetime.datetime.now() + datetime.timedelta(minutes=1)
        real_minute_value.date = next_minute_time.strftime('%Y%m%d')
        real_minute_value.trade_date = strategy_context.trade_date
        if end_date:
            real_minute_value.minute = '1530'
        else:
            real_minute_value.minute = next_minute_time.strftime('%H%M')
        collection = self._business_mongo.real_minute_value
        query = {'run_id': run_info.run_id, 'date': real_minute_value.date, 'minute': real_minute_value.minute}
        collection.update_one(query, {"$set": real_minute_value.__dict__}, upsert=True)

    def save_account_info(self, end_date=False):
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        all_account_info = list()

        all_account = XbBacktestAccount()
        all_account.account_id = '0'
        all_account.gmt_create = strategy_context.trade_date
        all_account.mock_id = run_info.run_id
        all_account.type = 2

        stock_position_list = list()
        future_position_list = list()

        # 股票账号信息：
        for trade_reverse_result in self.stock_result_dict.values():
            stock_daily_result = trade_reverse_result
            xb_back_test_account = stock_daily_result.xb_back_test_account

            if xb_back_test_account.start_capital == 0 or xb_back_test_account.start_capital is None:
                return

            all_account.available_funds += xb_back_test_account.available_funds
            all_account.total_profit += xb_back_test_account.total_profit
            all_account.add_profit += xb_back_test_account.add_profit
            all_account.market_value += xb_back_test_account.market_value
            all_account.cost += xb_back_test_account.cost
            all_account.margin += xb_back_test_account.margin
            all_account.start_capital += xb_back_test_account.start_capital
            all_account.yes_total_capital += xb_back_test_account.yes_total_capital
            all_account.daily_pnl += xb_back_test_account.daily_pnl
            all_account.withdraw += xb_back_test_account.withdraw
            all_account.deposit += xb_back_test_account.deposit
            all_account.today_withdraw += xb_back_test_account.today_withdraw
            all_account.today_deposit += xb_back_test_account.today_deposit

            all_account_info.append(xb_back_test_account.__dict__)

            for symbol, future_long_position in stock_daily_result.position_dict.items():
                stock_position_list.append(dict.copy(future_long_position.__dict__))

        # 期货账号信息：
        for future_reverse_result in self.future_result_dict.values():

            future_daily_result = future_reverse_result
            xb_back_test_account = future_daily_result.xb_back_test_account

            if xb_back_test_account.start_capital == 0 or xb_back_test_account.start_capital is None:
                return

            all_account.available_funds += xb_back_test_account.available_funds
            all_account.total_profit += xb_back_test_account.total_profit
            all_account.add_profit += xb_back_test_account.add_profit
            all_account.market_value += xb_back_test_account.market_value
            all_account.cost += xb_back_test_account.cost
            all_account.margin += xb_back_test_account.margin
            all_account.start_capital += xb_back_test_account.start_capital
            all_account.yes_total_capital += xb_back_test_account.yes_total_capital
            all_account.daily_pnl += xb_back_test_account.daily_pnl
            all_account.holding_pnl += xb_back_test_account.holding_pnl
            all_account.realized_pnl += xb_back_test_account.realized_pnl
            all_account.frozen_capital += xb_back_test_account.frozen_capital
            all_account.withdraw += xb_back_test_account.withdraw
            all_account.deposit += xb_back_test_account.deposit
            all_account.today_withdraw += xb_back_test_account.today_withdraw
            all_account.today_deposit += xb_back_test_account.today_deposit

            all_future_position_long_dict = dict()
            for symbol, future_long_position in future_daily_result.long_position_dict.items():
                all_future_position_long_dict[symbol] = dict.copy(future_long_position.__dict__)

            all_future_position_short_dict = dict()
            for symbol, future_short_position in future_daily_result.short_position_dict.items():
                all_future_position_short_dict[symbol] = dict.copy(future_short_position.__dict__)

            all_account_info.append(xb_back_test_account.__dict__)

            future_position_list.extend(list(all_future_position_long_dict.values()))
            future_position_list.extend(list(all_future_position_short_dict.values()))

        # 统计持仓
        all_position_list = list()
        all_position_list.append(stock_position_list)
        all_position_list.append(future_position_list)

        self._redis_client.setHashRedis(real_trade_account_positions, run_info.run_id, json.dumps(all_position_list))
        self.all_account = all_account
        all_account_info.append(all_account.__dict__)
        if end_date is False:
            self._redis_client.setHashRedis(real_trade_account_assets, run_info.run_id, json.dumps(all_account_info))
            self.save_minute_result()
        else:
            collection = self._business_mongo.xb_real_account
            for account_dict in all_account_info:
                account_dict['trade_date'] = strategy_context.trade_date
                query = {'run_id': run_info.run_id, 'trade_date': strategy_context.trade_date,
                         'account_id': account_dict['account_id']}
                # collection.update_one(query, account_dict, upsert=True)
                collection.update_one(query, {"$set": account_dict}, upsert=True)
    def get_trade_reverse_result(self, account):
        return self.stock_result_dict[account]

    def get_future_reverse_result(self, account):
        return self.future_result_dict[account]

    def cash_moving(self, from_account, to_account, cash, move_type):
        pass

    def on_stock_rtn_order(self, order):
        trade_reverse_result = self.stock_result_dict[order.account]
        trade_reverse_result.on_stock_rtn_order(order)

    def on_stock_rtn_trade(self, trade):
        trade_reverse_result = self.stock_result_dict[trade.account_id]
        trade_reverse_result.on_stock_rtn_trade(trade)

    def on_stock_rtn_transfer(self, real_withdraw_deposit):
        trade_reverse_result = self.stock_result_dict[real_withdraw_deposit.account_id]
        trade_reverse_result.on_stock_rtn_transfer(real_withdraw_deposit)

    def refresh_stock_position(self, bar_data):
        for trade_reverse_result in self.stock_result_dict.values():
            trade_reverse_result.refresh_position(bar_data)

    def on_future_rtn_order(self, order):
        future_reverse_result = self.future_result_dict[order.account]
        future_reverse_result.on_future_rtn_order(order)

    def on_future_rtn_trade(self, trade):
        future_reverse_result = self.future_result_dict[trade.account_id]
        future_reverse_result.on_future_rtn_trade(trade)

    def on_future_rtn_transfer(self, real_withdraw_deposit):
        future_reverse_result = self.future_result_dict[real_withdraw_deposit.account_id]
        future_reverse_result.on_future_rtn_transfer(real_withdraw_deposit)

    def refresh_future_position(self, bar_data):
        for future_reverse_result in self.future_result_dict.values():
            future_reverse_result.refresh_position(bar_data)

    def refresh_future_asset(self, xb_back_test_account):
        future_reverse_result = self.future_result_dict[xb_back_test_account.account_id]
        future_reverse_result.update_asset(xb_back_test_account)

    def refresh_stock_asset(self, xb_back_test_account):
        trade_reverse_result = self.stock_result_dict[xb_back_test_account.account_id]
        trade_reverse_result.update_asset(xb_back_test_account)

    def on_future_all_pos_refresh(self, account, position_dict):
        future_reverse_result = self.future_result_dict[account]
        future_reverse_result.update_positions(position_dict)

    def on_future_trade_pos_refresh(self, account, position_dict):
        future_reverse_result = self.future_result_dict[account]
        future_reverse_result.update_trade_positions(position_dict)

    def on_stock_all_pos_refresh(self, account, position_dict):
        trade_reverse_result = self.stock_result_dict[account]
        trade_reverse_result.update_positions(position_dict)

    def on_stock_trade_pos_refresh(self, account, position_dict):
        trade_reverse_result = self.stock_result_dict[account]
        trade_reverse_result.update_trade_positions(position_dict)

    def refresh_standard_symbol(self, bar_data):
        self.standard_symbol_result.refresh_position(bar_data)
