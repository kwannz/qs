#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-27 下午5:02
# @Author : wlb
# @File   : trade_reverse_result.py
# @desc   :

from panda_backtest.backtest_common.model.result.panda_backtest_position import PandaBacktestPosition as XbBacktestPosition, SIDE_SELL, PartTradedNotQueueing, \
    CANCELLED, ACTIVE
from panda_backtest.backtest_common.model.result.panda_real_withdraw_deposit import PandaRealWithdrawDeposit as XbRealWithdrawDeposit
from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.backtest_common.system.event.event import Event, ConstantEvent
from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData
from panda_backtest.backtest_common.model.result.panda_backtest_account import PandaBacktestAccount as XbBacktestAccount
from common.connector.mongodb_handler import DatabaseHandler as MongoClient
from utils.log.log_factory import LogFactory


class TradeReverseResult(object):

    def __init__(self, account):
        # 相关数据
        self.logger = LogFactory.get_logger()
        self.context = CoreContext.get_instance()
        self.business_mongo = MongoClient.get_mongo_db()
        # self.mysql_client = MysqlClient.get_mysql_client()

        self.xb_back_test_account = XbBacktestAccount()
        self.xb_back_test_account.account_id = account
        self.xb_back_test_account.type = 0
        self.xb_back_test_account.mock_id = self.context.strategy_context.run_info.run_id
        self.xb_back_test_account.gmt_create = self.context.strategy_context.trade_date

        self.position_dict = dict()  # 持仓
        # self.strategy_position_dict = dict()  # 策略本地持仓
        self.init_capital()
        self.init_withdraw_deposit()

    def init_capital(self):

        # 初始化初始资金和昨日总权益
        self.xb_back_test_account.start_capital = self.context.strategy_context.run_info.stock_starting_cash
        self.logger.info('初始资金==》' + str(self.xb_back_test_account.start_capital))
        xb_real_account_collection = self.business_mongo.xb_real_account
        xb_real_account_cur = xb_real_account_collection.find(
            {'mock_id': self.context.strategy_context.run_info.run_id,
             'account_id': self.xb_back_test_account.account_id,
             'type': 0}) \
            .sort([('trade_date', -1)]).limit(1)
        xb_real_account_list = list(xb_real_account_cur)
        if len(xb_real_account_list) > 0:
            xb_real_account = xb_real_account_list[0]
            self.xb_back_test_account.yes_total_capital = xb_real_account['total_profit']
        else:
            self.xb_back_test_account.yes_total_capital = self.xb_back_test_account.start_capital

    def init_withdraw_deposit(self):
        self.xb_back_test_account.withdraw = 0
        self.xb_back_test_account.deposit = 0
        real_withdraw_deposit_col = self.business_mongo.real_withdraw_deposit
        real_withdraw_deposit_cur = real_withdraw_deposit_col.find(
            {'run_id': self.context.strategy_context.run_info.run_id,
             'account_id': self.xb_back_test_account.account_id,
             'account_type': 0})
        real_withdraw_deposit_list = list(real_withdraw_deposit_cur)
        for real_withdraw_deposit in real_withdraw_deposit_list:
            if real_withdraw_deposit['type'] == 0:
                self.xb_back_test_account.deposit += real_withdraw_deposit['money']
            else:
                self.xb_back_test_account.withdraw += real_withdraw_deposit['money']

    def on_stock_rtn_transfer(self, real_withdraw_deposit):
        return
        if real_withdraw_deposit.type == 0:
            self.xb_back_test_account.deposit += real_withdraw_deposit.money
        else:
            self.xb_back_test_account.withdraw += real_withdraw_deposit.money

    def save_today_transfer(self, real_withdraw_deposit):
        real_withdraw_deposit_col = self.business_mongo.real_withdraw_deposit
        key = {'run_id': self.context.strategy_context.run_info.run_id, 'date': real_withdraw_deposit.date,
               'account_id': self.xb_back_test_account.account_id, 'type': real_withdraw_deposit.type,
               'account_type': real_withdraw_deposit.account_type}
        real_withdraw_deposit_col.update_one(key, real_withdraw_deposit.__dict__, upsert=True)
        self.init_withdraw_deposit()

    def save_stock_start_capital(self):
        self.logger.info("更新股票账号初始权益")
        strategy_context = self.context.strategy_context
        run_id = strategy_context.run_info.run_id
        update_dict = {'fund_stock': self.xb_back_test_account.start_capital}
        result_col = self.business_mongo.real_xb_back_test
        result_col.update_one({'product_strategy_id': run_id},
                              {'$set': update_dict})

    def init_data(self):
        pass

    def update_asset(self, xb_back_test_account):
        self.xb_back_test_account.__dict__.update(xb_back_test_account.__dict__)
        self.xb_back_test_account.total_profit = self.xb_back_test_account.market_value + \
                                                 self.xb_back_test_account.available_funds + \
                                                 self.xb_back_test_account.frozen_capital
        print('update_asset动态权益==》' + str(self.xb_back_test_account.total_profit))
        if self.xb_back_test_account.today_deposit > 0:
            # 保存当天入金
            real_withdraw_deposit = XbRealWithdrawDeposit()
            real_withdraw_deposit.run_id = self.context.strategy_context.run_info.run_id
            real_withdraw_deposit.date = self.context.strategy_context.trade_date
            real_withdraw_deposit.type = 0
            real_withdraw_deposit.money = self.xb_back_test_account.today_deposit
            real_withdraw_deposit.account_id = self.xb_back_test_account.account_id
            real_withdraw_deposit.account_type = 0
            self.save_today_transfer(real_withdraw_deposit)

        if self.xb_back_test_account.today_withdraw > 0:
            # 保存当天出金
            real_withdraw_deposit = XbRealWithdrawDeposit()
            real_withdraw_deposit.run_id = self.context.strategy_context.run_info.run_id
            real_withdraw_deposit.date = self.context.strategy_context.trade_date
            real_withdraw_deposit.type = 1
            real_withdraw_deposit.account_type = 0
            real_withdraw_deposit.money = self.xb_back_test_account.today_withdraw
            real_withdraw_deposit.account_id = self.xb_back_test_account.account_id
            self.save_today_transfer(real_withdraw_deposit)

    def update_positions(self, position_dict):
        """
        账号更新所有持仓
        :param position_dict:
        :return:
        """
        bar_dict = QuotationData.get_instance().bar_dict
        market_value = 0
        sub_list = list()
        for symbol, xb_back_test_position in position_dict.items():
            if xb_back_test_position.position == 0:
                continue
            sub_list.append(symbol)
            xb_back_test_position.back_id = self.xb_back_test_account.mock_id
            xb_back_test_position.account_id = self.xb_back_test_account.account_id
            xb_back_test_position.last_price = bar_dict[symbol].last
            if xb_back_test_position.last_price == 0:
                if bar_dict[symbol].preclose != 0:
                    xb_back_test_position.last_price = bar_dict[symbol].preclose
                else:
                    xb_back_test_position.last_price = xb_back_test_position.price
            xb_back_test_position.market_value = xb_back_test_position.last_price * \
                                                 xb_back_test_position.position
            total_price = xb_back_test_position.position * xb_back_test_position.price
            xb_back_test_position.accumulate_profit = xb_back_test_position.market_value - total_price
            xb_back_test_position.frozen_position = xb_back_test_position.position - xb_back_test_position.sellable
            self.position_dict[symbol] = xb_back_test_position
            market_value += xb_back_test_position.market_value

        self.xb_back_test_account.market_value = market_value
        self.xb_back_test_account.total_profit = self.xb_back_test_account.market_value + \
                                                 self.xb_back_test_account.available_funds + \
                                                 self.xb_back_test_account.frozen_capital
        print('update_positions动态权益==》' + str(self.xb_back_test_account.total_profit))
        if self.xb_back_test_account.start_capital is None or self.xb_back_test_account.start_capital == 0:
            self.xb_back_test_account.start_capital = self.xb_back_test_account.total_profit - \
                                                      self.xb_back_test_account.today_deposit + \
                                                      self.xb_back_test_account.today_withdraw
            self.xb_back_test_account.yes_total_capital = self.xb_back_test_account.start_capital
            self.save_stock_start_capital()

        self.update_total_capital()
        event_bus = self.context.event_bus
        event = Event(ConstantEvent.SYSTEM_STOCK_QUOTATION_START_SUB, sub_symbol_list=sub_list)
        event_bus.publish_event(event)
        strategy_context = self.context.strategy_context
        strategy_context.init_stock_account_position_status(self.xb_back_test_account.account_id)

    def update_trade_positions(self, position_data_dict):
        """
        xtp回报更新补充持仓的保证金手续费等金额异步数据（仓位提前在报单和成交回报中处理）
        :param position_data_dict:
        :return:
        """
        sub_list = list()
        for xb_back_test_position in position_data_dict.values():
            symbol = xb_back_test_position.contract_code
            if symbol in self.position_dict.keys():
                position_item = self.position_dict[symbol]
            else:
                continue
            sub_list.append(symbol)
            position_item.price = xb_back_test_position.price
            position_item.open_cost = xb_back_test_position.open_cost
            position_item.position_cost = xb_back_test_position.position_cost
            position_item.hold_price = xb_back_test_position.hold_price
            position_item.cost = xb_back_test_position.cost
            position_item.round_lot = 100
        event_bus = self.context.event_bus
        event = Event(ConstantEvent.SYSTEM_STOCK_QUOTATION_START_SUB, sub_symbol_list=sub_list)
        event_bus.publish_event(event)

    def refresh_account(self):
        bar_dict = QuotationData.get_instance().bar_dict
        for symbol, xb_back_test_position in self.position_dict.items():
            bar_data = bar_dict[xb_back_test_position.contract_code]
            self.refresh_position(bar_data)

    def refresh_position(self, bar_data):
        if bar_data.last == 0:
            return
        if bar_data.symbol in self.position_dict.keys():
            xb_back_test_position = self.position_dict[bar_data.symbol]
            xb_back_test_position.last_price = bar_data.last
            old_market_value = xb_back_test_position.market_value
            xb_back_test_position.market_value = xb_back_test_position.last_price * xb_back_test_position.position
            total_price = xb_back_test_position.position * xb_back_test_position.price
            xb_back_test_position.accumulate_profit = xb_back_test_position.market_value - total_price
            self.xb_back_test_account.market_value += xb_back_test_position.market_value - old_market_value
            self.xb_back_test_account.total_profit = \
                self.xb_back_test_account.market_value + self.xb_back_test_account.available_funds + \
                self.xb_back_test_account.frozen_capital
            self.xb_back_test_account.add_profit = \
                self.xb_back_test_account.total_profit - self.xb_back_test_account.start_capital + \
                self.xb_back_test_account.withdraw - \
                self.xb_back_test_account.deposit
            self.xb_back_test_account.daily_pnl = \
                self.xb_back_test_account.total_profit - self.xb_back_test_account.yes_total_capital + \
                self.xb_back_test_account.today_withdraw - \
                self.xb_back_test_account.today_deposit

    def update_total_capital(self):
        """
        更新账号总资产
        :return:
        """
        if self.xb_back_test_account.start_capital:
            self.update_profit()

    def update_profit(self):

        self.xb_back_test_account.add_profit = self.xb_back_test_account.total_profit - \
                                               self.xb_back_test_account.start_capital + \
                                               self.xb_back_test_account.withdraw - \
                                               self.xb_back_test_account.deposit
        self.xb_back_test_account.daily_pnl = self.xb_back_test_account.total_profit - \
                                              self.xb_back_test_account.yes_total_capital + \
                                              self.xb_back_test_account.today_withdraw - \
                                              self.xb_back_test_account.today_deposit

    def on_stock_rtn_order(self, order):
        if hasattr(order, 'is_close_local') and order.is_close_local == 1:
            # 本地持仓
            pass
            return
        strategy_context = self.context.strategy_context
        self.handle_order_change_position(order, False)
        # if order.now_system_order == 1 and order.client_id == strategy_context.run_info.run_id:
        #     self.handle_order_change_position(order, True)

    def handle_order_change_position(self, order, is_strategy_position):
        position_dict = self.position_dict

        # 在查询持仓回调前提前处理冻结持仓问题
        if order.side == SIDE_SELL:
            if order.order_book_id in position_dict.keys():
                position_item = position_dict[order.order_book_id]
            else:
                return

            if order.status == PartTradedNotQueueing or order.status == CANCELLED:
                print('撤销冻结')
                position_item.frozen_position -= order.unfilled_quantity
                position_item.sellable = position_item.position - position_item.frozen_position

            elif order.status == ACTIVE:
                print('开始冻结')
                position_item.frozen_position += order.unfilled_quantity
                position_item.sellable = position_item.position - position_item.frozen_position

    def on_stock_rtn_trade(self, trade_data):
        strategy_context = self.context.strategy_context
        trade_data.trade_date = strategy_context.trade_date
        trade_data.run_id = strategy_context.run_info.run_id
        # if trade_data.is_close_local == 1:
        #     # 本地持仓
        #     self.handle_close_local_trade(trade_data)
        #     return

        # 处理成交持仓变化
        self.handle_trade_change_position(trade_data.account_id, trade_data, False)

        # # 判断是否为当前进程订单,回报成交回报
        # if trade_data.now_system_order == 1 and trade_data.client_id == strategy_context.run_info.run_id:
        #     self.handle_trade_change_position(trade_data.account_id, trade_data, True)

        trade_collection = self.business_mongo.redefine_real_trade
        key = {'run_id': trade_data.run_id, 'trade_id': trade_data.trade_id, 'account': self.xb_back_test_account.account_id}
        trade_collection.update_one(key, {"$set":trade_data.__dict__}, upsert=True)

    # def handle_close_local_trade(self, trade_data):
    #     symbol_position = self.strategy_position_dict[trade_data.contract_code]
    #     symbol_position.position -= trade_data.volume
    #     symbol_position.sellable -= trade_data.volume
    #     symbol_position.td_position -= trade_data.close_td_pos

    def handle_trade_change_position(self, account, trade, is_strategy_position):
        strategy_context = self.context.strategy_context
        # # 在查询持仓回调前提前处理冻结持仓问题
        # if is_strategy_position:
        #     position_dict = self.strategy_position_dict
        # else:
        #     position_dict = self.position_dict

        position_dict = self.position_dict

        if trade.business == SIDE_SELL:
            if trade.contract_code in position_dict.keys():
                position_item = position_dict[trade.contract_code]
            else:
                return
            position_item.position -= trade.volume
            position_item.frozen_position -= trade.volume
            position_item.yd_position -= trade.volume
            position_item.sellable = position_item.position - position_item.frozen_position
            # position_item.yd_position -= trade.volume

            if position_item.position == 0:
                position_dict.pop(trade.contract_code)

        else:
            if trade.contract_code in position_dict.keys():
                position_item = position_dict[trade.contract_code]
            else:
                position_item = XbBacktestPosition()
                position_item.contract_code = trade.contract_code
                position_item.yd_position = 0
                position_item.frozen_position = 0
                position_item.frozen_td_position = 0
                position_item.sellable = 0

            print('position', position_item.position)
            print('td_position', position_item.td_position)
            print('frozen_position', position_item.frozen_position)
            print('volume', trade.volume)
            position_item.account_id = account
            position_item.type = 0
            position_item.direction = trade.business
            position_item.position += trade.volume
            position_item.td_position += trade.volume
            position_item.frozen_position += trade.volume
            position_item.frozen_td_position += trade.volume
            position_item.gmt_create = strategy_context.trade_date

            position_dict[trade.contract_code] = position_item

    def day_start(self):
        pass

    def new_date(self):
        self.xb_back_test_account.gmt_create = self.context.strategy_context.trade_date
        self.xb_back_test_account.yes_total_capital = self.xb_back_test_account.total_profit
        self.xb_back_test_account.today_deposit = 0
        self.xb_back_test_account.today_withdraw = 0
        self.xb_back_test_account.daily_pnl = 0

    def end_date(self):
        self.handle_strategy_position_change_day()

    def handle_strategy_position_change_day(self, day_end=True):
        pass
        # strategy_context = self.context.strategy_context
        # # 维护本地持仓，将今日仓位换为昨日仓位
        # for strategy_position in self.strategy_position_dict.values():
        #     if day_end or strategy_position.gmt_create is '' or strategy_position.gmt_create is \
        #             None or strategy_position.gmt_create != strategy_context.trade_date:
        #         strategy_position.yd_position += strategy_position.td_position
        #         strategy_position.td_position = 0
        #         strategy_position.gmt_create = strategy_context.trade_date

    def restore_save(self, mock_id):
        pass

    def restore_read(self, mock_id):
        pass