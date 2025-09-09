import copy
import os
import pickle
import time
from collections import OrderedDict

from panda_backtest.backtest_common.constant.strategy_constant import OPEN, SIDE_BUY, PartTradedNotQueueing, CANCELLED, ACTIVE, \
    SIDE_SELL, CLOSE

from panda_backtest.backtest_common.model.result.panda_backtest_position import PandaBacktestPosition as XbBacktestPosition
from panda_backtest.backtest_common.model.result.panda_real_withdraw_deposit import PandaRealWithdrawDeposit as XbRealWithdrawDeposit
from panda_backtest.backtest_common.model.result.panda_backtest_account import PandaBacktestAccount as XbBacktestAccount

from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.backtest_common.system.event.event import Event, ConstantEvent
# from redefine_trade.connector.mysql_client import MysqlClient

from common.connector.redis_client import RedisClient

from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData
from common.connector.mongodb_handler import DatabaseHandler as MongoClient


class FutureReverseResult(object):

    def __init__(self, account):
        self.context = CoreContext.get_instance()
        self.business_mongo = MongoClient.get_mongo_db()
        self.xb_back_test_account = XbBacktestAccount()
        self.xb_back_test_account.account_id = account
        self.xb_back_test_account.type = 1
        self.xb_back_test_account.mock_id = self.context.strategy_context.run_info.run_id
        self.xb_back_test_account.gmt_create = self.context.strategy_context.trade_date

        self.xb_back_test_trade_dict = dict()  # 成交记录

        self.long_position_dict = dict()  # 持仓
        self.short_position_dict = dict()  # 持仓
        # self.strategy_long_position_dict = dict()  # 策略本地持仓
        # self.strategy_short_position_dict = dict()  # 策略本地持仓
        self.initEnd = False  # 初始化状态
        # self.mysql_client = MysqlClient.get_mysql_client()
        self.init_capital()
        self.init_withdraw_deposit()

    def init_capital(self):
        self.xb_back_test_account.start_capital = self.context.strategy_context.run_info.future_starting_cash
        xb_real_account_collection = self.business_mongo.xb_real_account
        xb_real_account_cur = xb_real_account_collection.find(
            {'mock_id': self.context.strategy_context.run_info.run_id,
             'account_id': self.xb_back_test_account.account_id,
             'type': 1}) \
            .sort([('trade_date', -1)]).limit(1)
        xb_real_account_list = list(xb_real_account_cur)
        if len(xb_real_account_list) > 0:
            xb_real_account = xb_real_account_list[0]
            self.xb_back_test_account.yes_total_capital = xb_real_account['total_profit']
        else:
            self.xb_back_test_account.yes_total_capital = self.xb_back_test_account.start_capital

    def init_withdraw_deposit(self):
        self.xb_back_test_account.deposit = 0
        self.xb_back_test_account.withdraw = 0
        real_withdraw_deposit_col = self.business_mongo.real_withdraw_deposit
        real_withdraw_deposit_cur = real_withdraw_deposit_col.find(
            {'run_id': self.context.strategy_context.run_info.run_id,
             'account_id': self.xb_back_test_account.account_id,
             'account_type': 1})
        real_withdraw_deposit_list = list(real_withdraw_deposit_cur)
        for real_withdraw_deposit in real_withdraw_deposit_list:
            if real_withdraw_deposit['type'] == 0:
                self.xb_back_test_account.deposit += real_withdraw_deposit['money']
            else:
                self.xb_back_test_account.withdraw += real_withdraw_deposit['money']

    # def on_future_rtn_transfer(self, real_withdraw_deposit):
    #     real_withdraw_deposit.run_id = self.context.strategy_context.run_info.run_id
    #     real_withdraw_deposit.date = self.context.strategy_context.trade_date
    #     real_withdraw_deposit_col = self.business_mongo.real_withdraw_deposit
    #     real_withdraw_deposit_col.insert(real_withdraw_deposit.__dict__)
    #     if real_withdraw_deposit.type == 0:
    #         self.xb_back_test_account.deposit += real_withdraw_deposit.money
    #         # self.xb_back_test_account.today_deposit += real_withdraw_deposit.money
    #     else:
    #         self.xb_back_test_account.withdraw += real_withdraw_deposit.money
    #         # self.xb_back_test_account.today_withdraw += real_withdraw_deposit.money

    def save_future_start_capital(self):
        strategy_context = self.context.strategy_context
        run_id = strategy_context.run_info.run_id
        update_dict = {'fund_futures': self.xb_back_test_account.start_capital}
        result_col = self.business_mongo.real_xb_back_test
        result_col.update_one({'product_strategy_id': run_id},
                              {'$set': update_dict})

    def on_future_rtn_transfer(self, real_withdraw_deposit):
        return
        # if real_withdraw_deposit.type == 0:
        #     self.xb_back_test_account.deposit += real_withdraw_deposit.money
        # else:
        #     self.xb_back_test_account.withdraw += real_withdraw_deposit.money

    def save_today_transfer(self, real_withdraw_deposit):
        real_withdraw_deposit_col = self.business_mongo.real_withdraw_deposit
        key = {'run_id': self.context.strategy_context.run_info.run_id, 'date': real_withdraw_deposit.date,
               'account_id': self.xb_back_test_account.account_id, 'type': real_withdraw_deposit.type,
               'account_type': real_withdraw_deposit.account_type}
        real_withdraw_deposit_col.update_one(key, {"$set":real_withdraw_deposit.__dict__}, upsert=True)
        self.init_withdraw_deposit()

    def update_asset(self, xb_back_test_account):
        self.xb_back_test_account.__dict__.update(xb_back_test_account.__dict__)
        if self.xb_back_test_account.start_capital == 0 or self.xb_back_test_account.start_capital is None:
            self.xb_back_test_account.start_capital = self.xb_back_test_account.total_profit - \
                                                      self.xb_back_test_account.today_deposit + \
                                                      self.xb_back_test_account.today_withdraw
            self.xb_back_test_account.yes_total_capital = self.xb_back_test_account.start_capital
            self.save_future_start_capital()

        if self.xb_back_test_account.today_deposit > 0:
            # 保存当天入金
            real_withdraw_deposit = XbRealWithdrawDeposit()
            real_withdraw_deposit.run_id = self.context.strategy_context.run_info.run_id
            real_withdraw_deposit.date = self.context.strategy_context.trade_date
            real_withdraw_deposit.type = 0
            real_withdraw_deposit.money = self.xb_back_test_account.today_deposit
            real_withdraw_deposit.account_id = self.xb_back_test_account.account_id
            real_withdraw_deposit.account_type = 1
            self.save_today_transfer(real_withdraw_deposit)

        if self.xb_back_test_account.today_withdraw > 0:
            # 保存当天出金
            real_withdraw_deposit = XbRealWithdrawDeposit()
            real_withdraw_deposit.run_id = self.context.strategy_context.run_info.run_id
            real_withdraw_deposit.date = self.context.strategy_context.trade_date
            real_withdraw_deposit.type = 1
            real_withdraw_deposit.account_type = 1
            real_withdraw_deposit.money = self.xb_back_test_account.today_withdraw
            real_withdraw_deposit.account_id = self.xb_back_test_account.account_id
            self.save_today_transfer(real_withdraw_deposit)

        self.update_total_capital()

    def update_positions(self, position_data_dict):
        """
        更新账号持仓
        :param position_data_dict:
        :return:
        """
        strategy_context = self.context.strategy_context
        self.long_position_dict.clear()
        self.short_position_dict.clear()
        self.xb_back_test_account.holding_pnl = 0
        self.xb_back_test_account.market_value = 0
        bar_dict = QuotationData.get_instance().bar_dict
        sub_list = list()
        for xb_back_test_position in position_data_dict.values():
            xb_back_test_position.gmt_create = strategy_context.trade_date
            xb_back_test_position.back_id = self.xb_back_test_account.mock_id
            xb_back_test_position.account_id = self.xb_back_test_account.account_id
            symbol = xb_back_test_position.contract_code
            xb_back_test_position.last_price = bar_dict[symbol].last
            if xb_back_test_position.last_price == 0:
                xb_back_test_position.last_price = xb_back_test_position.price

            xb_back_test_position.accumulate_profit = \
                xb_back_test_position.position * xb_back_test_position.round_lot * \
                (xb_back_test_position.last_price - xb_back_test_position.price)
            xb_back_test_position.market_value = xb_back_test_position.last_price * xb_back_test_position.position * \
                                                 xb_back_test_position.round_lot

            # 添加到订阅列表
            sub_list.append(symbol)
            if xb_back_test_position.direction == SIDE_BUY:
                # 多头
                if xb_back_test_position.position == 0:
                    if symbol in self.long_position_dict.keys():
                        self.long_position_dict.pop(symbol)
                        continue
                    else:
                        continue

                self.xb_back_test_account.holding_pnl += xb_back_test_position.holding_pnl
                self.xb_back_test_account.market_value += xb_back_test_position.market_value
                self.long_position_dict[symbol] = xb_back_test_position
                # if self.initEnd is False:
                #     self.strategy_long_position_dict[symbol] = copy.deepcopy(xb_back_test_position)

            elif xb_back_test_position.direction == SIDE_SELL:
                if xb_back_test_position.position == 0:
                    if symbol in self.short_position_dict.keys():
                        self.short_position_dict.pop(symbol)
                        continue
                    else:
                        continue

                self.xb_back_test_account.holding_pnl += xb_back_test_position.holding_pnl
                self.xb_back_test_account.market_value += xb_back_test_position.market_value
                self.short_position_dict[symbol] = xb_back_test_position
                # if self.initEnd is False:
                #     self.strategy_short_position_dict[symbol] = copy.deepcopy(xb_back_test_position)

        # print('当前总的持仓收益================》' + str(self.xb_back_test_account.holding_pnl))
        event_bus = self.context.event_bus
        event = Event(ConstantEvent.SYSTEM_FUTURE_QUOTATION_START_SUB, sub_symbol_list=sub_list)
        event_bus.publish_event(event)
        # TODO：判断第一次没有持仓会不会进来
        self.initEnd = True
        strategy_context.init_future_account_position_status(self.xb_back_test_account.account_id)

    def update_trade_positions(self, position_data_dict):
        """
        ctp回报更新补充持仓的保证金手续费等金额异步数据（仓位提前在报单和成交回报中处理）
        :param position_data_dict:
        :return:
        """
        sub_list = list()
        for xb_back_test_position in position_data_dict.values():
            symbol = xb_back_test_position.contract_code
            if xb_back_test_position.direction == SIDE_BUY:
                # 多头
                if symbol in self.long_position_dict.keys():
                    position_item = self.long_position_dict[symbol]
                else:
                    continue
            else:
                if symbol in self.short_position_dict.keys():
                    position_item = self.short_position_dict[symbol]
                else:
                    continue
            sub_list.append(symbol)
            position_item.open_cost = xb_back_test_position.open_cost
            position_item.position_cost = xb_back_test_position.position_cost
            position_item.hold_price = xb_back_test_position.hold_price
            position_item.price = xb_back_test_position.price
            position_item.cost = xb_back_test_position.cost
            position_item.margin = xb_back_test_position.margin
            # position_item.holding_pnl = xb_back_test_position.holding_pnl
            position_item.realized_pnl = xb_back_test_position.realized_pnl
            position_item.round_lot = xb_back_test_position.round_lot
            # 组合合约的要更新持仓
            if '&' in symbol:
                position_item.position = xb_back_test_position.position
                position_item.yd_position = xb_back_test_position.yd_position
                position_item.td_position = xb_back_test_position.td_position
                if position_item.position == 0:
                    if xb_back_test_position.direction == SIDE_BUY:
                        self.long_position_dict.pop(symbol)
                    else:
                        self.short_position_dict.pop(symbol)

        event_bus = self.context.event_bus
        event = Event(ConstantEvent.SYSTEM_FUTURE_QUOTATION_START_SUB, sub_symbol_list=sub_list)
        event_bus.publish_event(event)

    def refresh_position(self, bar_data):
        # print('行情==》' + str(bar_data.__dict__))
        if bar_data.last == 0:
            return

        if bar_data.symbol in self.long_position_dict.keys():
            xb_back_test_position = self.long_position_dict[bar_data.symbol]

            if xb_back_test_position.hold_price == 0 and xb_back_test_position.position != 0:
                return

            xb_back_test_position.last_price = bar_data.last
            xb_back_test_position.accumulate_profit = \
                xb_back_test_position.position * xb_back_test_position.round_lot * \
                (xb_back_test_position.last_price - xb_back_test_position.price)
            old_pnl = xb_back_test_position.holding_pnl
            xb_back_test_position.holding_pnl = \
                xb_back_test_position.position * xb_back_test_position.round_lot * \
                (xb_back_test_position.last_price - xb_back_test_position.hold_price)
            xb_back_test_position.settlement = bar_data.settle
            old_market_value = xb_back_test_position.market_value
            xb_back_test_position.market_value = \
                xb_back_test_position.last_price * xb_back_test_position.position * xb_back_test_position.round_lot

            self.xb_back_test_account.market_value += xb_back_test_position.market_value - old_market_value
            # print('合约:%s，偏差：%s,成本价：%s, ' %
            #       (str(xb_back_test_position.contract_code), str(xb_back_test_position.holding_pnl - old_pnl),
            #        str(xb_back_test_position.hold_price)))
            # print(xb_back_test_position.holding_pnl, old_pnl, xb_back_test_position.round_lot)
            self.xb_back_test_account.holding_pnl += xb_back_test_position.holding_pnl - old_pnl

            if xb_back_test_position.position == 0:
                self.long_position_dict.pop(xb_back_test_position.contract_code)

        if bar_data.symbol in self.short_position_dict.keys():
            xb_back_test_position = self.short_position_dict[bar_data.symbol]

            if xb_back_test_position.hold_price == 0 and xb_back_test_position.position != 0:
                return

            xb_back_test_position.last_price = bar_data.last
            xb_back_test_position.accumulate_profit = \
                xb_back_test_position.position * xb_back_test_position.round_lot * \
                (xb_back_test_position.price - xb_back_test_position.last_price)
            old_pnl = xb_back_test_position.holding_pnl
            xb_back_test_position.holding_pnl = \
                xb_back_test_position.position * xb_back_test_position.round_lot * \
                (xb_back_test_position.hold_price - xb_back_test_position.last_price)
            xb_back_test_position.settlement = bar_data.settle
            old_market_value = xb_back_test_position.market_value
            xb_back_test_position.market_value = \
                xb_back_test_position.last_price * xb_back_test_position.position * xb_back_test_position.round_lot

            self.xb_back_test_account.market_value += xb_back_test_position.market_value - old_market_value
            self.xb_back_test_account.holding_pnl += xb_back_test_position.holding_pnl - old_pnl

            if xb_back_test_position.position == 0:
                self.short_position_dict.pop(xb_back_test_position.contract_code)

        # print('holding_pnl==>', str(self.xb_back_test_account.holding_pnl))
        self.xb_back_test_account.total_profit = \
            self.xb_back_test_account.static_profit + self.xb_back_test_account.holding_pnl + \
            self.xb_back_test_account.realized_pnl - self.xb_back_test_account.cost + \
            self.xb_back_test_account.today_deposit - self.xb_back_test_account.today_withdraw
        self.xb_back_test_account.available_funds = self.xb_back_test_account.total_profit - \
                                                    self.xb_back_test_account.margin - self.xb_back_test_account.frozen_capital
        self.xb_back_test_account.daily_pnl = self.xb_back_test_account.total_profit - self.xb_back_test_account.yes_total_capital + \
                                              self.xb_back_test_account.today_withdraw - \
                                              self.xb_back_test_account.today_deposit
        self.xb_back_test_account.add_profit = \
            self.xb_back_test_account.total_profit - self.xb_back_test_account.start_capital + \
            self.xb_back_test_account.withdraw - \
            self.xb_back_test_account.deposit

    def update_total_capital(self):
        """
        更新账号总资产
        :return:
        """
        if self.xb_back_test_account.start_capital:
            self.update_profit()

    def update_profit(self):
        self.xb_back_test_account.daily_pnl = self.xb_back_test_account.total_profit - \
                                              self.xb_back_test_account.yes_total_capital + \
                                              self.xb_back_test_account.today_withdraw - \
                                              self.xb_back_test_account.today_deposit
        self.xb_back_test_account.add_profit = self.xb_back_test_account.total_profit - \
                                               self.xb_back_test_account.start_capital + \
                                               self.xb_back_test_account.withdraw - \
                                               self.xb_back_test_account.deposit

    def on_future_rtn_order(self, order):
        if hasattr(order, 'is_close_local') and order.is_close_local == 1:
            # 本地持仓
            pass
            return
        strategy_context = self.context.strategy_context
        self.handle_order_change_position(order, False)
        # if order.now_system_order == 1 and order.client_id == strategy_context.run_info.run_id:
        #     self.handle_order_change_position(order, True)

    def handle_order_change_position(self, order, is_strategy_position):
        # if is_strategy_position:
        #     short_position_dict = self.strategy_short_position_dict
        #     long_position_dict = self.strategy_long_position_dict
        # else:
        #     short_position_dict = self.short_position_dict
        #     long_position_dict = self.long_position_dict

        short_position_dict = self.short_position_dict
        long_position_dict = self.long_position_dict

        # 在查询持仓回调前提前处理冻结持仓问题
        if order.effect == CLOSE:  # 平仓
            if order.side == SIDE_BUY:  # 做多
                if order.order_book_id in short_position_dict.keys():
                    position_item = short_position_dict[order.order_book_id]
                else:
                    return
            else:  # 做空
                if order.order_book_id in long_position_dict.keys():
                    position_item = long_position_dict[order.order_book_id]
                else:
                    return

            if order.status == PartTradedNotQueueing or order.status == CANCELLED:
                if order.is_td_close == 1:
                    position_item.frozen_td_position -= order.unfilled_quantity
                    position_item.frozen_position -= order.unfilled_quantity
                    position_item.sellable = position_item.position - position_item.frozen_position
                else:
                    position_item.frozen_position -= order.unfilled_quantity
                    position_item.sellable = position_item.position - position_item.frozen_position
            elif order.status == ACTIVE and order.order_sys_id == '':
                if order.is_td_close == 1:
                    position_item.frozen_td_position += order.unfilled_quantity
                    position_item.frozen_position += order.unfilled_quantity
                    position_item.sellable = position_item.position - position_item.frozen_position
                else:
                    position_item.frozen_position += order.unfilled_quantity
                    position_item.sellable = position_item.position - position_item.frozen_position

    def on_future_rtn_trade(self, trade_data):
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
        key = {'run_id': trade_data.run_id, 'trade_id': trade_data.trade_id,
               'account': self.xb_back_test_account.account_id}
        trade_collection.update_one(key, {"$set":trade_data.__dict__}, upsert=True)

    # def handle_close_local_trade(self, trade_data):
    #     if trade_data.business == SIDE_BUY:  # 做多
    #         symbol_position_dict = self.strategy_short_position_dict
    #     else:
    #         symbol_position_dict = self.strategy_long_position_dict
    #
    #     symbol_position = symbol_position_dict[trade_data.contract_code]
    #     symbol_position.position -= trade_data.volume
    #     symbol_position.sellable -= trade_data.volume
    #     symbol_position.td_position -= trade_data.close_td_pos

    def handle_trade_change_position(self, account, trade, is_strategy_position):
        strategy_context = self.context.strategy_context
        # if is_strategy_position:
        #     short_position_dict = self.strategy_short_position_dict
        #     long_position_dict = self.strategy_long_position_dict
        # else:
        #     short_position_dict = self.short_position_dict
        #     long_position_dict = self.long_position_dict

        short_position_dict = self.short_position_dict
        long_position_dict = self.long_position_dict

        if trade.direction == CLOSE:  # 平仓
            if trade.business == SIDE_BUY:  # 做多
                if trade.contract_code in short_position_dict.keys():
                    position_item = short_position_dict[trade.contract_code]
                else:
                    return

            else:  # 做空
                if trade.contract_code in long_position_dict.keys():
                    position_item = long_position_dict[trade.contract_code]
                else:
                    return

            position_item.gmt_create = strategy_context.trade_date
            if trade.is_td_close == 1:
                position_item.position -= trade.volume
                position_item.td_position -= trade.volume
                position_item.frozen_td_position -= trade.volume
                position_item.frozen_position -= trade.volume
                position_item.sellable = position_item.position - position_item.frozen_position

            else:
                position_item.position -= trade.volume
                close_td_pos = trade.volume - position_item.yd_position if trade.volume > position_item.yd_position \
                    else 0
                position_item.yd_position -= trade.volume - close_td_pos
                position_item.td_position -= close_td_pos
                position_item.frozen_position -= trade.volume
                position_item.sellable = position_item.position - position_item.frozen_position

            if position_item.position == 0 and is_strategy_position:
                if trade.business == SIDE_BUY:  # 做多
                    short_position_dict.pop(trade.contract_code)
                else:
                    long_position_dict.pop(trade.contract_code)

        else:
            if trade.business == SIDE_BUY:  # 做多
                if trade.contract_code in long_position_dict.keys():
                    position_item = long_position_dict[trade.contract_code]
                else:
                    position_item = XbBacktestPosition()
                    position_item.contract_code = trade.contract_code
                    position_item.yd_position = 0
                    position_item.frozen_position = 0
                    position_item.frozen_td_position = 0
            else:  # 做空
                if trade.contract_code in short_position_dict.keys():
                    position_item = short_position_dict[trade.contract_code]
                else:
                    position_item = XbBacktestPosition()
                    position_item.contract_code = trade.contract_code
                    position_item.yd_position = 0
                    position_item.frozen_position = 0
                    position_item.frozen_td_position = 0

            position_item.gmt_create = strategy_context.trade_date
            position_item.account_id = account
            position_item.type = 1
            position_item.direction = trade.business
            position_item.position += trade.volume
            position_item.td_position += trade.volume
            position_item.sellable = position_item.position - position_item.frozen_position

            if trade.business == SIDE_BUY:  # 做多
                long_position_dict[trade.contract_code] = position_item
            else:
                short_position_dict[trade.contract_code] = position_item

    def day_start(self):
        pass

    def new_date(self):
        self.xb_back_test_account.gmt_create = self.context.strategy_context.trade_date
        self.xb_back_test_account.yes_total_capital = self.xb_back_test_account.total_profit
        self.xb_back_test_account.today_deposit = 0
        self.xb_back_test_account.today_withdraw = 0
        self.xb_back_test_account.daily_pnl = 0

    def end_date(self):
        pass
        # self.handle_strategy_position_change_day()

    def handle_strategy_position_change_day(self, day_end=True):
        pass
        # strategy_context = self.context.strategy_context
        # # 维护本地持仓，将今日仓位换为昨日仓位
        # for strategy_long_position in self.strategy_long_position_dict.values():
        #     if day_end or strategy_long_position.gmt_create is '' or strategy_long_position.gmt_create is \
        #             None or strategy_long_position.gmt_create != strategy_context.trade_date:
        #         strategy_long_position.yd_position += strategy_long_position.td_position
        #         strategy_long_position.td_position = 0
        #         strategy_long_position.gmt_create = strategy_context.trade_date
        #
        # for strategy_short_position in self.strategy_short_position_dict.values():
        #     if day_end or strategy_short_position.gmt_create is '' or strategy_short_position.gmt_create is \
        #             None or strategy_short_position.gmt_create != strategy_context.trade_date:
        #         strategy_short_position.yd_position += strategy_short_position.td_position
        #         strategy_short_position.td_position = 0
        #         strategy_short_position.gmt_create = strategy_context.trade_date

    def restore_save(self, mock_id):
        pass

    def restore_read(self, mock_id):
        pass
