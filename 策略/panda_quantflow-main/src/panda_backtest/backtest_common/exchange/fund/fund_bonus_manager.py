import re
import logging

import time
from datetime import datetime

from panda_backtest.backtest_common.model.quotation.dividend import Dividend
from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.backtest_common.system.event.event import Event, ConstantEvent
from panda_backtest.util.time.time_util import TimeUtil

class FundBonusManager(object):

    def __init__(self, quotation_mongo_db):
        self.quotation_mongo_db = quotation_mongo_db
        self.context = CoreContext.get_instance()

    def get_fund_bonus(self):
        strategy_context = self.context.strategy_context

        trade_date = strategy_context.trade_date

        event_bus = self.context.event_bus
        run_info = strategy_context.run_info

        all_pos_list = list()
        for fund_account in strategy_context.fund_account_dict.values():
            all_pos_list.extend(fund_account.positions.keys())

        if run_info.standard_type == 2:
            stand_symbol = strategy_context.run_info.benchmark
            all_pos_list.append(stand_symbol)

        all_pos_set = set(all_pos_list)

        if len(all_pos_set) == 0:
            return

        collection = self.quotation_mongo_db.fund_dividend
        dividend_list = collection.find(
            {'symbol': {'$in': list(all_pos_set)},
             'out_right_date': trade_date},
            {'_id': 0, 'symbol': 1, 'unit_ataxdev': 1})
        for dividend_dict in dividend_list:
            dividend = Dividend()
            dividend.symbol = dividend_dict['symbol']
            dividend.fund_unit_ataxdev = dividend_dict['unit_ataxdev']

            # 推送分红事件
            event = Event(ConstantEvent.SYSTEM_FUND_DIVIDEND, dividend=dividend)
            event_bus.publish_event(event)
        # print('基金分红查询耗时===》' + str(time.time() - old_time))
