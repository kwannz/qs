#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/9/16 16:44
# @Author : wlb
# @File   : fund_split_manager.py
# @desc   :
import datetime
import logging

import time

from panda_backtest.backtest_common.model.quotation.fund_split import FundSplit
from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.backtest_common.system.event.event import Event, ConstantEvent
from panda_backtest.util.time.time_util import TimeUtil

class FundSplitManager(object):

    def __init__(self, quotation_mongo_db):
        self.quotation_mongo_db = quotation_mongo_db
        self.context = CoreContext.get_instance()

    def get_fund_split(self):
        old_time = time.time()
        strategy_context = self.context.strategy_context
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
        trade_date = strategy_context.trade_date

        event_bus = self.context.event_bus
        collection = self.quotation_mongo_db.fund_split
        fund_split_list = collection.find(
            {'symbol': {'$in': list(all_pos_set)},
             'split_date': trade_date},
            {'_id': 0, 'symbol': 1, 'split_ratio': 1})
        for fund_split_dict in fund_split_list:
            fund_split = FundSplit()
            if fund_split_dict['split_ratio'] is not None:
                fund_split.symbol = fund_split_dict['symbol']
                fund_split.split_ratio = fund_split_dict['split_ratio']
                # 推送分红事件
                event = Event(ConstantEvent.SYSTEM_FUND_SPLIT, fund_split=fund_split)
                event_bus.publish_event(event)

        print('基金拆分查询耗时===》' + str(time.time() - old_time))

