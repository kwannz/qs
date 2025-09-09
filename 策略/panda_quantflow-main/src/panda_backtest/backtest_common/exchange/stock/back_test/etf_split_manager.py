#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/9/16 16:44
# @Author : wlb
# @File   : etf_split_manager.py
# @desc   :
import datetime
import logging

import time

from panda_backtest.backtest_common.model.quotation.etf_split import ETFSplit
from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.backtest_common.system.event.event import Event, ConstantEvent
from common.config.config import config
from panda_backtest.util.time.time_util import TimeUtil

class ETFSplitManager(object):

    def __init__(self, quotation_mongo_db):
        self.quotation_mongo_db = quotation_mongo_db
        self.context = CoreContext.get_instance()

    def get_etf_split(self):
        old_time = time.time()
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info

        all_pos_list = list()
        for stock_account in strategy_context.stock_account_dict.values():
            for pos_item in stock_account.positions.values():
                if pos_item.stock_type == 2:
                    all_pos_list.append(pos_item.order_book_id)

        if run_info.standard_stock_type == 2:
            stand_symbol = strategy_context.run_info.benchmark
            all_pos_list.append(stand_symbol)

        all_pos_set = set(all_pos_list)

        if len(all_pos_set) == 0:
            return
        trade_date = strategy_context.trade_date

        event_bus = self.context.event_bus
        collection = "etf_split"
        etf_split_list = self.quotation_mongo_db.mongo_find(config["MONGO_DB"],collection_name=collection,query={'symbol': {'$in': list(all_pos_set)},
             'trade_date': trade_date},projection={'_id': 0, 'symbol': 1, 'divcvratio': 1})
        # etf_split_list = collection.find(
        #     {'symbol': {'$in': list(all_pos_set)},
        #      'trade_date': trade_date},
        #     {'_id': 0, 'symbol': 1, 'divcvratio': 1})
        for etf_split_dict in etf_split_list:
            etf_split = ETFSplit()
            if etf_split_dict['divcvratio'] is not None:
                etf_split.symbol = etf_split_dict['symbol']
                etf_split.divcvratio = etf_split_dict['divcvratio']
                # 推送分红事件
                event = Event(ConstantEvent.SYSTEM_ETF_SPLIT, etf_split=etf_split)
                event_bus.publish_event(event)

        # print('ETF拆分查询耗时===》' + str(time.time() - old_time))

