#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2021/5/18 9:42
# @Author : wlb
# @File   : future_trade_time_verify.py
# @desc   :
import datetime
import logging

import pickle
import re
import time

from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY, CLOSE, REJECTED
from panda_backtest.backtest_common.constant.string_constant import FUTURE_ORDER_FAILED_MESSAGE, SYMBOL_NOT_TRADE_IN_THIS_TIME
from panda_backtest.backtest_common.data.future.future_info_map import FutureInfoMap
from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData
from panda_backtest.backtest_common.system.context.core_context import CoreContext

class FutureTradeTimeVerify(object):
    def __init__(self, quotation_mongo_db, redis_client):
        self.context = CoreContext.get_instance()
        self.future_info_map = FutureInfoMap(quotation_mongo_db)
        self.pz_dict = dict()
        self.redis_client = redis_client

    def init_future_trade_time(self):
        res = self.redis_client.getRedis('future_trade_time_dict')
        if res:
            self.pz_dict = pickle.loads(res)

    def can_submit_order(self, account, order_result):
        strategy_context = self.context.strategy_context
        trade_date = strategy_context.trade_date
        if order_result.side == SIDE_BUY:
            order_side = '买入'
        else:
            order_side = '卖出'
        if order_result.effect == CLOSE:
            order_effect = '平仓'
        else:
            order_effect = '开仓'

        symbol = order_result.order_book_id
        is_trade_time = False
        code_type = re.sub(r'\d+', '', symbol.split(".")[0])  # 转换成品种名，如AP
        if code_type in self.pz_dict.keys():
            time_stamp = datetime.datetime.now()
            now_time = time_stamp.strftime('%H:%M')
            is_trade_time = self.pz_dict[code_type].overlaps_point(time.strptime(now_time, "%H:%M"))

        if is_trade_time:
            future_info = self.future_info_map[symbol]
            if future_info['starttradedate'] < trade_date < future_info['lasttradedate']:
                return True

        order_result.status = REJECTED
        message = FUTURE_ORDER_FAILED_MESSAGE % (order_result.order_book_id, str(order_result.quantity),
                                                 account, order_effect, order_side, order_result.order_id,
                                                 SYMBOL_NOT_TRADE_IN_THIS_TIME)
        order_result.message = message
        return False
