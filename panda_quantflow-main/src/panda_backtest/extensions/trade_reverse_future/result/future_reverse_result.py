#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-27 下午5:02
# @Author : wlb
# @File   : trade_reverse_result.py
# @desc   :
from common.connector.mongodb_handler import DatabaseHandler
import logging

from panda_backtest.backtest_common.result.future.back_test.base_future_reverse_result import BaseFutureReverseResult
from common.config.config import config

class FutureReverseResult(BaseFutureReverseResult):

    def __init__(self, account):
        quotation_mongo_db = DatabaseHandler(config)  # mongodb客户端连接
        super().__init__(account, quotation_mongo_db)