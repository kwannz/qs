#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-27 下午5:02
# @Author : wlb
# @File   : trade_reverse_result.py
# @desc   :
from panda_backtest.backtest_common.result.fund.back_test.base_fund_reverse_result import BaseFundReverseResult
import logging

class FundReverseResult(BaseFundReverseResult):

    def __init__(self, account):
        super().__init__(account)