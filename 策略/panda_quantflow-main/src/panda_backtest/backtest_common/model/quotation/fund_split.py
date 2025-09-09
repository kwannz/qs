#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/9/16 16:48
# @Author : wlb
# @File   : fund_split.py
# @desc   :

from panda_backtest.backtest_common.constant.strategy_constant import *
import logging

class FundSplit:
    symbol = EMPTY_STRING                      # 证券代码
    split_date = EMPTY_STRING
    split_ratio = EMPTY_FLOAT                    # 拆分比例
