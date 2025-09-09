#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-2-25 下午5:56
# @Author : wlb
# @File   : panda_backtest_profit.py
# @desc   : 回测结果收益概览图标数据
from panda_backtest.backtest_common.constant.strategy_constant import *
import logging

class PandaBacktestProfit:
    back_id = EMPTY_STRING                          # 关联回测结果主键id
    account_id = EMPTY_STRING                       # 账号ID
    strategy_profit = EMPTY_FLOAT                   # 策略累计收益（百分比）
    csi_stock = EMPTY_FLOAT                         # 沪深300
    overful_profit = EMPTY_FLOAT                    # 超额收益（百分比）
    day_profit = EMPTY_FLOAT                        # 当日盈利（金额）
    day_loss = EMPTY_FLOAT                          # 当日亏损（金额）
    day_purchase = EMPTY_FLOAT                      # 当日买入（金额）
    day_put = EMPTY_FLOAT                           # 当日卖出（金额）
    gmt_create = EMPTY_STRING                       # 日期
    gmt_create_time = EMPTY_STRING                  # 时间

