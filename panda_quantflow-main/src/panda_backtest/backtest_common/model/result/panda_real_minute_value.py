#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-2-25 下午5:56
# @Author : wlb
# @File   : panda_simulation_minute_value.py.py
# @desc   : 回测结果收益概览图标数据
from panda_backtest.backtest_common.constant.strategy_constant import *
import logging

class PandaRealMinuteValue:
    run_id = EMPTY_STRING                           # 关联回测结果主键id
    total_value = EMPTY_FLOAT                       # 策略总权益
    add_profit = EMPTY_FLOAT                        # 累计收益
    daily_pnl = EMPTY_FLOAT                         # 今日收益
    date = EMPTY_STRING                             # 日期
    trade_date = EMPTY_STRING                       # 交易日
    minute = EMPTY_STRING                           # 时间

