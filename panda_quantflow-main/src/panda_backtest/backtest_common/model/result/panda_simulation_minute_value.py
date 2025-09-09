#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-2-25 下午5:56
# @Author : wlb
# @File   : panda_simulation_minute_value.py.py
# @desc   : 回测结果收益概览图标数据
from panda_backtest.backtest_common.constant.strategy_constant import *
import logging

class PandaSimulationMinuteValue:
    simulation_id = EMPTY_STRING                          # 关联回测结果主键id
    value = EMPTY_FLOAT                   # 策略收益（百分比）
    date = EMPTY_STRING                       # 日期
    minute = EMPTY_STRING                  # 时间
    # today_profit = EMPTY_FLOAT                      # 今日收益（百分比）

