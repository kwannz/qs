#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-9 下午8:00
# @Author : wlb
# @File   : run_info.py
# @desc   : 策略运行信息

from panda_backtest.backtest_common.constant.strategy_constant import *
import logging

class RunInfo:
    # 策略id
    strategy_id = EMPTY_STRING
    # 策略id
    strategy_name = EMPTY_STRING
    # 运行id
    run_id = EMPTY_STRING
    # 运行类型（0：编译，1：运行）
    run_type = None
    # 策略开始运行日期（回测）
    start_date = EMPTY_INT
    # 策略结束运行日期（回测）
    end_date = EMPTY_INT
    # 策略频率（1d:日线，1M:分钟）
    frequency = EMPTY_STRING
    # 股票起始资金
    stock_starting_cash = EMPTY_FLOAT
    # 期货起始资金
    future_starting_cash = EMPTY_FLOAT
    # 基金起始资金
    fund_starting_cash = EMPTY_FLOAT
    # 滑点
    slippage = EMPTY_FLOAT
    # 滑点
    future_slippage = EMPTY_FLOAT
    # 保证金倍率
    margin_multiplier = EMPTY_FLOAT
    # 手续费倍率
    commission_multiplier = EMPTY_FLOAT
    # 基准
    benchmark = EMPTY_STRING
    # 撮合类型（0：当前bar收盘，1：当前bar开盘）
    matching_type = EMPTY_INT
    # 运行状态
    status = EMPTY_INT
    # 运行策略类型（0：回测，1：模拟）
    run_strategy_type = EMPTY_INT
    # 运行时间（0：交易时间，1：所有时间）
    date_type = EMPTY_INT
    # 账号类型（0:股票，1：期货，2：股票、期货，3：基金，4：股票基金，5：期货基金，6：所有）
    account_type = EMPTY_INT
    # 基准类型(0:股票,1:基金，2：期货)
    standard_type = EMPTY_INT
    # 基准股票类型（0:普通，1:国债, 2:etf,3:指数）
    standard_stock_type = EMPTY_INT
    # 股票账号
    stock_account = EMPTY_STRING
    # 期货账号
    future_account = EMPTY_STRING
    # 基金账号
    fund_account = EMPTY_STRING
    # 基金基本手续费配置
    rate_dict_data_str = EMPTY_STRING
    # 策略启动时间
    start_run_time = EMPTY_FLOAT
    # 产品id(实盘）
    product_id = EMPTY_STRING
    # 产品名称（实盘）
    product_name = EMPTY_STRING
    # 自定义标签
    custom_tag = EMPTY_STRING
