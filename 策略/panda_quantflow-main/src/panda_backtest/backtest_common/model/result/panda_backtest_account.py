#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-2-26 上午11:59
# @Author : wlb
# @File   : panda_backtest_account.py
# @desc   :
from panda_backtest.backtest_common.constant.strategy_constant import *
import logging

class PandaBacktestAccount:
    type = EMPTY_INT                                      # 类型（0：股票  1：期货 2:统计, 3:基金）
    back_id = EMPTY_STRING                                # 关联回测结果主键id
    account_id = EMPTY_STRING                             # 账号ID
    available_funds = EMPTY_FLOAT                         # 可用资金
    total_profit = EMPTY_FLOAT                            # 总权益
    static_profit = EMPTY_FLOAT                           # 静态权益
    market_value = EMPTY_FLOAT                            # 市值
    cost = EMPTY_FLOAT                                    # 费用
    gmt_create = EMPTY_STRING                             # 交易日期
    mock_id = EMPTY_STRING                                # 实盘和
    frozen_capital = EMPTY_FLOAT                          # 冻结资金
    margin = EMPTY_FLOAT                                  # 保证金
    buy_margin = EMPTY_FLOAT                              # 多头保证金
    sell_margin = EMPTY_FLOAT                             # 空头保证金
    add_profit = EMPTY_FLOAT                              # 累计盈亏

    start_capital = EMPTY_FLOAT
    daily_pnl = EMPTY_FLOAT                               # 当日盈亏，当日持仓盈亏 + 当日平仓盈亏 - 当日费用
    holding_pnl = EMPTY_FLOAT                             # 当日持仓盈亏
    realized_pnl = EMPTY_FLOAT                            # 当日平仓盈亏
    yes_total_capital = EMPTY_FLOAT
    no_settle_total_capital = EMPTY_FLOAT
    deposit = EMPTY_FLOAT                                  # 入金金额
    withdraw = EMPTY_FLOAT                                  # 出金金额
    today_deposit = EMPTY_FLOAT                                  # 今日入金金额
    today_withdraw = EMPTY_FLOAT                                  # 今日出金金额
    buy_market_value = EMPTY_FLOAT
    sell_market_value = EMPTY_FLOAT

