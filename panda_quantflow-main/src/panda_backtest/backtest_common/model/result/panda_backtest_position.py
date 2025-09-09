#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-2-26 上午10:55
# @Author : wlb
# @File   : panda_backtest_position.py
# @desc   :
from panda_backtest.backtest_common.constant.strategy_constant import *
import logging

class PandaBacktestPosition:
    back_id = EMPTY_STRING                              # 关联回测结果主键id
    account_id = EMPTY_STRING                           # 账号ID
    contract_code = EMPTY_STRING                        # 合约代码
    contract_name = EMPTY_STRING                        # 合约名称
    type = EMPTY_INT                                    # 类型（0：股票  1：期货 2:基金）
    direction = EMPTY_INT                               # 期货方向（0：多  1：空）
    price = EMPTY_FLOAT                                 # 开仓成交价
    position = EMPTY_INT                                # 仓位
    last_price = EMPTY_FLOAT                            # 最新价
    market_value = EMPTY_FLOAT                          # 市值
    accumulate_profit = EMPTY_FLOAT                     # 累计盈亏
    margin = EMPTY_FLOAT                                # 保证金(期货)
    cost = EMPTY_FLOAT                                  # 费用
    gmt_create = EMPTY_STRING                           # 交易日期
    sellable = EMPTY_INT                                # 中间变量，可平量
    settlement = EMPTY_FLOAT                            # 结算价
    pre_settlement = EMPTY_FLOAT                        # 上一结算价(期货)

    holding_pnl = EMPTY_FLOAT                           # 持仓盈亏(持仓均价为准)
    realized_pnl = EMPTY_FLOAT                          # 平仓盈亏(每日累计)
    yd_position = EMPTY_INT                             # 昨日仓位(期货)
    td_position = EMPTY_INT                             # 今日仓位(期货)
    hold_price = EMPTY_INT                              # 持仓均价(期货,昨日仓位以昨日结算价,加上今日仓位)
    round_lot = EMPTY_INT                               # 一手对应多少股(期货)

    frozen_position = EMPTY_INT                         # 冻结仓位
    frozen_td_position = EMPTY_INT                      # 今日冻结仓位
    frozen_yd_position = EMPTY_INT                      # 今日冻结仓位

    open_cost = EMPTY_INT
    position_cost = EMPTY_INT
    position_detail_list = list()                       # 基金仓位详细日期
    buy_margin = EMPTY_FLOAT                            # 组合持仓多头保证金
    sell_margin = EMPTY_FLOAT                           # 组合持仓空头保证金
    stock_type = EMPTY_INT                              # 股票具体类型（0:普通，1:国债, 2:etf,3:指数）

