#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-2-26 上午9:43
# @Author : wlb
# @File   : panda_backtest_trade.py
# @desc   : 策略回测交易详情数据
from panda_backtest.backtest_common.constant.strategy_constant import *
import logging

class PandaBacktestTrade:
    back_id = EMPTY_STRING                          # 关联回测结果主键id
    account_id = EMPTY_STRING                       # 账号ID
    contract_code = EMPTY_STRING                    # 合约代码
    contract_name = EMPTY_STRING                    # 合约名称
    business = EMPTY_INT                            # 买卖（0：买  1：卖）
    volume = EMPTY_INT                              # 成交量
    price = EMPTY_FLOAT                             # 成交价
    cost = EMPTY_FLOAT                              # 费用
    type = EMPTY_INT                                # 类型（0：股票  1：期货）
    direction = EMPTY_INT                           # 期货方向（0：开  1：平）
    trade_date = EMPTY_STRING                       # 交易日期
    gmt_create = EMPTY_STRING                       # 交易日期
    gmt_create_time = EMPTY_STRING                  # 交易时间
    trade_id = EMPTY_STRING                         # 成交记录编号
    run_type = EMPTY_INT                            # 实盘和交易用到的类型
    status = EMPTY_INT                              # 实盘和交易用到的状态

    margin = EMPTY_FLOAT                            # 成交保证金(期货)
    round_lot = EMPTY_INT                           # 一手对应多少股
    close_td_pos = EMPTY_INT                        # 平今仓位
    is_td_close = EMPTY_INT                         # 是否平今
    order_id = EMPTY_STRING                         # 对应报单唯一表示
    order_sys_id = EMPTY_INT
    order_remark = EMPTY_STRING                     # 订单备注
    market = EMPTY_STRING
    client_id = EMPTY_STRING
    now_system_order = EMPTY_INT
    is_close_local = EMPTY_INT
    run_id = EMPTY_STRING
    stock_type = EMPTY_INT
