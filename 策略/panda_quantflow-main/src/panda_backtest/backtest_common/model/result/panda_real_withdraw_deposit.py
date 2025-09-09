#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2021/4/15 17:08
# @Author : wlb
# @File   : panda_real_withdraw_deposit.py
# @desc   :
from panda_backtest.backtest_common.constant.strategy_constant import *
import logging

class PandaRealWithdrawDeposit:
    run_id = EMPTY_STRING                           # 关联结果主键id
    account_id = EMPTY_STRING                       # 账号ID
    account_type = EMPTY_INT                        # 账号类型（0：股票，1：期货，2：基金，3：资产）
    type = EMPTY_INT                                # 出入金类型（0：入金，1：出金）
    money = EMPTY_FLOAT                             # 买卖（0：买  1：卖）
    date = EMPTY_STRING                             # 金额
