#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-6 下午9:45
# @Author : wlb
# @File   : dividend.py
# @desc   : 分配除权信息

from panda_backtest.backtest_common.constant.strategy_constant import *
import logging

class Dividend:
    symbol = EMPTY_STRING                      # 证券代码
    ann_date = EMPTY_STRING                    # 公告日期
    end_date = EMPTY_STRING                    # 分红年度截至日
    process_stauts = EMPTY_STRING              # 事件进程
    publish_date = EMPTY_STRING                # 分红实施公告日
    record_date = EMPTY_STRING                 # 股权登记日
    exdiv_date = EMPTY_STRING                  # 除权除息日
    cash = EMPTY_FLOAT                       # 每十股分红(税前)
    cash_div_tax = EMPTY_FLOAT                   # 每十股分红(税后）
    share_ratio = EMPTY_FLOAT                # 送股比例
    share_trans_ratio = EMPTY_FLOAT          # 转赠比例
    cashpay_date = EMPTY_STRING                # 派现日
    bonus_list_date = EMPTY_STRING             # 送股上市日
    fund_unit_ataxdev = EMPTY_FLOAT
