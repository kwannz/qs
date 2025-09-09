#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 19-4-4 下午5:31
# @Author : wlb
# @File   : strategy_constant.py
# @desc   :

# 订单买卖方向
SIDE_BUY = 0
SIDE_SELL = 1

# 交易所
MKT_SZ = 'SZ'
MKT_SH = 'SH'
# MKT_CFE = 'CFE'
MKT_CFE = 'CFFEX'
# MKT_CZC = 'CZC'
MKT_CZC = 'CZCE'
MKT_DCE = 'DCE'
# MKT_SHF = 'SHF'
MKT_SHF = 'SHFE'
MKT_INE = 'INE'
MKT_GFE = 'GFEX'
MKT_UNKNOWN = '未知'

# 订单类型
MARKET = 1
LIMIT = 2

# 默认空值
EMPTY_STRING = ''
EMPTY_UNICODE = u''
EMPTY_INT = 0
EMPTY_FLOAT = 0.0

# 开平
OPEN = 0
CLOSE = 1

# 交易状态
WAIT = 0  # < 可撤
ACTIVE = 1  # < 可撤
FILLED = 2  # < 全成
CANCELLED = 3  # < 已撤
PartTradedQueueing = 4  # < 部分成交还在队列中
PartTradedNotQueueing = 5  # < 部分成交不在队列中
NoTradeQueueing = 6  # < 未成交还在队列中
NoTradeNotQueueing = 7  # < 未成交不在队列中
NotTouched = 8  # < 尚未触发
Touched = 9  # < 已触发
REJECTED = -1

