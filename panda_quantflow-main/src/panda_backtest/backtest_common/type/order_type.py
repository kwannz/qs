#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 19-4-4 下午5:33
# @Author : wlb
# @File   : order_type.py
# @desc   :

class MarketOrderStyle:
    """下单类型

    市价单
    """
    pass

class LimitOrderStyle:
    """下单类型

    限价单，指定价格
    """

    def __init__(self, limit_price):
        self.limit_price = limit_price
