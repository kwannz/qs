#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-23 上午11:08
# @Author : wlb
# @File   : quotation_data.py
# @desc   : 行情数据容器

class QuotationData(object):
    _instance = None

    def __init__(self):
        self.bar_dict = None
        self.static_bar_dict = None         # 模拟盘可能用到，暂时保留

    @classmethod
    def get_instance(cls):
        if QuotationData._instance is None:
            QuotationData._instance = QuotationData()
        return QuotationData._instance

    def init_bar_dict(self, bar_dict):
        self.bar_dict = bar_dict
