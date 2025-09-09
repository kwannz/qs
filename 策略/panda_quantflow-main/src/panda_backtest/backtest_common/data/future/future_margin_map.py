#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-4-12 上午10:43
# @Author : wlb
# @File   : symbol_info_map.py
# @desc   :
import six
import logging

from common.connector.mongodb_handler import DatabaseHandler
from panda_backtest.backtest_common.data.future.base_future_margin_map import BaseFutureMarginMap
from panda_backtest.util.annotation.singleton_annotation import singleton
from common.config.config import config

@singleton
class FutureMarginMap(BaseFutureMarginMap):
    def __init__(self):
        self._cache = {}
        self.quotation_mongo_db = DatabaseHandler(config)

    def process_symbol(self,symbol: str) -> str:
        if symbol.endswith(".SHFE"):
            return symbol.replace(".SHFE", ".SHF")
        elif symbol.endswith(".CFFEX"):
            return symbol.replace(".CFFEX", ".CFE")
        elif symbol.endswith(".CZCE"):
            return symbol.replace(".CZCE", ".CZC")
        elif symbol.endswith(".GFEX"):
            return symbol.replace(".GFEX", ".GFE")
        else:
            return symbol  # 如果没有匹配的后缀，返回原始符号
    def get_future_margin_info(self, symbol, trade_date):
        # collection = self.quotation_mongo_db.future_margin
        process_symbol=self.process_symbol(symbol)
        instrument_info = self.quotation_mongo_db.mongo_find_one(db_name="panda",collection_name="future_margin",query=
            {'symbol': str(process_symbol), 'trade_date': trade_date}, projection={'long_margin': 1, 'short_margin': 1, 'margin': 1})
        if instrument_info:
            instrument_info['name'] = symbol
            return instrument_info
        else:
            instrument_info = dict()
            instrument_info['name'] = '未知'
            instrument_info['trade_date'] = trade_date
            instrument_info['emcode'] = symbol
            return instrument_info

if __name__ == '__main__':
    future_margin_map = FutureMarginMap()
    print(future_margin_map.get_future_margin_info('SC2406.INE',20240415))