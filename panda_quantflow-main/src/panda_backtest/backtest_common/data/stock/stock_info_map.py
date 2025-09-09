#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-4-12 上午10:43
# @Author : wlb
# @File   : symbol_info_map.py
# @desc   :
import time
import logging

from panda_backtest.util.annotation.singleton_annotation import singleton
from panda_backtest.backtest_common.data.stock.base_stock_info_map import BaseStockInfoMap
from common.config.config import config

@singleton
class StockInfoMap(BaseStockInfoMap):
    def __init__(self, quotation_mongo_db):
        self._cache = {}
        self.quotation_mongo_db = quotation_mongo_db

    def __getitem__(self, key):
        if key in self._cache.keys():
            return self._cache[key]
        else:
            # TODO: 不同环境
            # collection = self.quotation_mongo_db.stock_info_new
            # collection = self.quotation_mongo_db.stock_info
            # start = time.time()

            instrument_info = self.quotation_mongo_db.mongo_find_one(db_name=config["MONGO_DB"],
                                                                     collection_name="stock_info_new",
                                                                     query={'symbol': str(key)},
                                                                     projection={'_id': 0, 'symbol': 1, 'name': 1, 'type': 1}
                                                                     )
            if instrument_info:
                self._cache[key] = instrument_info
                # print('股票基本信息耗时：' + str(time.time() - start))
                return instrument_info
            else:
                instrument_info = dict()
                instrument_info['name'] = '未知'
                instrument_info['type'] = 0
                self._cache[key] = instrument_info
                return instrument_info
