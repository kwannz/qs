#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-4-12 上午10:43
# @Author : wlb
# @File   : symbol_info_map.py
# @desc   :
from common.connector.mongodb_handler import DatabaseHandler as MongoClient
import six
from utils.annotation.singleton_annotation import singleton
from common.connector.redis_client import RedisClient
import json
from panda_backtest.backtest_common.data.future.base_future_info_map import BaseFutureInfoMap


@singleton
class FutureInfoMap(BaseFutureInfoMap):
    def __init__(self):
        self._cache = {}
        self.quotation_mongo_db = MongoClient.get_mongo_db()
        self.redis_client = RedisClient()

    def __getitem__(self, key):
        if not isinstance(key, six.string_types):
            instrument_info = dict()
            instrument_info['name'] = '未知'
            return instrument_info

        try:
            return self._cache[key]
        except KeyError:

            collection = self.quotation_mongo_db.future_info
            instrument_info = collection.find_one(
                {'symbol': str(key)}, {'emcode': 1, 'name': 1, 'ftmktsname': 1, 'deliverydate': 1, 'starttradedate': 1,
                                       'lasttradedate': 1, 'emcodetype': 1, 'contractmul': 1, 'listdate': 1,
                                       'fttransmargin': 1, 'ctpcode': 1, 'symbol': 1})
            if instrument_info:
                self._cache[key] = instrument_info
                return instrument_info
            else:
                var_data = self.redis_client.getHashRedis('ctp_future_info', key)
                if var_data:
                    instrument_info = json.loads(var_data)
                else:
                    instrument_info = dict()
                    instrument_info['name'] = '未知'
                    instrument_info['emcode'] = key
                    instrument_info['contractmul'] = 1
                return instrument_info

    def get_by_ctp_code(self, key):
        if not isinstance(key, six.string_types):
            print('异常')
            instrument_info = dict()
            instrument_info['name'] = '未知'
            return instrument_info

        try:
            return self._cache[key]
        except KeyError:

            collection = self.quotation_mongo_db.future_info
            instrument_info = collection.find_one(
                {'ctpcode': str(key)}, {'emcode': 1, 'name': 1, 'ftmktsname': 1, 'deliverydate': 1, 'starttradedate': 1,
                                       'lasttradedate': 1, 'emcodetype': 1, 'contractmul': 1, 'listdate': 1,
                                       'fttransmargin': 1, 'ctpcode': 1, 'symbol': 1})
            if instrument_info:
                self._cache[key] = instrument_info
                return instrument_info
            else:
                instrument_info = dict()
                instrument_info['name'] = '未知'
                instrument_info['emcode'] = key
                instrument_info['contractmul'] = 1
                return instrument_info


if __name__ == '__main__':
    tset = FutureInfoMap()
    tset.czc_symbol_chagne('00001.CZC', 20180505)
