#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-4-12 上午10:43
# @Author : wlb
# @File   : symbol_info_map.py
# @desc   :
import six
import logging

from panda_backtest.util.annotation.singleton_annotation import singleton
from panda_backtest.backtest_common.data.future.base_future_info_map import BaseFutureInfoMap
import re
from common.connector.mongodb_handler import DatabaseHandler
from common.config.config import config

@singleton
class FutureInfoMap(BaseFutureInfoMap):
    def __init__(self, quotation_mongo_db):
        self._cache = {}
        self.quotation_mongo_db = quotation_mongo_db

    def __getitem__(self, key):
        if not isinstance(key, six.string_types):
            print('异常')
            instrument_info = dict()
            instrument_info['name'] = '未知'
            return instrument_info

        try:
            return self._cache[key]
        except KeyError:

            collection="future_info"
            instrument_info =  self.quotation_mongo_db.mongo_find_one(db_name="panda",collection_name=collection,query=
                {'symbolcode': str(key.split(".")[0])}, projection={'emcode': 1, 'name': 1, 'ftmktsname': 1, 'deliverydate': 1, 'starttradedate': 1,
                                       'lasttradedate': 1, 'emcodetype': 1, 'contractmul': 1, 'listdate': 1,
                                       'fttransmargin': 1, 'ftfirsttransmargin': 1, 'ftpricelimit': 1,
                                       'ftminpricechg': 1})
            if instrument_info:
                instrument_info['ftfirsttransmargin'] = extract_number(instrument_info['ftfirsttransmargin'])
                # instrument_info['emcode'] = key
                instrument_info['ftminpricechg'] = re.search(r"\d+(\.\d+)?", instrument_info['ftminpricechg']).group()
                self._cache[key] = instrument_info
                return instrument_info
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
            instrument_info_cur = collection.find(
                {'ctpcode': str(key)}, {'emcode': 1, 'name': 1, 'ftmktsname': 1, 'deliverydate': 1, 'starttradedate': 1,
                                        'lasttradedate': 1, 'emcodetype': 1, 'contractmul': 1, 'listdate': 1,
                                        'fttransmargin': 1, 'ctpcode': 1, 'symbol': 1}).sort(
                [('starttradedate', -1)]).limit(1)
            instrument_info_list = list(instrument_info_cur)
            if len(instrument_info_list) > 0:
                self._cache[key] = instrument_info_list[0]
                return instrument_info_list[0]
            else:
                instrument_info = dict()
                instrument_info['symbol'] = key
                instrument_info['name'] = '未知'
                instrument_info['emcode'] = key
                instrument_info['contractmul'] = 1
                self._cache[key] = instrument_info
                return instrument_info

def extract_number(ftfirsttransmargin):
    """
    从 instrument_info 中指定的键值中提取符合条件的数字。
    - 确保数字部分之间最多只有一个小数点。

    参数:
    instrument_info (dict): 包含字符串数据的字典。
    key (str): 需要提取数字的键。

    返回:
    int 或 float: 提取并转换后的数字。

    异常:
    ValueError: 如果找不到符合条件的数字部分，或中间有超过一个小数点。
    """

    # 找到第一个数字出现的位置
    first_digit_position = re.search(r'\d', ftfirsttransmargin).start()

    # 正则表达式来匹配符合条件的数字部分
    pattern = r'\d+(\.\d+)?'
    matches = list(re.finditer(pattern, ftfirsttransmargin))

    if matches:
        # 最后一个匹配到的数字部分的位置
        last_match = matches[-1]
        last_digit_position = last_match.end() - 1

        # 提取数字部分
        extracted_number = ftfirsttransmargin[first_digit_position:last_digit_position + 1]

        # 检查中间是否只有一个小数点或没有小数点
        if len(re.findall(r'\.', extracted_number)) <= 1:
            return float(extracted_number) if '.' in extracted_number else int(extracted_number)
        else:
            return 0
    else:
        return 0

if __name__ == '__main__':
    quotation_mongo_db = DatabaseHandler(config)
    future_info_map = FutureInfoMap(quotation_mongo_db)
    instrument_info = future_info_map['IM2410.CFE']
    print(instrument_info['ftfirsttransmargin'] /100)
