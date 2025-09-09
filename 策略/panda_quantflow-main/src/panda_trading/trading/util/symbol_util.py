#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/5/29 下午5:38
# @Author : wlb
# @File   : symbol_util.py
# @desc   :
import re
import pandas as pd

from common.config.config import config
from common.connector.mongodb_handler import DatabaseHandler as  MongoClient


class SymbolUtil(object):
    __data_db = MongoClient(config).get_mongo_db()

    @classmethod
    def get_stock_list(cls):
        """
        只获取A股合约
        :return:
        """
        collection = cls.__data_db.stock_info
        stock_symbol_cur = collection.find(
            {'market': {'$in': ['SZ', 'SH']}})

        stock_symbol_df = pd.DataFrame(list(stock_symbol_cur))
        stock_symbol_list = stock_symbol_df['symbol'].tolist()
        if len(stock_symbol_list) > 0:
            return stock_symbol_list
        else:
            return None

    @classmethod
    def get_all_stock_list(cls):
        collection = cls.__data_db.instrument_info
        stock_symbol_cur = collection.find(
            {'market': {'$in': ['SZ', 'SH']}})

        stock_symbol_df = pd.DataFrame(list(stock_symbol_cur))
        stock_symbol_list = stock_symbol_df['symbol'].tolist()
        if len(stock_symbol_list) > 0:
            return stock_symbol_list
        else:
            return None

    @classmethod
    def get_stock_info(cls, symbol):
        collection = cls.__data_db.instrument_info
        stock_symbol_cur = collection.find(
            {'symbol': symbol}, {'_id': 0, 'name': 1, 'symbol': '1'}).limit(1)

        stock_symbol_list = list(stock_symbol_cur)
        if len(stock_symbol_list) > 0:
            return stock_symbol_list[0]
        else:
            return None

    @classmethod
    def get_future_code_and_type_list(cls, trade_date):
        collection = cls.__data_db.future_info
        future_symbol_cur = collection.find(
            {'starttradedate': {'$lte': str(trade_date)}, 'lasttradedate': {'$gte': str(trade_date)}},
            {'_id': 0, 'emcode': 1, 'emcodetype': '1'})

        future_symbol_df = pd.DataFrame(list(future_symbol_cur))
        if future_symbol_df.empty:
            return None, None
        future_code_list = future_symbol_df['emcode'].tolist()
        future_type_list = list(set(future_symbol_df['emcodetype'].tolist()))
        print(future_code_list)
        return future_code_list, future_type_list

    @classmethod
    def get_future_code_and_type_list_by_time(cls, start_date, end_date):
        collection = cls.__data_db.future_info
        future_symbol_cur = collection.find(
            {'starttradedate': {'$lte': str(end_date)}, 'lasttradedate': {'$gte': str(start_date)}},
            {'_id': 0, 'emcode': 1, 'emcodetype': '1'})

        future_symbol_df = pd.DataFrame(list(future_symbol_cur))
        if future_symbol_df.empty:
            return None, None
        future_code_list = future_symbol_df['emcode'].tolist()
        future_type_list = list(set(future_symbol_df['emcodetype'].tolist()))
        return future_code_list, future_type_list

    @staticmethod
    def symbol_to_ctp_code(symbol):
        """
        平台合约转ctp合约
        :param symbol:
        :return:
        """
        if len(symbol.split('.')) < 2:
            # 组合合约
            return symbol
        code = symbol.split('.')[0]
        market = symbol.split('.')[1]
        if market == 'SHF':
            ctp_code = code.lower()
        elif market == 'CZC':
            new = re.sub(r"\D", "", code)
            new = new[1:]
            ctp_code = re.sub(r"\d+", new, code).upper()
        elif market == 'DCE':
            ctp_code = code.lower()
        elif market == 'CFE':
            ctp_code = code
        elif market == 'INE':
            ctp_code = code.lower()
        else:
            ctp_code = code
        return ctp_code