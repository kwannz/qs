import re
import logging

import time
from datetime import datetime
import pandas as pd
import json

from common.connector.mongodb_handler import DatabaseHandler
from common.config.config import config
from panda_backtest.util.time.time_util import TimeUtil

class FundRate:

    def __init__(self, quotation_mongo_db):
        self.quotation_mongo_db = quotation_mongo_db
        self.purchase_rate_df_dict = dict()
        self.redeem_rate_np_dict = dict()
        self.rate_dict_data = dict()
        self.purchase_rate_df_dict_cache = dict()
        self.redeem_rate_df_dict_cache = dict()

    def clear_cache_data(self):
        self.purchase_rate_df_dict_cache = dict()
        self.redeem_rate_df_dict_cache = dict()

    def init_data(self, rate_dict_data_str):
        default_rate_dict_data_str = '{"101401":{"purchase":0.015, "redeem":0.005},' \
                                     '"101402":{"purchase":0.015, "redeem":0.005}, ' \
                                     '"101403":{"purchase":0.006, "redeem":0.005}, ' \
                                     '"101404":{"purchase":0.015, "redeem":0.005}, ' \
                                     '"101405":{"purchase":0.015, "redeem":0.005}, ' \
                                     '"101406":{"purchase":0.015, "redeem":0.005}}'
        self.rate_dict_data = json.loads(default_rate_dict_data_str)

        if rate_dict_data_str is not None and rate_dict_data_str != '':
            custom_rate_dict = json.loads(rate_dict_data_str)
            self.rate_dict_data.update(custom_rate_dict)

    def get_purchase_rate_by_symbol(self, symbol, trade_date, trade_amount, fund_type):
        """
        申购费用
        :param symbol:
        :param trade_date:
        :param trade_amount:
        :return:
        """
        if symbol in self.purchase_rate_df_dict_cache.keys():
            symbol_rate_np = self.purchase_rate_df_dict_cache[symbol]
        else:
            if symbol in self.purchase_rate_df_dict.keys():
                all_date_symbol_rate_np = self.purchase_rate_df_dict[symbol]

                if all_date_symbol_rate_np is not None:
                    symbol_rate_np = all_date_symbol_rate_np[
                        ((all_date_symbol_rate_np['begin_date'] <= trade_date) | (
                            all_date_symbol_rate_np['begin_date'].isna()))
                        & ((all_date_symbol_rate_np['end_date'] >= trade_date) | (
                            all_date_symbol_rate_np['end_date'].isna()))]
                    self.purchase_rate_df_dict_cache[symbol] = symbol_rate_np
                else:
                    symbol_rate_np = None

            else:
                print('查询基金申购手续费==============》')
                collection = self.quotation_mongo_db.fund_fee

                db_query = dict()
                db_query['symbol'] = symbol
                db_query['rate_name1_code'] = 0

                data_cur = collection.find(
                    db_query,
                    {'_id': 0, 'insert_time': 0})

                if data_cur:
                    all_date_symbol_rate_np = pd.DataFrame(list(data_cur))

                    if all_date_symbol_rate_np.empty:
                        self.redeem_rate_np_dict[symbol] = None
                        symbol_rate_np = None
                    else:
                        self.purchase_rate_df_dict[symbol] = all_date_symbol_rate_np
                        symbol_rate_np = all_date_symbol_rate_np[
                            ((all_date_symbol_rate_np['begin_date'] <= trade_date) | (
                                all_date_symbol_rate_np['begin_date'].isna()))
                            & ((all_date_symbol_rate_np['end_date'] >= trade_date) | (
                                all_date_symbol_rate_np['end_date'].isna()))]

                        self.purchase_rate_df_dict_cache[symbol] = symbol_rate_np

                else:
                    self.purchase_rate_df_dict[symbol] = None
                    symbol_rate_np = None

        if symbol_rate_np is None or symbol_rate_np.empty:
            if fund_type in self.rate_dict_data.keys():
                return trade_amount * float(self.rate_dict_data[fund_type]['purchase'])
            return 10
        else:
            symbol_rate_np = symbol_rate_np[
                ((symbol_rate_np['app_minamt'] <= trade_amount) | (symbol_rate_np['app_minamt'].isna()))
                & ((symbol_rate_np['app_maxamt'] > trade_amount) | (symbol_rate_np['app_maxamt'].isna()))]

            if symbol_rate_np is None or symbol_rate_np.empty:
                if fund_type in self.rate_dict_data.keys():
                    return trade_amount * float(self.rate_dict_data[fund_type]['purchase'])
                return 10
            result = symbol_rate_np.iloc[0]
            if result['costcalc_mode_code'] == 0:
                return float(result['cost'])
            else:
                return float(result['cost'] * trade_amount)

    def get_redeem_rate_by_symbol(self, symbol, trade_date, trade_amount, position_day, fund_type):
        """
        赎回费用
        :param symbol:
        :param trade_date:
        :param trade_amount:
        :param position_day:
        :return:
        """
        if symbol in self.redeem_rate_np_dict.keys():
            symbol_rate_np = self.redeem_rate_np_dict[symbol]
        else:
            if symbol in self.redeem_rate_np_dict:
                all_date_symbol_rate_np = self.redeem_rate_np_dict[symbol]

                if all_date_symbol_rate_np is not None:
                    symbol_rate_np = all_date_symbol_rate_np[
                        ((all_date_symbol_rate_np['begin_date'] <= trade_date) | (
                            all_date_symbol_rate_np['begin_date'].isna()))
                        & ((all_date_symbol_rate_np['end_date'] >= trade_date) | (
                            all_date_symbol_rate_np['end_date'].isna()))]
                else:
                    symbol_rate_np = None

            else:
                collection = self.quotation_mongo_db.fund_fee

                db_query = dict()
                db_query['symbol'] = symbol
                db_query['rate_name1_code'] = 1

                print('查询基金赎回手续费==============》')
                data_cur = collection.find(
                    db_query,
                    {'_id': 0, 'insert_time': 0})

                if data_cur:
                    all_date_symbol_rate_np = pd.DataFrame(list(data_cur))

                    if all_date_symbol_rate_np.empty:
                        self.redeem_rate_np_dict[symbol] = None
                        symbol_rate_np = None
                    else:
                        self.redeem_rate_np_dict[symbol] = all_date_symbol_rate_np
                        symbol_rate_np = all_date_symbol_rate_np[
                            ((all_date_symbol_rate_np['begin_date'] <= trade_date) | (
                                all_date_symbol_rate_np['begin_date'].isna()))
                            & ((all_date_symbol_rate_np['end_date'] >= trade_date) | (
                                all_date_symbol_rate_np['end_date'].isna()))]
                else:
                    self.redeem_rate_np_dict[symbol] = None
                    symbol_rate_np = None

        if symbol_rate_np is None or symbol_rate_np.empty:
            if fund_type in self.rate_dict_data.keys():
                return trade_amount * float(self.rate_dict_data[fund_type]['purchase'])
            return 10
        else:
            symbol_rate_np = symbol_rate_np[
                ((symbol_rate_np['hold_minperiod'] <= position_day) | (symbol_rate_np['hold_minperiod'].isna()))
                & ((symbol_rate_np['hold_maxperiod'] > position_day) | (
                    symbol_rate_np['hold_maxperiod'].isna()))]
            if symbol_rate_np is None or symbol_rate_np.empty:
                if fund_type in self.rate_dict_data.keys():
                    return trade_amount * float(self.rate_dict_data[fund_type]['purchase'])
                return 10
            if symbol_rate_np.iloc[0]['costcalc_mode_code'] == 0:
                return float(symbol_rate_np.iloc[0]['cost'])
            else:
                return float(symbol_rate_np.iloc[0]['cost'] * trade_amount)

if __name__ == '__main__':
    quotation_mongo_db = DatabaseHandler(config)
    fund_rate = FundRate(quotation_mongo_db)
    fund_rate.init_data(None)
    # TODO
    # res = fund_rate.get_purchase_rate_by_symbol('000311.OF', 20191205, 2000000, '101403')
    res1 = fund_rate.get_redeem_rate_by_symbol('000311.OF', '20200605', 100, 6, '101403')
    # res = fund_rate.get_redeem_rate_by_symbol('110030.OF', 20180103, 120028.37796871999, 364, '101403')
    # res1 = fund_rate.get_redeem_rate_by_symbol('000311.OF', 20191205, 1500, 30, '101403')
    print(res1)
    # print(res1)