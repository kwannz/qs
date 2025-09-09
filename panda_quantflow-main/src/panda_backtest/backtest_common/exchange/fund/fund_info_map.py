import re
import logging

import time

class FundInfoMap(object):

    def __init__(self, quotation_mongo_db):
        self.quotation_mongo_db = quotation_mongo_db
        self.fund_info_dict = dict()

    def get_fund_info(self, symbol):

        if symbol in self.fund_info_dict.keys():
            return self.fund_info_dict[symbol]

        old_time = time.time()
        collection = self.quotation_mongo_db.fund_info

        instrument_info = collection.find_one(
            {'symbol': str(symbol)},
            {'_id': 0, 'symbol': 1, 'redpay_date': 1, 'fund_name': 1, 'fund_type_level1_code': 1})
        if instrument_info is None:
            instrument_info = dict()
            instrument_info['fund_name'] = '未知'
            instrument_info['symbol'] = symbol
            self.fund_info_dict[symbol] = instrument_info
            return instrument_info
        else:
            instrument_info['symbol'] = symbol
            self.fund_info_dict[symbol] = instrument_info
        print('基金信息查询耗时===》' + str(time.time() - old_time))
        return instrument_info
