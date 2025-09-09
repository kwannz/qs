from common.connector.mongodb_handler import DatabaseHandler as MongoClient
import six
from panda_backtest.backtest_common.data.future.base_future_info_map import BaseFutureInfoMap
from common.config.config import config


class FutureInfoMap(BaseFutureInfoMap):
    def __init__(self):
        self._cache = {}
        self.quotation_mongo_db = MongoClient(config).get_mongo_db()

    def __getitem__(self, key):
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
                {'symbol': str(key)}, {'emcode': 1, 'name': 1, 'ftmktsname': 1, 'deliverydate': 1, 'starttradedate': 1,
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
            instrument_info_cur = collection.find_one(
                {'ctpcode': str(key)}, {'emcode': 1, 'name': 1, 'ftmktsname': 1, 'deliverydate': 1, 'starttradedate': 1,
                                        'lasttradedate': 1, 'emcodetype': 1, 'contractmul': 1, 'listdate': 1,
                                        'fttransmargin': 1, 'ctpcode': 1, 'symbol': 1}).sort(
                [('starttradedate', -1)]).limit(1)
            if instrument_info_cur:
                instrument_info_list = list(instrument_info_cur)
                self._cache[key] = instrument_info_list[0]
                return instrument_info_list[0]
            else:
                instrument_info = dict()
                instrument_info['name'] = '未知'
                instrument_info['emcode'] = key
                instrument_info['contractmul'] = 1
                return instrument_info


# if __name__ == '__main__':
#     tset = FutureInfoMap()
#     name = tset.get_by_ctp_code('CF001')
#     print(name)
