from common.connector.mongodb_handler import DatabaseHandler as MongoClient
from common.config.config import config
import six


class TradeDateData(object):
    def __init__(self):
        self._cache = {}
        self.quotation_mongo_db = MongoClient(config).get_mongo_db()
        self.cache_dict = dict()

    def is_trade_date(self, date):
        try:
            if date in self.cache_dict.keys():
                return self.cache_dict[date]
            self.cache_dict.clear()
            collection = self.quotation_mongo_db.trade_calendar
            next_date_cur = collection.find({
                'nature_date': date, 'is_trade': 1, "exchange": "SH"})
            next_date_list = list(next_date_cur)
            if len(next_date_list) <= 0:
                self.cache_dict[date] = False
                return False
            self.cache_dict[date] = True
            return True
        except Exception:
            return False
