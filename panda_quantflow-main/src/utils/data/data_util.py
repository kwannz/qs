import pymongo

from common.connector.mongodb_handler import DatabaseHandler as  MongoClient
from common.config.config import config


class DateUtil(object):
    __quotation_db = MongoClient(config).get_mongo_db()

    @classmethod
    def get_pre_date(cls, trade_date):
        trade_cal_col = cls.__quotation_db.trade_calendar
        pre_trade_date_cur = trade_cal_col.find(
            {'nature_date': {'$lt': str(trade_date)}, 'is_trade': 1, "exchange": "SH"}) \
            .sort([('nature_date', -1)]).limit(1)
        pre_trade_date_list = list(pre_trade_date_cur)
        if len(pre_trade_date_list) <= 0:
            return None
        return pre_trade_date_list[0]['nature_date']

    @classmethod
    def get_next_trade_date(cls, trade_date, operate='$gt'):
        collection = cls.__quotation_db.trade_calendar
        next_date_cur = collection.find({
            'nature_date': {operate: str(trade_date)}, 'is_trade': 1, "exchange": "SH"}).sort('nature_date',
                                                                                              pymongo.ASCENDING).limit(
            1)
        next_date_list = list(next_date_cur)
        if len(next_date_list) <= 0:
            return None
        return next_date_list[0]['nature_date']


    @staticmethod
    def hand_str_list(mes_str, split_str):
        if isinstance(mes_str, list):
            return mes_str
        else:
            return mes_str.split(split_str)
