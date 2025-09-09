#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/6/16 下午4:50
# @Author : wlb
# @File   : date_util.py
# @desc   :
import pymongo
import logging

from common.connector.mongodb_handler import DatabaseHandler
from common.config.config import config


class DateUtil(object):
    _quotation_db = DatabaseHandler(config)

    @classmethod
    def get_pre_date(cls, trade_date,pre_number=1):
        curr_trade_date_cur = cls._quotation_db.mongo_find_one(db_name="panda", collection_name="trading_calendar_all",
                                                               query={
                                                                   'trading_date': str(trade_date),
                                                               })
        if not curr_trade_date_cur:
            return None
        sort_dex=curr_trade_date_cur['sort_idx']-pre_number
        pre_trade_date = cls._quotation_db.mongo_find_one(db_name="panda", collection_name="trading_calendar_all",
                                                               query={
                                                                   'sort_idx': int(sort_dex),
                                                               })
        return pre_trade_date['trading_date']

    @classmethod
    def get_next_trade_date(cls, trade_date,next_number=1):
        curr_trade_date_cur = cls._quotation_db.mongo_find_one(db_name="panda", collection_name="trading_calendar_all",
                                                              query={
                                                                  'trading_date': str(trade_date),
                                                              })
        if not curr_trade_date_cur:
            return None
        sort_dex = curr_trade_date_cur['sort_idx'] + next_number
        trade_date = cls._quotation_db.mongo_find_one(db_name="panda", collection_name="trading_calendar_all",
                                                          query={
                                                              'sort_idx': int(sort_dex),
                                                          })
        return trade_date['trading_date']


if __name__ == '__main__':
    print(DateUtil.get_pre_date('20240531'))
