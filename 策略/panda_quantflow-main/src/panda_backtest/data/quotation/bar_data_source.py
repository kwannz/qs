#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-7 下午3:42
# @Author : wlb
# @File   : bar_map.py
# @desc   : 行情自定义字典
import calendar
import logging

import datetime
import time
import traceback
from panda_backtest.backtest_common.data.quotation.back_test.base_bar_data_source import BaseBarDataSource
from panda_backtest.backtest_common.model.quotation.daily_quotation_data import DailyQuotationData
from panda_backtest.backtest_common.model.quotation.bar_quotation_data import BarQuotationData
from common.connector.mongodb_handler import DatabaseHandler
from common.config.config import config
from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.util.annotation.singleton_annotation import singleton
import pymongo
from panda_backtest.util.time.time_util import TimeUtil
from panda_backtest.backtest_common.data.stock.stock_info_map import StockInfoMap
from sympy import false


class BarDataSource(BaseBarDataSource):
    def __init__(self):
        self.quotation_mongo_db = DatabaseHandler(config=config)
        self.stock_daily_bar = dict()
        self.future_daily_bar = dict()
        self.future_all_minute_bar = dict()
        self.stock_minute_bar = dict()
        self.fund_daily_bar = dict()
        self.context = CoreContext.get_instance()
        self.stock_info_map = StockInfoMap(self.quotation_mongo_db)
        self.last_field = 'close'

    def change_last_field(self, field_type):
        if field_type == 0:
            self.last_field = 'open'
        else:
            self.last_field = 'close'

    # def get_stock_daily_bar(self, symbol, trade_date):
    #     # start = time.time()
    #
    #     if symbol in self.stock_daily_bar.keys():
    #         bar = self.stock_daily_bar[symbol]
    #         bar.last = getattr(bar, self.last_field)
    #         return bar
    #     else:
    #
    #         stock_type = self.stock_info_map[symbol]['type']
    #
    #         if stock_type == 1:
    #             collection = "index_daily_price"
    #             # collection = "stock_market"
    #         elif stock_type == 2:
    #             collection = "etf_daily_quotation_v2"
    #         else:
    #             # collection = "stock_daily_quotation"
    #             collection = "stock_market_2"
    #         try:
    #             bar_dict_list = self.quotation_mongo_db.mongo_find(
    #                 db_name=config["MONGO_DB"],
    #                 collection_name=collection,
    #                 query={"symbol": symbol, "date": trade_date},
    #                 projection={'_id': 0}
    #             )
    #
    #             if not bar_dict_list:
    #                 # 查询结果为空，跳过（return 或 continue 看语境）
    #                 print(f"[SKIP] 无数据：symbol={symbol}, date={trade_date}, collection={collection}")
    #                 return None  # 或 continue，如果在循环中
    #             bar_dict = bar_dict_list[0]
    #         except Exception as e:
    #             print("symbol"+symbol, "date"+ trade_date,"collection_name"+collection)
    #         bar = DailyQuotationData()
    #         if bar_dict:
    #             bar.__dict__ = bar_dict
    #             bar.last = bar_dict[self.last_field]
    #             self.stock_daily_bar[symbol] = bar
    #         else:
    #             self.stock_daily_bar[symbol] = bar
    #         # print('股票日线查询====》合约：%s,耗时：%s' % (symbol, str(time.time() - start)))
    #         return bar
    def get_stock_daily_bar(self, symbol, trade_date):
        # 缓存命中，直接返回
        if symbol in self.stock_daily_bar:
            bar = self.stock_daily_bar[symbol]
            bar.last = getattr(bar, self.last_field)
            return bar

        # 未命中缓存，准备查询
        stock_type = self.stock_info_map[symbol]['type']
        if stock_type == 1:
            collection = "index_daily_price"
        elif stock_type == 2:
            collection = "etf_daily_quotation_v2"
        else:
            collection = "stock_market"

        bar_dict = None  # ✅ 先初始化，避免 try 异常后 bar_dict 未定义

        try:
            bar_dict_list = self.quotation_mongo_db.mongo_find(
                db_name=config["MONGO_DB"],
                collection_name=collection,
                query={"symbol": symbol, "date": trade_date},
                projection={'_id': 0}
            )

            if not bar_dict_list:
                print(f"[SKIP] 无数据：symbol={symbol}, date={trade_date}, collection={collection}")
                return None

            bar_dict = bar_dict_list[0]

        except Exception as e:
            print(f"[ERROR] 查询异常：symbol={symbol}, date={trade_date}, collection={collection}")
            print("Exception:", e)

        bar = DailyQuotationData()
        if bar_dict:
            bar.__dict__ = bar_dict
            bar.last = bar_dict.get(self.last_field)
        self.stock_daily_bar[symbol] = bar

        return bar
    def get_stock_minute_bar(self, symbol, trade_date, trade_time):
        # start = time.time()
        trade_time = int(trade_time)
        if symbol in self.stock_minute_bar.keys():
            if trade_time in self.stock_minute_bar[symbol].keys():
                bar = self.stock_minute_bar[symbol][trade_time]
                bar.last = getattr(bar, self.last_field)
                return bar
            else:
                return BarQuotationData()

        # 获取一天的数据

        stock_type = self.stock_info_map[symbol]['type']
        if stock_type == 1:
            # TODO
            collection = self.quotation_mongo_db.stock_quotation_min_data
        else:
            collection = self.quotation_mongo_db.stock_quotation_min_data

        bar_dict = collection.find({"trade_date": int(trade_date), "symbol": symbol})
        bar_obj_dict = dict()
        for bar in bar_dict:
            bar_obj = BarQuotationData()
            bar_obj.__dict__ = bar
            bar_obj.last = bar['close']
            bar_obj_dict[bar_obj.time] = bar_obj
        self.stock_minute_bar[symbol] = bar_obj_dict
        # print('股票分钟查询====》合约：%s,耗时：%s' % (symbol, str(time.time() - start)))
        if trade_time in self.stock_minute_bar[symbol].keys():
            return self.stock_minute_bar[symbol][trade_time]
        else:
            return BarQuotationData()

    def get_future_daily_bar(self, symbol, date):
        start = time.time()
        # print('期货日线查询1====》合约：%s 日期：%s' % (symbol, date))

        if symbol in self.future_daily_bar.keys():
            bar = self.future_daily_bar[symbol]
            # print(bar)
            bar.last = getattr(bar, self.last_field)
            return bar

        # collection = "daily_future_quotation"
        collection = "future_1d_market"

        # bar_dict = self.quotation_mongo_db.mongo_find_one(db_name="panda",collection_name=collection,query={"trade_date": int(date), "symbol": symbol})
        bar_dict = self.quotation_mongo_db.mongo_find_one(db_name="panda",collection_name=collection,query={"date": str(date), "symbol": symbol.split(".")[0]})
        bar = DailyQuotationData()
        if bar_dict:
            bar.__dict__ = bar_dict
            # print(bar_dict)
            bar.last = bar_dict[self.last_field]
            self.future_daily_bar[symbol] = bar
        else:
            self.fund_daily_bar[symbol] = bar
        # print('期货日线查询2====》合约：%s 日期：%s' % (symbol, date))

        return bar

    def get_future_minute_bar(self, symbol, trade_date, trade_time):
        # start = time.time()
        trade_time = int(trade_time)
        if symbol in self.future_all_minute_bar.keys():
            if trade_time in self.future_all_minute_bar[symbol].keys():
                bar = self.future_all_minute_bar[symbol][trade_time]
                bar.last = getattr(bar, self.last_field)
                return bar
            else:
                return BarQuotationData()

        # 获取一天的数据
        collection = self.quotation_mongo_db.future_quotation_min_data_v2
        bar_dict = collection.find({"trade_date": int(trade_date), "symbol": symbol}, {'_id': 0})
        bar_obj_dict = dict()
        settlement = self.get_future_daily_bar(symbol, trade_date).settlement
        for bar in bar_dict:
            bar_obj = BarQuotationData()
            bar_obj.__dict__ = bar
            bar_obj.last = bar['close']
            bar_obj.settlement = settlement
            if bar_obj.settlement == 0:
                bar_obj.settlement = bar_obj.close
            bar_obj_dict[bar_obj.time] = bar_obj

        self.future_all_minute_bar[symbol] = bar_obj_dict

        # print('期货分钟查询====》合约：%s,耗时：%s' % (symbol, str(time.time() - start)))
        if trade_time in self.future_all_minute_bar[symbol].keys():
            return self.future_all_minute_bar[symbol][trade_time]
        else:
            return BarQuotationData()

    def get_fund_daily_bar(self, symbol, trade_date):
        # start = time.time()
        if symbol in self.fund_daily_bar.keys():
            result = self.fund_daily_bar[symbol]
            return result
        else:
            strategy_context = self.context.strategy_context
            pre_trade_date = strategy_context.get_next_count_date(trade_date, -1)
            collection = self.quotation_mongo_db.fund_daily_quotation

            if pre_trade_date == '99990101':
                find_dict = {'publish_date': {'$lte': trade_date}, 'symbol': symbol}
            else:
                find_dict = {'publish_date': {'$lte': trade_date, '$gt': pre_trade_date}, 'symbol': symbol}

            bar_dict = collection.find_one(find_dict,
                                           {'_id': 0, 'insert_time': 0})
            bar = DailyQuotationData()
            # print('基金日线查询====》合约：%s,耗时：%s' % (symbol, str(time.time() - start)))
            if bar_dict:
                bar.symbol = bar_dict['symbol']
                bar.unit_nav = bar_dict['nav']
                bar.last = bar.unit_nav
            self.fund_daily_bar[bar.symbol] = bar
            return bar

    def init_stock_list_daily_quotation(self, symbol_list, trade_date, freq='1d'):
        if len(symbol_list) == 0:
            return

        stock_list = list()
        stock_fund_list = list()
        index_list = list()

        for symbol in symbol_list:
            stock_type = self.stock_info_map[symbol]['type']
            if stock_type == 1:
                index_list.append(symbol)
            elif stock_type == 2:
                stock_fund_list.append(symbol)
            else:
                stock_list.append(symbol)

        if freq == '1d':
            if len(stock_list) > 0:
                # start = time.time()
                # collection = "stock_daily_quotation"
                collection = "stock_market"
                self.init_stock_list_daily_quotation_by_collection(stock_list, trade_date, freq, collection)
                # print('股票初始化查询耗时：' + str(time.time() - start))

            if len(stock_fund_list) > 0:
                # start = time.time()
                collection = "etf_daily_quotation_v2"
                self.init_stock_list_daily_quotation_by_collection(stock_fund_list, trade_date, freq, collection)
                # print('场内基金初始化耗时：' + str(time.time() - start))

            if len(index_list) > 0:
                # start = time.time()
                collection = "index_daily_quotation"
                self.init_stock_list_daily_quotation_by_collection(index_list, trade_date, freq, collection)
                # print('指数初始化耗时：' + str(time.time() - start))
        else:
            if freq == '1d':
                if len(stock_list) > 0:
                    collection = self.quotation_mongo_db.stock_quotation_min_data
                    self.init_stock_list_daily_quotation_by_collection(stock_list, trade_date, freq, collection)

                if len(stock_fund_list) > 0:
                    collection = self.quotation_mongo_db.stock_quotation_min_data
                    self.init_stock_list_daily_quotation_by_collection(stock_fund_list, trade_date, freq, collection)

                if len(index_list) > 0:
                    collection = self.quotation_mongo_db.stock_quotation_min_data
                    self.init_stock_list_daily_quotation_by_collection(index_list, trade_date, freq, collection)

    def init_stock_list_daily_quotation_by_collection(self, symbol_list, trade_date, freq='1d', collection=None):
        if freq == '1d':
            bar_cur = self.quotation_mongo_db.mongo_find(config["MONGO_DB"],collection_name=collection,query={"symbol": {'$in': symbol_list}, "trade_date": trade_date},projection={'_id': 0, 'insert_time': 0})
            # bar_cur = collection.find({"symbol": {'$in': symbol_list}, "trade_date": trade_date},
            #                           {'_id': 0, 'insert_time': 0})
            for bar_dict in bar_cur:
                bar = DailyQuotationData()
                if bar_dict:
                    bar.__dict__ = bar_dict
                    bar.last = bar_dict[self.last_field]
                    self.stock_daily_bar[bar_dict['symbol']] = bar
            # print('股票日线初始化查询====》合约：%s,耗时：%s' % (str(symbol_list), str(time.time() - start)))
        else:

            for symbol in symbol_list:
                self.stock_minute_bar[symbol] = dict()

            bar_cur = collection.find({"trade_date": int(trade_date), "symbol": {'$in': symbol_list}}, {'_id': 0})
            bar_cur = self.quotation_mongo_db.mongo_find(config["MONGO_DB"], collection_name=collection,
                                                         query={"symbol": {'$in': symbol_list},
                                                                "trade_date": trade_date},
                                                         projection={'_id': 0})
            for bar_dict in bar_cur:
                bar = BarQuotationData()
                if bar_dict:
                    bar.__dict__ = bar_dict
                    bar.last = bar_dict[self.last_field]
                    self.stock_minute_bar[bar_dict['symbol']][bar_dict['time']] = bar

    def init_future_list_daily_quotation(self, symbol_list, trade_date, freq='1d'):
        if len(symbol_list) == 0:
            return
        if freq == '1d':
            self.init_future_daily_quotation(symbol_list, trade_date)
        else:
            self.init_future_daily_quotation(symbol_list, trade_date)
            self.init_future_min_quotation(symbol_list, trade_date)

    def init_future_daily_quotation(self, symbol_list, trade_date):
        # collection = self.quotation_mongo_db.daily_future_quotation
        if symbol_list:
            processed_symbol_list = [symbol.split(".")[0] for symbol in symbol_list]
        bar_cur = self.quotation_mongo_db.mongo_find(db_name="panda",collection_name="future_1d_market",query={"date": str(trade_date), "symbol": {'$in': processed_symbol_list}})
        # print('期货初始化日线查询====》trade_date：%s ' % trade_date)
        for bar_dict in bar_cur:
            bar = DailyQuotationData()
            if bar_dict:
                # print('期货初始化日线查询====》symbol：%s' % (bar_dict['symbol']))
                bar.__dict__ = bar_dict
                # bar.last = bar_dict[self.last_field]
                bar_dict.get(self.last_field, 1)
                self.future_daily_bar[bar_dict['symbol']] = bar
        # print('期货初始化日线查询====》合约：%s,耗时：%s' % (str(symbol_list), str(time.time() - start)))

    def init_future_min_quotation(self, symbol_list, trade_date):
        start = time.time()
        for symbol in symbol_list:
            self.future_all_minute_bar[symbol] = dict()
        collection = self.quotation_mongo_db.future_quotation_min_data_v2
        bar_cur = collection.find({"trade_date": int(trade_date), "symbol": {'$in': symbol_list}}, {'_id': 0}).sort(
            [('symbol', pymongo.ASCENDING)])
        cur_symbol = None
        cur_settle = None
        for bar_dict in bar_cur:
            bar = BarQuotationData()
            if bar_dict:
                bar.__dict__ = bar_dict
                bar.last = bar_dict[self.last_field]
                if bar.symbol != cur_symbol:
                    cur_settle = self.get_future_daily_bar(bar_dict['symbol'], trade_date).settlement
                    cur_symbol = bar.symbol
                settlement = cur_settle
                bar.settlement = settlement
                if bar.settlement == 0:
                    bar.settlement = bar.close
                self.future_all_minute_bar[bar_dict['symbol']][bar_dict['time']] = bar
        # print('期货分钟初始化查询====》合约：%s,耗时：%s' % (str(symbol_list), str(time.time() - start)))

    def init_fund_list_daily_quotation(self, symbol_list, trade_date):
        if len(symbol_list) == 0:
            return
        start = time.time()
        for symbol in symbol_list:
            self.fund_daily_bar[symbol] = DailyQuotationData()

        strategy_context = self.context.strategy_context
        pre_trade_date = strategy_context.get_next_count_date(trade_date, -1)
        collection = self.quotation_mongo_db.fund_daily_quotation

        if pre_trade_date == '99990101':
            find_dict = {'symbol': {'$in': symbol_list},
                         'publish_date': {'$lte': trade_date}}
        else:
            find_dict = {'symbol': {'$in': symbol_list},
                         'publish_date': {'$lte': trade_date, '$gt': pre_trade_date},
                         }

        bar_cur = collection.find(find_dict).sort(
            [('end_date', pymongo.ASCENDING)])

        for bar_dict in bar_cur:
            bar = DailyQuotationData()
            if bar_dict:
                bar.symbol = bar_dict['symbol']
                bar.unit_nav = bar_dict['nav']
                bar.last = bar.unit_nav
            self.fund_daily_bar[bar.symbol] = bar
        # print('基金初始化查询====》耗时：%s' % str(time.time() - start))

    def clear_cache_data(self):
        self.stock_daily_bar = dict()
        self.stock_minute_bar = dict()
        self.future_daily_bar = dict()
        self.future_all_minute_bar = dict()
        self.fund_daily_bar = dict()
