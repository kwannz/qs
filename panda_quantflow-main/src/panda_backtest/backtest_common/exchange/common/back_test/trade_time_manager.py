import datetime
import logging

import time

import pymongo

from panda_backtest.backtest_common.exception.error_exception import ErrorException
from panda_backtest.backtest_common.exception.strategy_exception_builder import StrategyExceptionBuilder
from panda_backtest.util.time.time_util import TimeUtil
from common.config.config import config
from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.backtest_common.system.event.event import Event, ConstantEvent
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory


class TradeTimeManager(object):
    def __init__(self, quotation_mongo_db):
        self.quotation_mongo_db = quotation_mongo_db
        self.context = CoreContext.get_instance()
        self.now = None
        self.hms = None
        self.trade_time = None
        self.trade_date = None
        # 所有交易日，开始回测到结束回测期间的所有交易日
        self.all_date_list = list()
        # 所有自然日，多种情况
        # 7*24或者有期货且回测为分钟: 回测第一个自然日从开始回测的上一个最近交易日开始
        self.all_nature_date_list = list()

    def is_trade_date(self):
        return self.trade_date == self.now

    def get_next_count_date(self, date, count):
        index_num = self.all_date_list.index(date)
        if index_num + count >= len(self.all_date_list) or index_num + count < 0:
            return '99990101'
        return self.all_date_list[index_num + count]

    def get_next_count_nature_date(self, date, count):
        index_num = self.all_nature_date_list.index(date)
        if index_num + count >= len(self.all_nature_date_list) or index_num + count < 0:
            return '99990101'
        return self.all_nature_date_list[index_num + count]

    def get_date_distance(self, start_date, end_date):
        start_index = self.all_nature_date_list.index(start_date)
        end_index = self.all_nature_date_list.index(end_date)
        return end_index - start_index

    def start_trade_time_play(self):
        """
        回测日线、分钟线轮转
        :return:
        """
        strategy_context = self.context.strategy_context

        # collection = self.quotation_mongo_db

        run_info = strategy_context.run_info

        frequency = run_info.frequency
        date_type = run_info.date_type
        account_type = run_info.account_type
        start = run_info.start_date
        end = run_info.end_date
        event_bus = self.context.event_bus

        # 交易日期
        # trade_dates = collection.find({
        #     'nature_date': {'$gte': str(start), '$lte': str(end)}, 'is_trade': 1, 'exchange': 'SH'}).sort('nature_date')
        docs = self.quotation_mongo_db.mongo_find(config['MONGO_DB'], collection_name="trade_calendar",
                                                         query={
                                                             'nature_date': {'$gte': int(start), '$lte': int(end)},
                                                             'is_trade': 1, 'exchange': 'SH'},
                                                         sort="nature_date")
        trade_dates_list = list(docs)
        # trade_dates_list = [str(doc["nature_date"]) for doc in docs]
        if len(trade_dates_list) == 0:
            event = Event(ConstantEvent.SYSTEM_CALCULATE_RESULT)
            event_bus.publish_event(event)
            return

        for trade_cal in trade_dates_list:
            trade_date = str(trade_cal['nature_date'])
            self.all_date_list.append(trade_date)

        start_date = self.all_date_list[0]
        if frequency == '1M':
            if date_type == 1 or (account_type != 0 and account_type != 3 and account_type != 4):
                # start_date = collection.find({
                #     'nature_date': {'$lt': str(start)}, 'is_trade': 1, 'exchange': 'SH'}) \
                #     .sort('nature_date', pymongo.DESCENDING).limit(1)
                start_date = self.quotation_mongo_db.mongo_find_one(config['MONG_DB'], collection_name="trade_calendar",
                                                                    query={
                                                                        'nature_date': {'$lt': str(start)},
                                                                        'is_trade': 1, 'exchange': 'SH'},
                                                                    sort=[('nature_date', -1)])
                if start_date:
                    start_date_dict = start_date.next()
                    start_date = start_date_dict['nature_date']
        else:
            if date_type == 1:
                start_date = start

        self.all_nature_date_list = TimeUtil.get_begin_to_end_date_list(start_date, end)

        if date_type == 0:
            rang_date_list = self.all_date_list
        else:
            rang_date_list = self.all_nature_date_list

        self.now = self.all_nature_date_list[0]
        self.trade_date = self.all_date_list[0]
        self.hms = '000000'
        self.trade_time = datetime.datetime.strptime((self.now + ' 000000'), '%Y%m%d %H%M%S')
        start_time = time.time()
        if strategy_context.enable_risk_control:
            event = Event(
                ConstantEvent.RISK_CONTROL_INIT,
                context=strategy_context)
            event_bus.publish_event(event)
        # print('RISK_CONTROL_INIT耗时：' + str(time.time() - start_time))
        start_time = time.time()
        try:
            event = Event(
                ConstantEvent.STRATEGY_INIT,
                context=strategy_context)
            event_bus.publish_event(event)
        except Exception as e:
            raise ErrorException(StrategyExceptionBuilder.build_strategy_run_exception_msg(), '00001', None)
        # print('策略初始化耗时：' + str(time.time() - start_time))

        SRLogger = RemoteLogFactory.get_sr_logger()

        # 获取行情数据
        if frequency == '1d':
            self.trade_date = self.all_date_list[0]
            total = len(rang_date_list)
            i = 0
            for nature_date in rang_date_list:
                day_start_time = time.time()
                SRLogger.process(i, total)
                i = i + 1
                self.now = nature_date
                self.hms = '083000'
                self.trade_time = datetime.datetime.strptime((nature_date + ' 083000'), '%Y%m%d %H%M%S')

                if nature_date in self.all_date_list:
                    event = Event(ConstantEvent.SYSTEM_NEW_DATE)
                    event_bus.publish_event(event)

                # print('new_date耗时1113333：===》' + str(time.time() - day_start_time))

                event = Event(ConstantEvent.SYSTEM_DAY_START)
                event_bus.publish_event(event)

                # print('day_start耗时：===》' + str(time.time() - day_start_time))

                if run_info.matching_type == 1:
                    self.hms = '093000'
                    self.trade_time = datetime.datetime.strptime((nature_date + ' 093000'), '%Y%m%d %H%M%S')
                    event = Event(ConstantEvent.SYSTEM_HANDLE_BAR)
                    event_bus.publish_event(event)
                    self.hms = '150000'
                    self.trade_time = datetime.datetime.strptime((nature_date + ' 150000'), '%Y%m%d %H%M%S')
                else:
                    self.hms = '150000'
                    self.trade_time = datetime.datetime.strptime((nature_date + ' 150000'), '%Y%m%d %H%M%S')
                    event = Event(ConstantEvent.SYSTEM_HANDLE_BAR)
                    event_bus.publish_event(event)

                # print('SYSTEM_HANDLE_BAR耗时：===》' + str(time.time() - day_start_time))

                if nature_date in self.all_date_list:
                    event = Event(ConstantEvent.SYSTEM_END_DATE)
                    event_bus.publish_event(event)
                    self.trade_date = self.get_next_count_date(self.trade_date, 1)

                # print('每日行情耗时：===》' + str(time.time() - day_start_time))

        elif frequency == '1M':
            self.trade_date = self.all_date_list[0]
            self.now = self.all_nature_date_list[0]
            total = len(rang_date_list)
            i = 0
            for nature_date in rang_date_list:
                day_start_time = time.time()
                SRLogger.process(i, total)
                i = i + 1
                if date_type == 0:
                    if account_type != 0 and account_type != 3 and account_type != 4:
                        self.hms = '203000'
                        self.trade_time = datetime.datetime.strptime((nature_date + ' 203000'), '%Y%m%d %H%M%S')
                        event = Event(ConstantEvent.SYSTEM_NEW_DATE)
                        event_bus.publish_event(event)
                        self.run_trade('210100', '235900')
                        self.now = self.get_next_count_nature_date(self.now, 1)
                        self.run_trade('000000', '023000')
                        if self.now != self.trade_date:
                            self.now = self.trade_date
                        money_time = '090100'
                    else:
                        self.hms = '083000'
                        self.trade_time = datetime.datetime.strptime((nature_date + ' 083000'), '%Y%m%d %H%M%S')
                        event = Event(ConstantEvent.SYSTEM_NEW_DATE)
                        event_bus.publish_event(event)

                        money_time = '093100'

                    event = Event(ConstantEvent.SYSTEM_DAY_START)
                    event_bus.publish_event(event)

                    self.run_trade(money_time, '113000')
                    self.run_trade('130100', '150000')
                    event = Event(ConstantEvent.SYSTEM_END_DATE)
                    event_bus.publish_event(event)
                    self.trade_date = self.get_next_count_date(self.trade_date, 1)
                    if account_type == 0 or account_type == 3 or account_type == 4:
                        self.now = self.get_next_count_date(self.now, 1)
                    # print('每分钟行情耗时：===》' + str(time.time() - day_start_time))
                else:
                    self.now = nature_date

                    if nature_date == self.all_nature_date_list[0]:
                        self.hms = '153100'
                        self.trade_time = datetime.datetime.strptime((nature_date + ' 153100'), '%Y%m%d %H%M%S')
                        event = Event(ConstantEvent.SYSTEM_NEW_DATE)
                        event_bus.publish_event(event)
                        self.run_trade('153100', '235900')
                        continue
                    self.run_trade('000000', '082900')

                    event = Event(ConstantEvent.SYSTEM_DAY_START)
                    event_bus.publish_event(event)
                    self.run_trade('83000', '153000')

                    if nature_date in self.all_date_list:
                        event = Event(ConstantEvent.SYSTEM_END_DATE)
                        event_bus.publish_event(event)

                        self.trade_date = self.get_next_count_date(self.trade_date, 1)
                        self.trade_time = datetime.datetime.strptime((nature_date + ' 153100'), '%Y%m%d %H%M%S')
                        event = Event(ConstantEvent.SYSTEM_NEW_DATE)
                        event_bus.publish_event(event)

                    self.run_trade('153100', '235900')

        event = Event(ConstantEvent.SYSTEM_CALCULATE_RESULT)
        event_bus.publish_event(event)

    def run_trade(self, start_time, end_time):
        trade_time = start_time
        while start_time <= trade_time <= end_time:
            now_date_time = datetime.datetime.strptime(str(trade_time), '%H%M%S')
            self.trade_time = datetime.datetime.strptime((self.now + ' ' + trade_time), '%Y%m%d %H%M%S')
            self.hms = trade_time

            event_bus = self.context.event_bus
            event = Event(ConstantEvent.SYSTEM_HANDLE_BAR)
            event_bus.publish_event(event)

            now_date_time = now_date_time + datetime.timedelta(minutes=1)
            trade_time = now_date_time.strftime('%H%M%S')

    def is_stock_trade(self):
        if ('093000' <= self.hms <= '113000') or ('130000' <= self.hms <= '150000'):
            return True
        else:
            return False

    def is_future_trade(self):

        if '210000' <= self.hms <= '240000' or self.hms <= '023000' or '090000' <= self.hms <= '113000' or '130000' \
                <= self.hms <= '150000':
            return True
        else:
            return False
