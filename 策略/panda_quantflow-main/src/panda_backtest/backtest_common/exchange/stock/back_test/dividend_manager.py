import time
import logging

from panda_backtest.backtest_common.model.quotation.dividend import Dividend

from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.backtest_common.system.event.event import Event, ConstantEvent
from common.config.config import config

class DividendManager(object):
    def __init__(self, quotation_mongo_db):
        self.quotation_mongo_db = quotation_mongo_db
        self.context = CoreContext.get_instance()

    def start_dividend(self):
        """
        股票分红
        每个交易日开始前获取所有持仓（包括基准）当天分红数据，所有数据，则推送分红事件
        :return:
        """
        event_bus = self.context.event_bus
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        all_pos_list = list()
        for stock_account in strategy_context.stock_account_dict.values():
            all_pos_list.extend(stock_account.positions.keys())

        if run_info.standard_type == 0:
            stand_symbol = strategy_context.run_info.benchmark
            all_pos_list.append(stand_symbol)
        all_pos_set = set(all_pos_list)

        if len(all_pos_set) == 0:
            return

        # start = time.time()
        collection = "stock_dividends"
        dividend_cur = self.quotation_mongo_db.mongo_find(config["MONGO_DB"], collection_name=collection,
                                                          query={'symbol': {'$in': list(all_pos_set)},
                                                                 'ex_div_date': strategy_context.trade_date},
                                                          projection={'_id': 0, 'symbol': 1, 'share_trans_ratio': 1,
                                                                      'share_ratio': 1, 'unit_cash_div_tax': 1,
                                                                      'ex_div_date': 1})
        # dividend_cur = collection.find(
        #     {'symbol': {'$in': list(all_pos_set)}, 'ex_div_date': strategy_context.trade_date},
        #     {'_id': 0, 'symbol': 1, 'share_trans_ratio': 1, 'share_ratio': 1, 'unit_cash_div_tax': 1, 'ex_div_date': 1})
        for dividend_dict in dividend_cur:
            dividend = Dividend()
            dividend.__dict__ = dividend_dict
            if dividend.share_trans_ratio is None:
                dividend.share_trans_ratio = 0

            if dividend.share_ratio is None:
                dividend.share_ratio = 0

            if dividend.cash_div_tax is None:
                dividend.cash_div_tax = 0

            # 推送分红事件
            event = Event(ConstantEvent.SYSTEM_STOCK_DIVIDEND, dividend=dividend)
            event_bus.publish_event(event)
        # print('分红耗时：' + str(time.time() - start))
