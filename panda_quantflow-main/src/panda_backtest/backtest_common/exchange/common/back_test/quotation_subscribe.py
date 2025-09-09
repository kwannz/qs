import time
import logging

from panda_backtest.backtest_common.system.event.event import ConstantEvent, Event

from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData

from panda_backtest.backtest_common.system.context.core_context import CoreContext

class QuotationSubscribe(object):
    def __init__(self):
        self.context = CoreContext.get_instance()

    def start_fund_quotation_play(self):
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        sub_fund_symbol_list = strategy_context.sub_position_fund_symbol_list | strategy_context.sub_order_fund_symbol_list
        if run_info.standard_type == 2:
            sub_fund_symbol_list.add(run_info.benchmark)
        event_bus = self.context.event_bus
        bar_dict = QuotationData.get_instance().bar_dict
        bar_data_source = bar_dict.bar_data_source
        sub_fund_symbol_list = list(set(sub_fund_symbol_list))
        if len(sub_fund_symbol_list) > 0:
            for sub_fund_symbol in sub_fund_symbol_list:
                bar_data = bar_data_source.get_fund_daily_bar(sub_fund_symbol, strategy_context.trade_date)

                if bar_data.symbol == '' or bar_data.last is None:
                    continue

                event = Event(ConstantEvent.SYSTEM_FUND_QUOTATION_CHANGE, bar_data=bar_data)
                event_bus.publish_event(event)

                # 尝试撮合未成交的订单
                event = Event(ConstantEvent.SYSTEM_FUND_ORDER_CROSS, bar_data=bar_data)
                event_bus.publish_event(event)

    def start_quotation_play(self, time_type=0):
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        event_bus = self.context.event_bus
        bar_dict = QuotationData.get_instance().bar_dict
        bar_data_source = bar_dict.bar_data_source

        sub_stock_symbol_list = strategy_context.sub_position_stock_symbol_list | strategy_context.sub_order_stock_symbol_list
        sub_future_symbol_list = strategy_context.sub_position_future_symbol_list | strategy_context.sub_order_future_symbol_list
        if run_info.standard_type == 0:
            sub_stock_symbol_list.add(run_info.benchmark)
        elif run_info.standard_type == 1:
            sub_future_symbol_list.add(run_info.benchmark)

        if strategy_context.is_stock_trade():
            sub_stock_symbol_list = list(set(sub_stock_symbol_list))
            if len(sub_stock_symbol_list) > 0:
                for sub_stock_symbol in sub_stock_symbol_list:
                    if run_info.frequency == '1d':
                        bar_data = bar_data_source.get_stock_daily_bar(sub_stock_symbol, strategy_context.trade_date)
                    else:
                        bar_data = bar_data_source.get_stock_minute_bar(sub_stock_symbol, strategy_context.trade_date,
                                                                        strategy_context.hms)

                    if bar_data is None or not bar_data.symbol:
                        continue

                    if time_type == 0:
                        # 尝试撮合未成交的订单
                        # if run_info.matching_type == 1:
                        #     bar_data.last = bar_data.open

                        event = Event(ConstantEvent.SYSTEM_STOCK_QUOTATION_CHANGE, bar_data=bar_data)
                        event_bus.publish_event(event)

                        event = Event(ConstantEvent.SYSTEM_STOCK_ORDER_CROSS, bar_data=bar_data)
                        event_bus.publish_event(event)
                    else:
                        # bar_data.last = bar_data.close
                        event = Event(ConstantEvent.SYSTEM_STOCK_QUOTATION_CHANGE, bar_data=bar_data)
                        event_bus.publish_event(event)

        if strategy_context.is_future_trade():
            sub_future_symbol_list = list(set(sub_future_symbol_list))
            if len(sub_future_symbol_list) > 0:
                for sub_future_symbol in sub_future_symbol_list:
                    if run_info.frequency == '1d':
                        bar_data = bar_data_source.get_future_daily_bar(sub_future_symbol, strategy_context.trade_date)
                    else:
                        bar_data = bar_data_source.get_future_minute_bar(sub_future_symbol, strategy_context.trade_date,
                                                                         strategy_context.hms)
                    if bar_data.symbol == '':
                        continue
                    if time_type == 0:
                        # 尝试撮合未成交的订单
                        # if run_info.matching_type == 1:
                        #     bar_data.last = bar_data.open

                        # 尝试撮合未成交的订单
                        event = Event(ConstantEvent.SYSTEM_FUTURE_QUOTATION_CHANGE, bar_data=bar_data)
                        event_bus.publish_event(event)
                        event = Event(ConstantEvent.SYSTEM_FUTURE_ORDER_CROSS, bar_data=bar_data)
                        event_bus.publish_event(event)
                    else:
                        # bar_data.last = bar_data.close
                        event = Event(ConstantEvent.SYSTEM_FUTURE_QUOTATION_CHANGE, bar_data=bar_data)
                        event_bus.publish_event(event)

    def init_cache_data(self):
        bar_data_source = QuotationData.get_instance().bar_dict.bar_data_source
        bar_data_source.clear_cache_data()

    def init_daily_data(self):
        bar_data_source = QuotationData.get_instance().bar_dict.bar_data_source
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info

        strategy_context.un_sub_stock_symbol(None, 1)
        strategy_context.un_sub_future_symbol(None, 1)

        sub_stock_symbol_list = set()
        for stock_account in strategy_context.stock_account_dict.values():
            sub_stock_symbol_list.update(set(stock_account.positions.keys()))

        if run_info.standard_type == 0:
            sub_stock_symbol_list.add(run_info.benchmark)

        strategy_context.sub_position_stock_symbol_list = sub_stock_symbol_list

        if len(sub_stock_symbol_list) > 0:
            bar_data_source.init_stock_list_daily_quotation(list(sub_stock_symbol_list), strategy_context.trade_date,
                                                            run_info.frequency)

        sub_future_symbol_list = set()
        for future_account in strategy_context.future_account_dict.values():
            sub_future_symbol_list.update(set(future_account.positions.keys()))

        strategy_context.sub_position_future_symbol_list = sub_future_symbol_list

        if run_info.standard_type == 1:
            sub_future_symbol_list.add(run_info.benchmark)

        if len(sub_future_symbol_list) > 0:
            bar_data_source.init_future_list_daily_quotation(list(sub_future_symbol_list), strategy_context.trade_date,
                                                             run_info.frequency)

        sub_fund_symbol_list = set()
        for fund_account in strategy_context.fund_account_dict.values():
            sub_fund_symbol_list.update(set(fund_account.positions.keys()))

        strategy_context.sub_position_fund_symbol_list = sub_fund_symbol_list
        all_sub_fund_symbol_list = strategy_context.sub_position_fund_symbol_list | strategy_context.sub_order_fund_symbol_list
        if run_info.standard_type == 2:
            all_sub_fund_symbol_list.add(run_info.benchmark)

        if len(all_sub_fund_symbol_list) > 0:
            bar_data_source.init_fund_list_daily_quotation(list(all_sub_fund_symbol_list), strategy_context.trade_date)

