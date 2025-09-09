import pickle
import logging

from collections import defaultdict

from panda_backtest.backtest_common.model.result.order import Order, EMPTY_STRING, FILLED, CANCELLED, PartTradedNotQueueing, REJECTED, \
    NoTradeNotQueueing

class StockWorkOrderList(object):
    def __init__(self, collection, context):
        self.context = context
        self.collection = collection
        self.run_id = self.context.strategy_context.run_info.run_id
        self.order_id_dict = dict()
        self.order_sys_id_index_dict = dict()
        self.order_symbol_dict = defaultdict(list)

    def save_work_order(self, account, order, date):
        order.order_type = 0
        order.trade_date = self.context.strategy_context.trade_date
        order.run_id = self.run_id
        key = {'order_id': order.order_id, 'account': account, 'run_id': self.run_id, 'date': date, 'order_type': 0}
        self.collection.update(key, order.__dict__, upsert=True)
        if order.order_id is not EMPTY_STRING:
            self.order_id_dict[order.order_id] = order
            self.order_symbol_dict[order.order_book_id].append(order.order_id)
            if order.order_sys_id is not EMPTY_STRING:
                self.order_sys_id_index_dict[order.order_sys_id] = order.order_id

    def get_work_order_by_order_id(self, account, order_id, date):
        if order_id in self.order_id_dict.keys():
            return self.order_id_dict[order_id]

        order_dict = self.collection.find_one(
            {'order_id': order_id, 'account': account, 'run_id': self.run_id, 'date': date, 'order_type': 0},
            {'_id': 0})
        if order_dict:
            order = Order()
            order.__dict__ = order_dict
            return order
        else:
            return None

    def get_work_order_by_sys_id(self, account, order_sys_id, date):
        if order_sys_id in self.order_sys_id_index_dict.keys():
            order_id = self.order_sys_id_index_dict[order_sys_id]
            if order_id in self.order_id_dict.keys():
                return self.order_id_dict[order_id]

        order_dict = self.collection.find_one(
            {'order_sys_id': order_sys_id, 'account': account, 'run_id': self.run_id, 'date': date, 'order_type': 0},
            {'_id': 0})
        if order_dict:
            order = Order()
            order.__dict__ = order_dict
            return order
        else:
            return None

    def get_wait_work_oder(self):
        all_order_list = list()
        for order in list(self.order_id_dict.values()):
            if order.status != FILLED and order.status != CANCELLED and order.status != PartTradedNotQueueing\
                    and order.status != REJECTED and order.status != NoTradeNotQueueing:
                all_order_list.append(order)
        return all_order_list

    def clear_order(self):
        self.order_id_dict = dict()
        self.order_sys_id_index_dict = dict()
        self.order_symbol_dict = defaultdict(list)
