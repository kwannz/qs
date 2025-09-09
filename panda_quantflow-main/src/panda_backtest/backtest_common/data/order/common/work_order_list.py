import pickle
import logging

from panda_backtest.backtest_common.constant.strategy_constant import SIDE_SELL

class WorkOrderList(object):
    def __init__(self):
        self.order_dict = dict()

        # 合约索引
        self.symbol_index = dict()

        # 基金撮合日期索引
        self.cross_date_index = dict()

        # 基金到账日期索引
        self.fund_arrive_date_order = dict()

    def add_order(self, order, is_fund_order=False):
        self.order_dict[order.order_id] = order
        if order.order_book_id in self.symbol_index.keys():
            symbol_order_list = self.symbol_index[order.order_book_id]
        else:
            symbol_order_list = list()
            self.symbol_index[order.order_book_id] = symbol_order_list
        symbol_order_list.append(order.order_id)

        if is_fund_order:
            if order.fund_cross_date in self.cross_date_index.keys():
                cross_date_order_list = self.cross_date_index[order.fund_cross_date]
            else:
                cross_date_order_list = list()
                self.cross_date_index[order.fund_cross_date] = cross_date_order_list
            cross_date_order_list.append(order.order_id)

    def remove_fund_order_arrive_date(self, fund_arrive_date):
        del self.fund_arrive_date_order[fund_arrive_date]

    def add_fund_order_arrive_date(self, order):
        # 基金赎回到账日期记录
        if order.fund_arrive_date in self.fund_arrive_date_order.keys():
            fund_arrive_date_order_list = self.fund_arrive_date_order[order.fund_arrive_date]
        else:
            fund_arrive_date_order_list = list()
            self.fund_arrive_date_order[order.fund_arrive_date] = fund_arrive_date_order_list
        fund_arrive_date_order_list.append(order)

    def remove_order(self, order_id, is_fund_order=False):
        if order_id in self.order_dict.keys():
            order = self.order_dict[order_id]
            del self.order_dict[order_id]
            symbol_order_list = self.symbol_index[order.order_book_id]
            symbol_order_list.remove(order_id)
            if len(symbol_order_list) == 0:
                del self.symbol_index[order.order_book_id]

            if is_fund_order:
                cross_date_order_list = self.cross_date_index[order.fund_cross_date]
                cross_date_order_list.remove(order_id)

                if len(cross_date_order_list) == 0:
                    del self.cross_date_index[order.fund_cross_date]

    def get_fund_cross_date_symbol_list(self, cross_date):
        all_symbol_list = set()
        for cross_date_key in self.cross_date_index.keys():
            if cross_date >= cross_date_key:
                ids_list = self.cross_date_index[cross_date_key]
                for order_id in ids_list:
                    all_symbol_list.add(self.order_dict[order_id].order_book_id)

        return all_symbol_list

    def get_fund_arrive_order_list(self, arrive_date):
        if arrive_date not in self.fund_arrive_date_order.keys():
            return list()
        arrive_date_order_list = self.fund_arrive_date_order[arrive_date]
        return arrive_date_order_list

    def get_order_list(self, order_id=None, symbol=None, cross_date=None):
        if len(self.order_dict) == 0:
            return list()
        if order_id is not None:
            if order_id in self.order_dict.keys():
                order = self.order_dict[order_id]
                return [order]
            else:
                return list()

        if symbol is not None and cross_date is None:
            if symbol not in self.symbol_index.keys():
                return list()
            symbol_order_list = self.symbol_index[symbol]
            all_list = list()
            for order_id in symbol_order_list:
                all_list.append(self.order_dict[order_id])

            return all_list

        if symbol is not None and cross_date is not None:
            all_list = list()
            if symbol in self.symbol_index.keys():
                symbol_order_list = self.symbol_index[symbol]
                for order_id in symbol_order_list:
                    order = self.order_dict[order_id]
                    if order.fund_cross_date <= cross_date:
                        all_list.append(self.order_dict[order_id])
                return all_list
            else:
                return list()

        return list(self.order_dict.values())

    def clear(self):
        self.symbol_index = dict()
        self.order_dict = dict()
        self.cross_date_index = dict()
        self.fund_arrive_date_order = dict()

    def restore_save(self, redis_client, key, hkey):
        save_run_data_dict = dict()
        save_list = ['order_dict', 'symbol_index', 'cross_date_index', 'fund_arrive_date_order']
        for name, item in self.__dict__.items():
            if name in save_list:
                save_run_data_dict[name] = item
        redis_client.setHashRedis(key, hkey,
                                  pickle.dumps(save_run_data_dict))

    def restore_read(self, redis_client, key, hkey):
        var_run_data = redis_client.getHashRedis(key, hkey)
        if var_run_data:
            var_run_data = pickle.loads(var_run_data)
            for name, value in var_run_data.items():
                self.__setattr__(name, value)
