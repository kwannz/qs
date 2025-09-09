import time

from panda_backtest.api.api import insert_future_group_order
from panda_backtest.system.panda_log import SRLogger


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def initialize(context):
    context.index = 0


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    SRLogger.info("before_trading")


# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_data(context, data):
    long_position_dict = {'AG2110.SHF': 15}
    short_position_dict = {}
    insert_future_group_order('108404', long_position_dict, short_position_dict)
    return
    # stock_account = context.run_info.stock_account
    # account = context.run_info.future_account
    # long_position_dict = {'AG2106.SHF': 10, 'B2105.DCE': 1, 'AL2105.SHF': 9}
    # short_position_dict = {'AG2106.SHF': 15, 'MA2105.CZC': 30}
    # insert_future_group_order(account, long_position_dict, short_position_dict)
    # symbol_dict = {'000001.SZ': 5000, '600775.SH': 9900, '000002.SZ': 15600,
    #                '000004.SZ': 3800, '600000.SH': 3300, '600006.SH': 15200, '688002.SH': 23000,
    #                '688006.SH': 9000}
    # insert_stock_group_order(stock_account, symbol_dict)
    # order_shares(stock_account, '000001.SZ', 100)
    # order_list = buy_open(account, 'AG2108.SHF', 10)
    # while True:
    #     for order in order_list:
    #         print(order.status)
            # time.sleep(0.05)
    # sell_open(account, 'ZC2107.CZC', 10, style=LimitOrderStyle(800))
    # buy_open(account, 'AG2106.SHF', 10, style=LimitOrderStyle(2023))


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    print("after_trading")