from panda_backtest.api.api import *
import logging

import json
import pandas as pd

# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def initialize(context):
    SRLogger.info("策略初始化开始")
     # 下单时间
    context.order_date = '20220217'
    context.order_time = '0923'
    # 下单dataFrame
    context.close_data_list = ['000001.SZ']

# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    SRLogger.info("交易前1")

# 你选择的标的的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_data(context, bar_dict):
    # 下单逻辑
    stock_account = context.run_info.stock_account
    if context.trade_date == '20150408':
        order_shares(stock_account, '510500.SH', 1000)
        # order_shares(stock_account, '000001.SZ', 1000)

# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    SRLogger.info("交易后")
