from api.api import order_shares
from panda_backtest.api.api import *
import pandas as pd
import numpy as np
# import talib as ta
import copy
import datetime
import re
import pickle
import sys

# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def initialize(context):
    SRLogger.info("策略初始化")
    context.account = '5588'
    context.trading_code="AG2509.SHF"

# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    SRLogger.info("交易前")



# 你选择的标的的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_data(context, bar_dict):
    if (int(context.now) % 2) == 0:
        buy_open(account_id=context.account,id_or_ins=context.trading_code,amount=1)
    else:
        sell_open(account_id=context.account,id_or_ins=context.trading_code,amount=-1)

# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    SRLogger.info("交易后")

