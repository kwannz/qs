from panda_backtest.api.api import *
from panda_backtest.api.stock_api import *
import pandas as pd
import numpy as np
import copy
import datetime
import re
import pickle
import sys

def initialize(context):
    # 策略参数设置，可以根据实际情况进行修改
    context.s_top_n = 10               # 每次买入的前N只标的
    context.s_rb_period = 5            # 调仓周期（单位：天）

    # 预处理因子数据
    context.df_factor = context.df_factor.reset_index()
    context.df_factor['factor_value'] = pd.to_numeric(
        context.df_factor
        .groupby('symbol')[context.df_factor.columns[2]]
        .shift(1),
        errors='coerce'
    )

    print("策略初始化完成123")
    context.account = '8888'
    print(f"因子总行数: {len(context.df_factor)}")
    print(f"因子列名: {list(context.df_factor.columns)}")

def handle_data(context, bar_dict):
    if int(context.now) % context.s_rb_period != 0:
        return  # 非调仓日不执行任何操作

    print(f"调仓日：{context.now}")
    today = context.now

    # 获取今日因子值并按值排序
    df_today = context.df_factor[context.df_factor["date"] == today]
    df_today_sorted = df_today.sort_values('factor_value', ascending=False)
    buy_list = df_today_sorted.head(context.s_top_n)['symbol'].tolist()
    SRLogger.info(buy_list)
    # 获取行情数据
    quotation_df = stock_api_quotation(symbol_list=buy_list, start_date=today, end_date=today, period="1d")

    per_close = quotation_df.set_index('symbol')['close'].to_dict()
    symbols = list(per_close.keys())

    # 获取总资产
    total_value = context.stock_account_dict[context.account].total_value
    slinge_symbol_value=total_value/context.s_top_n
    # 构建下单指令
    orders = {}
    for symbol in symbols:
        if symbol not in per_close:
            print(f"缺失数据: {symbol}")
            continue
            # 计算目标购买的股数
        hands = slinge_symbol_value /  per_close[symbol]
        # 判断股票是否是创业板（通过股票代码的前缀判断）
        if symbol.startswith('300'):  # 如果是创业板
            contract_mul = 200  # 创业板合约乘数为200
            # 计算手数，创业板股票可以购买任意数目，但至少为200手
            hands = np.floor(np.abs(hands))  # 向下取整
            if hands < 200:
                hands = 200  # 确保至少购买200手
        else:  # 非创业板
            contract_mul = 100  # 非创业板合约乘数为100
            # 非创业板股票必须按100的倍数购买
            hands = np.floor(np.abs(hands) / 100) * 100  # 向下取整至100的倍数

        orders[symbol] = hands

        print(f"{symbol}: 合约乘数={contract_mul}, 收盘价={per_close[symbol]}, 下单手数={hands}")

    # SRLogger.info(pd.DataFrame(list(orders.items()), columns=['symbol', 'order_hands']))
    target_stock_group_order(context.account, orders, 0)
