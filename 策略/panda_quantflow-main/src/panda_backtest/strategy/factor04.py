import time
import numpy as np
from panda_backtest.api.api import *
from panda_backtest.api.stock_api import *
import panda_data
import traceback


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递对象将会在你的算法策略的任何方法之间做传递。
def initialize(context):
    start_time = time.time()  # 记录开始时间
    panda_data.init()
    SRLogger.info("策略初始化开始")
    SRLogger.info("adjustment_cycle:" + str(context.adjustment_cycle))
    SRLogger.info("\ngroup_number:" + str(context.group_number))

    # 自定义累计交易日，用于换仓判断
    context.trade_days = 0

    context.df_factor = context.df_factor.shift(1).reset_index()

    end_time = time.time()  # 记录结束时间
    SRLogger.info(f"initialize 执行时间: {end_time - start_time:.4f}秒")


# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    start_time = time.time()  # 记录开始时间
    # 根据调仓周期判断当天是否交易
    if context.trade_days == 0:
        context.flag = True
        # 获取目标持仓
        context.trade_df = get_target_positions(context)
    # 未到调仓周期
    elif context.trade_days % context.adjustment_cycle != 0:
        context.flag = False
    # 调仓
    elif context.trade_days % context.adjustment_cycle == 0:
        context.flag = True
        # 获取目标持仓
        df_target = get_target_positions(context)
        # 获取当前持仓
        df_current = context.stock_account_dict['8888'].positions.to_dataframe()
        context.trade_df = calculate_rebalance_table(df_target, df_current)

    end_time = time.time()  # 记录结束时间
    SRLogger.info(f"before_trading 执行时间: {end_time - start_time:.4f}秒")
    SRLogger.info(f"交易前{context.trade_days}")


# 你选择的标的的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_data(context, bar_dict):
    start_time = time.time()  # 记录开始时间
    # 下单逻辑
    stock_account = context.run_info.stock_account
    for index, row in context.trade_df.iterrows():
        symbol = row["symbol"]
        positions = int(row["positions"])
        try:
            start_time_1 = time.time()
            order_shares(stock_account, symbol, positions)
            end_time_1 = time.time()
            print(f"order_shares 执行时间: {start_time_1 - end_time_1:.4f}秒")
        except Exception as e:
            traceback.print_exc()
            print(f"下单失败: day={context.now}, symbol={symbol}, positions={positions}, 错误信息: {e}")

    end_time = time.time()  # 记录结束时间
    SRLogger.info(f"handle_data 执行时间: {end_time - start_time:.4f}秒")


# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    start_time = time.time()  # 记录开始时间
    SRLogger.info("交易后")
    # 交易日加1
    context.trade_days += 1
    print(f"交易后: day={context.now}, total_value ={context.stock_account_dict['8888'].total_value}")

    end_time = time.time()  # 记录结束时间
    SRLogger.info(f"after_trading 执行时间: {end_time - start_time:.4f}秒")


def count_symbol_positions(context, df_symbol: pd.DataFrame):
    start_time = time.time()  # 记录开始时间
    total_value = context.stock_account_dict['8888'].total_value
    # 通过股票列表，获取pre_close
    df_symbol = stock_api_pre_close(df_symbol, context.now)
    avg_value_per_stock = total_value / 100 / len(df_symbol)
    df_symbol["positions"] = np.floor(avg_value_per_stock / (df_symbol["pre_close"] * 1.1)).astype(int)
    df_symbol["positions"] = df_symbol["positions"] * 100
    end_time = time.time()  # 记录结束时间
    SRLogger.info(f"count_symbol_positions 执行时间: {end_time - start_time:.4f}秒")
    return df_symbol


# 获取当天买入的股票
def get_target_positions(context):
    start_time = time.time()  # 记录开始时间
    today = context.now  # 获取回测当天日期
    # 获取第三列的列名
    third_column = context.df_factor.columns[-1]
    today_df_factor = context.df_factor[context.df_factor["date"] == today].sort_values(third_column, ascending=False)
    # 获取前几行的 "symbol" 列，并转换为列表
    # today_trade_symbol = pd.DataFrame(
    #     today_df_factor.head(len(today_df_factor) // int(context.group_number))["symbol"].tolist(), columns=["symbol"])
    today_trade_symbol = pd.DataFrame(
        today_df_factor.head(50)["symbol"].tolist(), columns=["symbol"])
    trade_df = count_symbol_positions(context, today_trade_symbol)

    end_time = time.time()  # 记录结束时间
    SRLogger.info(f"get_target_positions 执行时间: {end_time - start_time:.4f}秒")
    return trade_df


def calculate_rebalance_table(df_target: pd.DataFrame, df_current: pd.DataFrame) -> pd.DataFrame:
    start_time = time.time()  # 记录开始时间
    # 只保留 symbol 和 positions 两列
    df_target = df_target[["symbol", "positions"]].rename(columns={"positions": "target"})
    df_current = df_current[["symbol", "positions"]].rename(columns={"positions": "current"})

    # 合并两个表，按 symbol 对齐
    df_merged = pd.merge(df_target, df_current, on="symbol", how="outer").fillna(0)

    # 计算调仓差值（目标 - 当前）
    df_merged["positions"] = (df_merged["target"] - df_merged["current"]).astype(int)

    # 只保留有换仓动作的项（非零）
    df_rebalance = df_merged[df_merged["positions"] != 0][["symbol", "positions"]]

    end_time = time.time()  # 记录结束时间
    SRLogger.info(f"calculate_rebalance_table 执行时间: {end_time - start_time:.4f}秒")

    return df_rebalance.reset_index(drop=True)
