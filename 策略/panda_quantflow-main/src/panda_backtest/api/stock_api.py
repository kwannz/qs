# -*- coding: utf-8 -*-
"""
File: stock_api.py
Author: peiqi
Date: 2025/5/29
Description: 
"""
import pandas as pd
from common.connector.mongodb_handler import DatabaseHandler
from common.config.config import config

api_list = []
quotation_mongo_db = DatabaseHandler(config=config)
def append_to_api_list(func):
    api_list.append(func.__name__)
    globals()[func.__name__] = func
    return func

"""
获取股票的pre_close
"""
@append_to_api_list
def stock_api_pre_close(df_stock_code: pd.DataFrame, date: str):
    # 初始化数据库连接
    stock_code_list=df_stock_code['symbol'].tolist()
    # 构造 Mongo 查询条件
    query = {
        "symbol": {"$in": stock_code_list},  # 注意这里
        "date": date
    }

    # 查询
    bar_dict = quotation_mongo_db.mongo_find(
        db_name=config["MONGO_DB"],
        # collection_name='stock_daily_quotation',
        collection_name='stock_market',
        query=query,
        projection={'_id': 0, 'symbol': 1, 'pre_close': 1}
    )
    df_bar = pd.DataFrame(bar_dict)
    return df_bar
"""
获取股票行情
"""
@append_to_api_list
def stock_api_quotation(symbol_list=None, start_date=None, end_date=None, fields=None, period=None):

    # 根据 period 确定集合名称
    collection_map = {
        "1d": "stock_market",
    }
    collection_name = collection_map.get(period)
    if not collection_name:
        raise ValueError("Invalid period value")

    # 设置要返回的字段
    fields_dict = {'_id': 0}
    if fields:
        fields_dict.update({'date': 1})  # 确保返回交易日期字段
        for f in fields:
            fields_dict[f] = 1

    # 构造 Mongo 查询条件
    query = {
        "symbol": {"$in": symbol_list},
        "date": {"$gte": start_date, "$lte": end_date}
    }

    # 查询
    bar_dict = quotation_mongo_db.mongo_find(
        db_name=config["MONGO_DB"],
        collection_name=collection_name,
        query=query,
        projection=fields_dict
    )

    # 处理查询结果为空的情况
    if not bar_dict:
        return pd.DataFrame()
        # 如果请求的字段中包含exchange，则对每个文档的exchange字段进行截取

    # 转换查询结果为 DataFrame
    df_bar = pd.DataFrame(bar_dict)

    return df_bar

if __name__ == '__main__':
    # print(">>> 测试 stock_api_pre_close")
    # test_df = pd.DataFrame({'symbol': ['000001.SZ', '600000.SH']})
    # test_date = '20250710'
    # df_pre_close = stock_api_pre_close(test_df, test_date)
    # print(df_pre_close)

    # 测试 stock_api_quotation
    print("\n>>> 测试 stock_api_quotation")
    df_quotation = stock_api_quotation(
        symbol_list=['000001.SZ', '600000.SH'],
        start_date='20250710',
        end_date='20250712',
        fields=['symbol','pre_close', 'close', 'volume'],
        period='1d'
    )
    print(df_quotation)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
