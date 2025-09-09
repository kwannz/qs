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
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
api_list = []


def append_to_api_list(func):
    """
    装饰器：将 API 函数名添加到全局的 api_list 中
    """
    api_list.append(func.__name__)
    globals()[func.__name__] = func
    return func


# 初始化数据库连接，避免每次调用时都创建新的数据库连接
quotation_mongo_db = DatabaseHandler(config=config)

"""
1.查询品种某一天的主力合约
"""
@append_to_api_list
def future_api_domain_symbol(symbol: str, date: str):
    # 构造 Mongo 查询条件
    query = {
        "symbol": symbol,
        "date": date
    }
    bar_dict = quotation_mongo_db.mongo_find_one(
        db_name=config["MONGO_DB"],
        collection_name='future_market',
        query=query,
        projection={'_id': 0, 'symbol': 1, 'exchange': 1, 'trading_code': 1}
    )

    return bar_dict


"""
2.获取期货行情
"""
@append_to_api_list
def future_api_quotation(symbol_list=None, start_date=None, end_date=None, fields=None, period=None):
    if symbol_list:
        processed_symbol_list = [symbol.split(".")[0] for symbol in symbol_list]
    # 根据 period 确定集合名称
    collection_map = {
        "1m": "future_1m_market",
        "5m": "future_5m_market",
        "15m": "future_15m_market",
        "30m": "future_30m_market",
        "1h": "future_60m_market",
        "1d": "future_1d_market"
    }
    collection_name = collection_map.get(period)
    if not collection_name:
        raise ValueError("Invalid period value")

    # 设置要返回的字段
    fields_dict = {'_id': 0}
    if fields:
        fields_dict.update({'trade_date': 1})  # 确保返回交易日期字段
        for f in fields:
            fields_dict[f] = 1

    # 构造 Mongo 查询条件
    query = {
        "symbol": {"$in": processed_symbol_list},
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
    df_bar['symbol'] = df_bar['symbol'].str.cat(df_bar['exchange'], sep='.')
    return df_bar


"""
3.获取合约最小乘数
"""
@append_to_api_list
def future_api_symbol_contractmul(symbol_list=None):
    #对symbol进行处理
    if symbol_list:
        exchange_mapping = {
            'CFFEX': 'CFE',
            'CZCE': 'CZC',
            'DCE': 'DCE',
            'SHFE': 'SHF',
            'INE': 'INE',
            'GFEX': 'GFE',
            # 可以继续添加其他交易所的映射
        }
        processed_symbol_list = []
        for symbol in symbol_list:
            for old, new in exchange_mapping.items():
                symbol = symbol.replace(old, new)
            processed_symbol_list.append(symbol)
    # 构造 Mongo 查询条件
    query = {
        "symbol": {"$in": processed_symbol_list},
    }
    bar_dict = quotation_mongo_db.mongo_find(
        db_name=config["MONGO_DB"],
        collection_name='future_info',
        query=query,
        projection={'_id': 0,  "symbol": 1,'contractmul': 1}
    )
    # 处理查询结果为空的情况
    if not bar_dict:
        return pd.DataFrame()

    # 转换查询结果为 DataFrame
    df_bar = pd.DataFrame(bar_dict)

    replacement_map = {
        r'\.SHF$': '.SHFE',
        r'\.CFE$': '.CFFEX',
        r'\.CZC$': '.CZCE',
        r'\.GFE$': '.GFEX'
    }
    for pattern, repl in replacement_map.items():
        df_bar['symbol'] = df_bar['symbol'].str.replace(pattern, repl, regex=True)

    # 定义替换规则
    return df_bar

"""
4.获取品种合约集合
"""
@append_to_api_list
def future_api_symbol_contracts(underlying_symbol: str, exchange:str):
    # 构造 Mongo 查询条件
    query = {
        "underlying_symbol": underlying_symbol,
        "exchange": exchange
    }
    bar_dict = quotation_mongo_db.mongo_find(
        db_name=config["MONGO_DB"],
        collection_name='future_symbol',
        query=query,
        projection={'_id': 0, 'underlying_symbol':1,'order_book_id': 1, 'exchange': 1}
    )
    df_bar = pd.DataFrame(bar_dict)
    df_bar['symbol'] = df_bar[['order_book_id', 'exchange']].astype(str).agg('.'.join, axis=1)
    return df_bar[['symbol','exchange','underlying_symbol']]




if __name__ == '__main__':
    # 测试查询主力合约
    # result = future_api_domain_symbol(symbol="AG88", date="20250605")
    # print(result)
    # result =future_api_symbol_contractmul(symbol_list=["IC2501.CFFEX","AG2504.SHFE","RI1907.CZC","V2502.DCE","BC2511.INE","LC2505.GFEX"])
    # print(result)
    # 测试查询期货行情
    # result = future_api_quotation(symbol_list=["IC88","AG2508.SHFE"],start_date="20250420",end_date="20250430",period="1d")
    # print(result)
    result =future_api_symbol_contracts(underlying_symbol="A",exchange="DCE")
    print(result)
