#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/8/5 下午5:23
# @Author : wlb
# @File   : ctp_mdu_quo.py
# @desc   :
import json
import os
import tempfile
import threading
import hashlib
import ctp
import ctp as mdapi
import time
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from common.config.config import config
from panda_backtest.backtest_common.data.future.future_info_map import FutureInfoMap
from panda_backtest.backtest_common.model.quotation.bar_quotation_data import BarQuotationData
from common.connector.mongodb_handler import DatabaseHandler as MongoClient
from common.connector.redis_client import RedisClient
from panda_trading.trading.util.symbol_util import SymbolUtil
from utils.data.data_util import DateUtil
from utils.thread.thread_util import ThreadUtil
from utils.time.time_util import TimeUtil


class MdSpi(ctp.CThostFtdcMdSpi):
    def __init__(self, front, broker_id, user_id, password):
        ctp.CThostFtdcMdSpi.__init__(self)

        self.front = front
        self.broker_id = broker_id
        self.user_id = user_id
        self.password = password

        self.request_id = 0
        self.connected = False
        self.loggedin = False
        self.subscribed = False
        self.data = None

        self.api = self.create()

        # 业务字段值
        self.future_code_list = list()
        self.__redis_client = RedisClient()
        self.__thread_pool_executor = ThreadPoolExecutor(max_workers=40)
        self.future_info_map = FutureInfoMap(MongoClient(config).get_mongo_db())
        self.now_trade_date_tuple = (datetime.now().strftime('%Y%m%d'),
                                     DateUtil.get_next_trade_date(datetime.now().strftime('%Y%m%d')))
        self.executor = ThreadPoolExecutor(max_workers=10)


    def create(self):
        dir = ''.join(('ctp', self.broker_id, self.user_id)).encode('UTF-8')
        dir = hashlib.md5(dir).hexdigest()
        dir = os.path.join(tempfile.gettempdir(), dir, 'Md') + os.sep
        if not os.path.isdir(dir): os.makedirs(dir)
        return ctp.CThostFtdcMdApi.CreateFtdcMdApi(dir)

    def run(self):
        self.api.RegisterSpi(self)
        self.api.RegisterFront(self.front)
        self.api.Init()
        self.api.Join()

    def login(self):
        field = ctp.CThostFtdcReqUserLoginField()
        field.BrokerID = self.broker_id
        field.UserID = self.user_id
        field.Password = self.password
        self.request_id += 1
        self.api.ReqUserLogin(field, self.request_id)

    def OnFrontConnected(self):
        print("OnFrontConnected")
        self.connected = True
        self.login()

    def OnRspUserLogin(self, pRspUserLogin:'CThostFtdcRspUserLoginField', pRspInfo:'CThostFtdcRspInfoField', nRequestID:'int', bIsLast:'bool'):
        print("OnRspUserLogin", pRspInfo.ErrorID, pRspInfo.ErrorMsg)
        if pRspInfo.ErrorID == 0:
            self.loggedin = True

    def OnRspError(self, pRspInfo:'CThostFtdcRspInfoField', nRequestID:'int', bIsLast:'bool'):
        print("OnRspError:", pRspInfo.ErrorID, pRspInfo.ErrorMsg)

    def OnRspSubMarketData(self, pSpecificInstrument: 'CThostFtdcSpecificInstrumentField', pRspInfo: 'CThostFtdcRspInfoField', nRequestID: 'int', bIsLast: 'bool'):
        print("OnRspSubMarketData:", pRspInfo.ErrorID, pRspInfo.ErrorMsg)
        if pRspInfo.ErrorID == 0:
            self.subscribed = True

    def OnRtnDepthMarketData(self, pDepthMarketData: 'CThostFtdcDepthMarketDataField'):
        """
        收到行情推送时的回调函数
        """
        print("【行情推送】已触发！")
        print(f"合约ID: {pDepthMarketData.InstrumentID}")
        print(f"最新价: {pDepthMarketData.LastPrice}")
        print(f"买一价: {pDepthMarketData.BidPrice1}, 卖一价: {pDepthMarketData.AskPrice1}")
        self.data = pDepthMarketData
        # self.save_data_task(pDepthMarketData)
        try:
            self.executor.submit(self.save_data_task, pDepthMarketData)
        except Exception as e:
            print("❌ 行情推送异常:", str(e))


    def save_data_task(self, tick_data):
        try:
            if tick_data is None:
                print('数据为空')
                return
            bar_quotation_data = self.depth_market_dat_to_symbol(tick_data)
            if bar_quotation_data is None:
                return
            key = bar_quotation_data.symbol
            print("********+"+key)
            self.__redis_client.setHashRedis('tushare_future_tick_quotation', key,
                                             json.dumps(bar_quotation_data.__dict__))
        except Exception as e:
            mes = traceback.format_exc()
            print(mes)

    def depth_market_dat_to_symbol(self, tick_data):
        try:
            print("depth_market_dat_to_symbol:" + tick_data.InstrumentID)
            bar_quotation_data = BarQuotationData()
            symbol_info = self.future_info_map.get_by_ctp_code(tick_data.InstrumentID)
            if symbol_info:
                bar_quotation_data.symbol = symbol_info['symbol']
            else:
                bar_quotation_data.symbol = tick_data.InstrumentID
            # bar_quotation_data.code = bar_data['code']
            bar_quotation_data.date = datetime.now().strftime('%Y%m%d')
            bar_quotation_data.time = tick_data.UpdateTime
            bar_quotation_data.trade_date = tick_data.TradingDay
            bar_quotation_data.open = tick_data.OpenPrice
            bar_quotation_data.high = tick_data.HighestPrice
            bar_quotation_data.low = tick_data.LowestPrice
            bar_quotation_data.close = tick_data.ClosePrice
            bar_quotation_data.volume = tick_data.Volume
            bar_quotation_data.oi = tick_data.OpenInterest
            bar_quotation_data.turnover = tick_data.Turnover
            # bar_quotation_data.vwap = bar_data['vwap']
            # bar_quotation_data.oi = bar_data['oi']
            bar_quotation_data.settle = tick_data.SettlementPrice
            bar_quotation_data.last = tick_data.LastPrice
            bar_quotation_data.preclose = tick_data.PreClosePrice
            bar_quotation_data.limit_up = tick_data.UpperLimitPrice
            bar_quotation_data.limit_down = tick_data.LowerLimitPrice
            bar_quotation_data.askprice1 = tick_data.AskPrice1
            bar_quotation_data.bidprice1 = tick_data.BidPrice1
            bar_quotation_data.askvolume1 = tick_data.AskVolume1
            bar_quotation_data.bidvolume1 = tick_data.BidVolume1
            print(f"bar_quotation_data.symbol {bar_quotation_data.symbol}")
            parts = bar_quotation_data.symbol.split('.')
            if len(parts) >= 2:
                exchange = parts[1]
            else:
                # 如果没有找到 '.'，可以设置默认值或处理异常情况
                exchange = None  # 或者 raise ValueError 或者使用其他默认逻辑
                print(f"Warning: Invalid symbol format - {bar_quotation_data.symbol}")
            if exchange == 'CZC':
                if TimeUtil.in_time_range('210000-235959'):
                    if datetime.now().strftime('%Y%m%d') == self.now_trade_date_tuple[0]:
                        bar_quotation_data.trade_date = self.now_trade_date_tuple[1]
                    else:
                        self.now_trade_date_tuple = (datetime.now().strftime('%Y%m%d'),
                                                     DateUtil.get_next_trade_date(datetime.now().strftime('%Y%m%d')))
                        bar_quotation_data.trade_date = self.now_trade_date_tuple[1]
                elif TimeUtil.in_time_range('000000-023000'):
                    if datetime.now().strftime('%Y%m%d') == self.now_trade_date_tuple[0]:
                        bar_quotation_data.trade_date = self.now_trade_date_tuple[1]
                    else:
                        self.now_trade_date_tuple = (datetime.now().strftime('%Y%m%d'),
                                                     DateUtil.get_next_trade_date(datetime.now().strftime('%Y%m%d'),
                                                                                  operate='$gte'))
                        bar_quotation_data.trade_date = self.now_trade_date_tuple[1]

            return bar_quotation_data
        except Exception as e:
            mes = traceback.format_exc()
            print('depth_market_dat_to_symbol异常：' + str(mes))
            return None

    def __del__(self):
        self.api.RegisterSpi(None)
        self.api.Release()

def spi(front, broker, user, password):
    assert front and broker and user and password, "missing arguments"
    _spi = MdSpi(front, broker, user, password)
    th = threading.Thread(target=_spi.run)
    th.daemon = True
    th.start()
    secs = 5
    while secs:
        if not (_spi.connected and _spi.loggedin):
            secs -= 1
            time.sleep(1)
        else:
            break
    return _spi

def get_future_list()->list:
    future_codes = list()
    trade_date = datetime.now().strftime('%Y%m%d')
    future_code_list = SymbolUtil.get_future_code_and_type_list(trade_date)[0]
    for future_code in future_code_list:
        future_codes.append(SymbolUtil.symbol_to_ctp_code(future_code))
    return future_codes

def main():
    # "userid": "242943",
    # "password": "20252025Wld~~",
    # "brokerid": "9999",
    # "md_address": "tcp://180.168.146.187:10211",
    # "appid": "simnow_client_test",
    # "auth_code": "0000000000000000"

    _spi = spi("tcp://182.254.243.31:30011","9999", "242943", "20252025Wld~~")
    # _spi = spi("tcp://180.168.146.187:10211","9999", "242943", "20252025Wld~~")

    codes = get_future_list()
    # ["rb2510"]
    # print(codes[1])
    _spi.api.SubscribeMarketData(codes)
    # _spi.api.SubscribeMarketData(["rb2510"])

    secs = 15
    while secs:
        continue

if __name__ == '__main__':
    main()
