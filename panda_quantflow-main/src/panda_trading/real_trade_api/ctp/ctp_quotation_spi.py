import traceback
from datetime import datetime
from utils.time.time_util import TimeUtil
from utils.data.symbol_util import SymbolUtil
from panda_backtest.backtest_common.model.quotation.bar_quotation_data import BarQuotationData
from panda_trading.real_trade_api.ctp.data.future_info_map import FutureInfoMap
import ctp as mdapi

class CtpQuotationSpi(mdapi.CThostFtdcMdSpi):

    def __init__(self, ctp_quotation_api, account):
        super(CtpQuotationSpi, self).__init__()
        self.ctp_quotation_api = ctp_quotation_api
        self.account = account
        self.future_info_map = FutureInfoMap()

    def OnFrontConnected(self):
        try:
            print("期货订阅行情建立连接")
            self.ctp_quotation_api.account_login(self.account)
        except Exception:
            print('期货订阅行情建立连接异常，错误日志:%s', str(traceback.format_exc()))

    def OnFrontDisconnected(self, nReason):
        try:
            print('OnFrontDisconnected')
            print('期货订阅行情断开连接，原因' + str(nReason))
            self.ctp_quotation_api.quotation_qry_thread.change_con_status(False)
        except Exception as e:
            print('OnFrontConnected:', str(e))

    def OnRspUserLogin(self, pRspUserLogin, pRspInfo, nRequestID, bIsLast):
        try:
            print("OnRspUserLogin")
            rsploginfield = pRspUserLogin
            rspinfofield = pRspInfo
            print("SessionID=", rsploginfield.SessionID)
            print("ErrorID=", rspinfofield.ErrorID)
            print("ErrorMsg=", rspinfofield.ErrorMsg)
            self.ctp_quotation_api.quotation_qry_thread.change_con_status(True)
            self.ctp_quotation_api.trade_adapter.on_quotation_account_login(self.account)
        except Exception as e:
            print('OnRspUserLogin:', str(e))

    def OnRtnDepthMarketData(self, pDepthMarketData):
        if pDepthMarketData is not None:
            bar_quotation_data = BarQuotationData()

            if pDepthMarketData.ExchangeID is None or pDepthMarketData.ExchangeID == '':
                instrument_info = self.future_info_map.get_by_ctp_code(pDepthMarketData.InstrumentID)
                bar_quotation_data.symbol = instrument_info['symbol']
            else:
                bar_quotation_data.symbol = SymbolUtil.ctp_code_to_code(pDepthMarketData.InstrumentID,
                                                                  pDepthMarketData.ExchangeID)
            bar_quotation_data.time = pDepthMarketData.UpdateTime
            bar_quotation_data.trade_date = pDepthMarketData.TradingDay
            bar_quotation_data.open = pDepthMarketData.OpenPrice
            bar_quotation_data.high = pDepthMarketData.HighestPrice
            bar_quotation_data.low = pDepthMarketData.LowestPrice
            bar_quotation_data.close = pDepthMarketData.ClosePrice
            bar_quotation_data.volume = pDepthMarketData.Volume
            bar_quotation_data.oi = pDepthMarketData.OpenInterest
            bar_quotation_data.turnover = pDepthMarketData.Turnover
            # bar_quotation_data.vwap = bar_data['vwap']
            # bar_quotation_data.oi = bar_data['oi']
            bar_quotation_data.settle = pDepthMarketData.SettlementPrice
            bar_quotation_data.last = pDepthMarketData.LastPrice
            bar_quotation_data.preclose = pDepthMarketData.PreClosePrice
            bar_quotation_data.limit_up = pDepthMarketData.UpperLimitPrice
            bar_quotation_data.limit_down = pDepthMarketData.LowerLimitPrice
            bar_quotation_data.askprice1 = pDepthMarketData.AskPrice1
            bar_quotation_data.bidprice1 = pDepthMarketData.BidPrice1
            bar_quotation_data.askvolume1 = pDepthMarketData.AskVolume1
            bar_quotation_data.bidvolume1 = pDepthMarketData.BidVolume1
            bar_quotation_data.date = pDepthMarketData.ActionDay

            self.ctp_quotation_api.trade_adapter.on_tick_back(bar_quotation_data)

    def OnRspSubMarketData(self, *args):
        print("OnRspSubMarketData")
        field = args[0]
        print("InstrumentID=", field.InstrumentID)
        rspinfofield = args[1]
        print("ErrorID=", rspinfofield.ErrorID)
        print("ErrorMsg=", rspinfofield.ErrorMsg)
