import json
import traceback

from panda_backtest.backtest_common.model.quotation.bar_quotation_data import BarQuotationData


class TushareFutureTickQuotation(object):

    def __init__(self, redis_client):
        self.my_bar_dict = dict()
        self.redis_client = redis_client
        self.tushare_quotation_key = 'tushare_future_tick_quotation'

    def __getitem__(self, item):
        try:
            bar_data_json = self.redis_client.getHashRedis(
                self.tushare_quotation_key, str.encode(item))
            if bar_data_json:
                bar_data = json.loads(bar_data_json)
                bar_quotation_data = BarQuotationData()
                bar_quotation_data.symbol = bar_data['symbol']
                # bar_quotation_data.code = bar_data['code']
                bar_quotation_data.date = str(bar_data['date'])
                bar_quotation_data.time = str(bar_data['time'])
                bar_quotation_data.trade_date = str(bar_data['trade_date'])
                bar_quotation_data.open = bar_data['open']
                bar_quotation_data.high = bar_data['high']
                bar_quotation_data.low = bar_data['low']
                bar_quotation_data.close = bar_data['close']
                bar_quotation_data.volume = bar_data['volume']
                bar_quotation_data.turnover = bar_data['turnover']
                # bar_quotation_data.vwap = bar_data['vwap']
                bar_quotation_data.oi = bar_data['oi']
                bar_quotation_data.settle = bar_data['settle']
                bar_quotation_data.last = bar_data['last']
                bar_quotation_data.preclose = bar_data['preclose']
                bar_quotation_data.limit_up = bar_data['limit_up']
                bar_quotation_data.limit_down = bar_data['limit_down']
                bar_quotation_data.askprice1 = bar_data['askprice1']
                bar_quotation_data.bidprice1 = bar_data['bidprice1']
                bar_quotation_data.askvolume1 = bar_data['askvolume1']
                bar_quotation_data.bidvolume1 = bar_data['bidvolume1']
                return bar_quotation_data

            bar = BarQuotationData()
            return bar
        except Exception as e:
            print('获取期货实时行情失败,合约：%s,原因：%s' % (str(item), str(traceback.format_exc())))
            bar = BarQuotationData()
            return bar

    def __setitem__(self, key, value):
        self.my_bar_dict[key] = value

    def keys(self):
        return self.my_bar_dict.keys()
