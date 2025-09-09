import json

from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY, CLOSE, MARKET, LIMIT, REJECTED, SIDE_SELL

from panda_backtest.backtest_common.constant.string_constant import FUTURE_ORDER_FAILED_MESSAGE, SYMBOL_NO_QUOTATION
from panda_backtest.backtest_common.data.future.future_info_map import FutureInfoMap
from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData
from panda_trading.trading.quotation.real_time.real_time_bar_map import RealTimeBarMap
from panda_trading.trading.quotation.tushare.tushare_future_tick_quotation import TushareFutureTickQuotation
from utils.log.log_factory import LogFactory
from common.connector.redis_client import RedisClient

class FutureOrderQuotationVerify(object):
    def __init__(self, quotation_mongo_db):
        self.future_info_map = FutureInfoMap(quotation_mongo_db)
        self.logger = LogFactory.get_logger()
        self.redis_client = RedisClient()

    def get_quotation(self) -> QuotationData:
        tushare_future_tick_quotation = TushareFutureTickQuotation(self.redis_client)
        real_time_bar_map = RealTimeBarMap(None, tushare_future_tick_quotation)
        quotation_data = QuotationData.get_instance()
        quotation_data.init_bar_dict(real_time_bar_map)
        return quotation_data


    def get_order_market_price(self, order_result):
        if order_result.side == SIDE_BUY:
            order_side = '买入'
        else:
            order_side = '卖出'
        if order_result.effect == CLOSE:
            order_effect = '平仓'
        else:
            order_effect = '开仓'

        bar_dict = self.get_quotation().bar_dict
        bar = bar_dict[order_result.order_book_id]
        if (order_result.side == SIDE_BUY and bar.askprice1 == float(1.7976931348623157e+308)) or \
                (order_result.side == SIDE_SELL and bar.bidprice1 == float(1.7976931348623157e+308)):
            order_result.status = REJECTED
            order_result.message = FUTURE_ORDER_FAILED_MESSAGE % (
                order_result.order_book_id, str(order_result.quantity),
                order_result.account, order_effect, order_side,
                order_result.order_id,
                SYMBOL_NO_QUOTATION)
            return order_result

        # 将市价换为限价
        if order_result.price_type == MARKET:
            print(order_result.order_book_id)
            instrument_info = self.future_info_map[order_result.order_book_id]
            ftminpricechg = instrument_info['ftminpricechg']
            if order_result.side == SIDE_BUY:
                order_result.price = int(round(bar.askprice1 / float(str(ftminpricechg)))) * \
                                     float(str(ftminpricechg))
            else:
                order_result.price = int(round(bar.bidprice1 / float(str(ftminpricechg)))) * \
                                     float(str(ftminpricechg))
            order_result.price_type = LIMIT
            if order_result.price == 0:

                order_result.status = REJECTED
                order_result.message = FUTURE_ORDER_FAILED_MESSAGE % (
                    order_result.order_book_id, str(order_result.quantity),
                    order_result.account, order_effect, order_side,
                    order_result.order_id,
                    SYMBOL_NO_QUOTATION)
        else:
            return order_result

        return order_result
