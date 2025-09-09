from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY, CLOSE, MARKET, LIMIT, REJECTED, SIDE_SELL
from panda_backtest.backtest_common.constant.string_constant import FUTURE_ORDER_FAILED_MESSAGE, SYMBOL_NO_QUOTATION
from panda_backtest.backtest_common.data.future.future_info_map import FutureInfoMap

from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData

class FutureOrderQuotationVerify(object):
    def __init__(self, quotation_mongo_db):
        self.future_info_map = FutureInfoMap(quotation_mongo_db)

    def get_order_market_price(self, order_result):
        if order_result.side == SIDE_BUY:
            order_side = '买入'
        else:
            order_side = '卖出'
        if order_result.effect == CLOSE:
            order_effect = '平仓'
        else:
            order_effect = '开仓'

        bar_dict = QuotationData.get_instance().bar_dict
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
