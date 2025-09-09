from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY, CLOSE, MARKET, LIMIT, REJECTED

from panda_backtest.backtest_common.constant.string_constant import STOCK_ORDER_FAILED_MESSAGE, SYMBOL_NO_QUOTATION

from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
class StockOrderQuotationVerify(object):
    def __init__(self, quotation_mongo_db):
        pass

    def get_order_market_price(self, order_result):

        if order_result.price_type != MARKET:
            return order_result

        bar_dict = QuotationData.get_instance().bar_dict
        bar_data = bar_dict[order_result.order_book_id]
        sr_logger = RemoteLogFactory.get_sr_logger()
        if order_result.side == SIDE_BUY:
            order_side = '买入'
            order_effect = '开仓'
        else:
            order_side = '卖出'
            order_effect = '平仓'
        if bar_data.last == 0:
            sr_logger.error(STOCK_ORDER_FAILED_MESSAGE % (order_result.order_book_id, str(order_result.quantity),
                                                          order_result.account, order_effect, order_side,
                                                          order_result.order_id,
                                                          SYMBOL_NO_QUOTATION))
            order_result.status = REJECTED
            order_result.message = SYMBOL_NO_QUOTATION

        if order_result.side == SIDE_BUY:
            order_result.price = bar_data.askprice1
        else:
            order_result.price = bar_data.bidprice1

        return order_result
