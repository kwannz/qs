from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY, CLOSE, SIDE_SELL

from panda_backtest.backtest_common.constant.string_constant import FUTURE_ORDER_FAILED_MESSAGE, FUTURE_NOT_LIMIT_DATA, \
    SYMBOL_LIMIT_HIGH, \
    SYMBOL_LIMIT_LOW, ORDER_PRICE_TOO_HIGH, ORDER_PRICE_TOO_LOW
from panda_backtest.backtest_common.data.future.future_info_map import FutureInfoMap
from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData

from panda_backtest.util.log.remote_log_factory import RemoteLogFactory

from panda_backtest.backtest_common.system.context.core_context import CoreContext


class FutureOrderLimitPriceVerify(object):
    def __init__(self, quotation_mongo_db):
        self.future_info_map = FutureInfoMap(quotation_mongo_db)

    def can_submit_order(self, account, order_result):
        bar_dict = QuotationData.get_instance().bar_dict
        bar = bar_dict[order_result.order_book_id]
        if order_result.side == SIDE_BUY:
            order_side = '买入'
        else:
            order_side = '卖出'
        if order_result.effect == CLOSE:
            order_effect = '平仓'
        else:
            order_effect = '开仓'

        if order_result.side == SIDE_SELL and bar.last <= bar.limit_down:
            mes = FUTURE_ORDER_FAILED_MESSAGE % (order_result.order_book_id, str(order_result.quantity),
                                                 account, order_effect, order_side, order_result.order_id,
                                                 SYMBOL_LIMIT_LOW)
            order_result.message = mes
            return False
        elif order_result.side == SIDE_BUY and bar.last >= bar.limit_up:
            mes = FUTURE_ORDER_FAILED_MESSAGE % (order_result.order_book_id, str(order_result.quantity),
                                                 account, order_effect, order_side, order_result.order_id,
                                                 SYMBOL_LIMIT_HIGH)
            order_result.message = mes
            return False

        return True
