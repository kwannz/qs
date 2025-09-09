from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY, CLOSE, SIDE_SELL

from panda_backtest.backtest_common.constant.string_constant import FUTURE_ORDER_FAILED_MESSAGE, FUTURE_NOT_LIMIT_DATA, \
    SYMBOL_LIMIT_HIGH, \
    SYMBOL_LIMIT_LOW, ORDER_PRICE_TOO_HIGH, ORDER_PRICE_TOO_LOW, SYMBOL_NO_QUOTATION
from panda_backtest.backtest_common.data.future.future_info_map import FutureInfoMap
from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData

from panda_backtest.backtest_common.system.context.core_context import CoreContext

class FutureOrderLimitPriceVerify(object):
    def __init__(self, quotation_mongo_db):
        self.context = CoreContext.get_instance()
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

        trad_price = order_result.price

        jz_price = bar.last

        if jz_price == 0 or bar.trade_date != self.context.strategy_context.trade_date:
            order_result.message = FUTURE_ORDER_FAILED_MESSAGE % (
                order_result.order_book_id, str(order_result.quantity),
                account, order_effect, order_side, order_result.order_id,
                SYMBOL_NO_QUOTATION)
            return False

        if bar.high >= jz_price >= bar.low:
            pass
        else:
            if jz_price >= bar.limit_up and order_result.side == SIDE_BUY:
                order_result.message = FUTURE_ORDER_FAILED_MESSAGE % (
                    order_result.order_book_id, str(order_result.quantity),
                    account, order_effect, order_side, order_result.order_id,
                    SYMBOL_LIMIT_HIGH)
                return False
            elif jz_price <= bar.limit_down and order_result.side == SIDE_SELL:
                order_result.message = FUTURE_ORDER_FAILED_MESSAGE % (
                    order_result.order_book_id, str(order_result.quantity),
                    account, order_effect, order_side, order_result.order_id,
                    SYMBOL_LIMIT_LOW)
                return False

        if bar.high >= trad_price >= bar.low:
            return True
        else:
            if trad_price > bar.limit_up:
                order_result.message = FUTURE_ORDER_FAILED_MESSAGE % (
                    order_result.order_book_id, str(order_result.quantity),
                    account, order_effect, order_side, order_result.order_id,
                    ORDER_PRICE_TOO_HIGH %
                    (str(order_result.price), str(bar.limit_up)))
                return False
            elif trad_price < bar.limit_down:
                order_result.message = FUTURE_ORDER_FAILED_MESSAGE % (
                    order_result.order_book_id, str(order_result.quantity),
                    account, order_effect, order_side, order_result.order_id,
                    ORDER_PRICE_TOO_LOW %
                    (str(order_result.price), str(bar.limit_down)))
                return False

        return True
