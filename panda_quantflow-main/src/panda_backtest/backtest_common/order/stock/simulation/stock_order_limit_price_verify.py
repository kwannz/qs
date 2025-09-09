from panda_backtest.backtest_common.constant.string_constant import STOCK_ORDER_FAILED_MESSAGE, \
ORDER_PRICE_TOO_HIGH, ORDER_PRICE_TOO_LOW, SYMBOL_LIMIT_HIGH, SYMBOL_LIMIT_LOW, SYMBOL_NO_QUOTATION
from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY, SIDE_SELL, CLOSE, REJECTED
from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData
from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.backtest_common.order.order_verify import OrderVerify
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
class StockOrderLimitPriceVerify(OrderVerify):

    def __init__(self):
        self.context = CoreContext.get_instance()

    def can_submit_order(self, account, order_result):
        sr_logger = RemoteLogFactory.get_sr_logger()
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        trade_date = strategy_context.trade_date

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

        trade_date = strategy_context.trade_date
        if bar.last == 0 or bar.trade_date != trade_date:
            order_result.message = STOCK_ORDER_FAILED_MESSAGE % (
                order_result.order_book_id, str(order_result.quantity),
                account, order_effect, order_side, order_result.order_id,
                SYMBOL_NO_QUOTATION)
            return False

        limit_up = bar.limit_up
        limit_down = bar.limit_down

        if limit_up == 0:
            limit_up = bar.open * 1.1
        if limit_down == 0:
            limit_down = bar.open * 0.9

        if limit_up == 0:
            limit_up = 9999
            sr_logger.error("涨停价格数据异常,不做限制")
        if limit_down == 0:
            limit_down = 0
            sr_logger.error("跌停价格数据异常,不做限制")

        jz_price = bar.last

        if jz_price >= limit_up and order_result.side == SIDE_BUY:
            order_result.message = STOCK_ORDER_FAILED_MESSAGE % (order_result.order_book_id, str(order_result.quantity),
                                                                 account, order_effect, order_side,
                                                                 order_result.order_id,
                                                                 SYMBOL_LIMIT_HIGH)

            return False
        if jz_price <= limit_down and order_result.side == SIDE_SELL:
            order_result.message = STOCK_ORDER_FAILED_MESSAGE % (order_result.order_book_id, str(order_result.quantity),
                                                                 account, order_effect, order_side,
                                                                 order_result.order_id,
                                                                 SYMBOL_LIMIT_LOW)
            return False

        if order_result.price > limit_up:
            order_result.message = STOCK_ORDER_FAILED_MESSAGE % (order_result.order_book_id, str(order_result.quantity),
                                                                 account, order_effect, order_side,
                                                                 order_result.order_id,
                                                                 ORDER_PRICE_TOO_HIGH %
                                                                 (str(order_result.price), str(limit_up)))
            return False
        if order_result.price < limit_down:
            order_result.message = STOCK_ORDER_FAILED_MESSAGE % (order_result.order_book_id, str(order_result.quantity),
                                                                 account, order_effect, order_side,
                                                                 order_result.order_id,
                                                                 ORDER_PRICE_TOO_LOW %
                                                                 (str(order_result.price), str(limit_down)))
            return False

        return True
