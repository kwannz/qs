from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY, CLOSE
import logging

from panda_backtest.backtest_common.constant.string_constant import STOCK_ORDER_FAILED_MESSAGE, STOCK_SU_SP, STOCK_NO_VOLUME, \
    STOCK_VOLUME_NOT_ENOUGH

from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData

from panda_backtest.backtest_common.system.context.core_context import CoreContext

from panda_backtest.backtest_common.order.order_verify import OrderVerify

class StockOrderVolumeVerify(OrderVerify):

    def __init__(self):
        self.context = CoreContext.get_instance()

    def can_submit_order(self, account, order_result):
        bar_dict = QuotationData.get_instance().bar_dict
        now_volume = bar_dict[order_result.order_book_id].volume
        if now_volume == 0:
            if order_result.side == SIDE_BUY:
                order_side = '买入'
            else:
                order_side = '卖出'
            if order_result.effect == CLOSE:
                order_effect = '平仓'
            else:
                order_effect = '开仓'
            order_result.message = STOCK_ORDER_FAILED_MESSAGE % (order_result.order_book_id, str(order_result.quantity),
                                                                 account, order_effect, order_side,
                                                                 order_result.order_id,
                                                                 STOCK_NO_VOLUME)
            return False
        # else:
        #     if order_result.quantity > now_volume:
        #         if order_result.side == SIDE_BUY:
        #             order_side = '买入'
        #         else:
        #             order_side = '卖出'
        #         if order_result.effect == CLOSE:
        #             order_effect = '平仓'
        #         else:
        #             order_effect = '开仓'
        #         order_result.message = STOCK_ORDER_FAILED_MESSAGE % (
        #             order_result.order_book_id, str(order_result.quantity),
        #             account, order_effect, order_side, order_result.order_id,
        #             STOCK_VOLUME_NOT_ENOUGH % (str(now_volume)))
        #         return False
        return True
