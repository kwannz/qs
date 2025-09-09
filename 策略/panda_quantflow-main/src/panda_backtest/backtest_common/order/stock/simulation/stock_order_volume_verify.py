from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY, CLOSE, OPEN, SIDE_SELL

from panda_backtest.backtest_common.constant.string_constant import STOCK_ORDER_FAILED_MESSAGE, STOCK_SU_SP, STOCK_NO_VOLUME, \
    STOCK_VOLUME_NOT_ENOUGH

from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData

from panda_backtest.backtest_common.system.context.core_context import CoreContext

from panda_backtest.backtest_common.order.order_verify import OrderVerify
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
class StockOrderVolumeVerify(OrderVerify):

    def __init__(self):
        self.context = CoreContext.get_instance()

    def can_submit_order(self, account, order_result):
        sr_logger = RemoteLogFactory.get_sr_logger()
        bar_dict = QuotationData.get_instance().bar_dict
        ask_volume1 = bar_dict[order_result.order_book_id].askvolume1
        bid_volume1 = bar_dict[order_result.order_book_id].bidvolume1
        if ask_volume1 == 0 and order_result.side == SIDE_BUY:
            order_side = '买入'
            order_effect = '开仓'
            order_result.message = STOCK_ORDER_FAILED_MESSAGE % (order_result.order_book_id, str(order_result.quantity),
                                                                 account, order_effect, order_side,
                                                                 order_result.order_id,
                                                                 STOCK_NO_VOLUME)
            return False
        elif bid_volume1 == 0 and order_result.side == SIDE_SELL:
            order_side = '卖出'
            order_effect = '平仓'
            order_result.message = STOCK_ORDER_FAILED_MESSAGE % (order_result.order_book_id, str(order_result.quantity),
                                                                 account, order_effect, order_side,
                                                                 order_result.order_id,
                                                                 STOCK_NO_VOLUME)
            return False
        return True
