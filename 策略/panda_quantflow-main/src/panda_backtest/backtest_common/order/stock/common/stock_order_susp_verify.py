from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY, CLOSE
import logging

from panda_backtest.backtest_common.constant.string_constant import STOCK_ORDER_FAILED_MESSAGE, STOCK_SU_SP

from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData

from panda_backtest.backtest_common.system.context.core_context import CoreContext

from panda_backtest.backtest_common.order.order_verify import OrderVerify
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
class StockOrderSuSpVerify(OrderVerify):

    def __init__(self, quotation_mongo_db):
        self.context = CoreContext.get_instance()
        self.quotation_mongo_db = quotation_mongo_db
        self.su_sp_dict_data = dict()

    def can_submit_order(self, account, order_result):
        sr_logger = RemoteLogFactory.get_sr_logger()
        strategy_context = self.context.strategy_context
        bar_dict = QuotationData.get_instance().bar_dict
        bar_data_source = bar_dict.bar_data_source
        bar_data = bar_data_source.get_stock_daily_bar(order_result.order_book_id, strategy_context.trade_date)
        # if '停牌' in bar_data.trade_status:
        # if bar_data.is_suspended:
        if bar_data.trade_status==1:

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
                                                                 STOCK_SU_SP)
            return False
        return True
