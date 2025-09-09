from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY, CLOSE, REJECTED
import logging

from panda_backtest.backtest_common.data.quotation.quotation_data import QuotationData

from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
class OrderQuotationVerify(object):
    def __init__(self):
        self.context = CoreContext.get_instance()

    def get_order_market_price(self, order_result):
        bar_dict = QuotationData.get_instance().bar_dict
        sr_logger = RemoteLogFactory.get_sr_logger()
        strategy_context = self.context.strategy_context
        run_info = strategy_context.run_info
        if run_info.matching_type == 1:
            hq_price = bar_dict[order_result.order_book_id].open
        elif run_info.matching_type == 3:
            hq_price = bar_dict[order_result.order_book_id].last
        else:
            hq_price = bar_dict[order_result.order_book_id].close
        return hq_price
