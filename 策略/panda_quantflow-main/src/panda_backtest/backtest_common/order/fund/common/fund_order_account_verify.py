from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY
import logging

from panda_backtest.backtest_common.constant.string_constant import FUND_ORDER_FAILED_MESSAGE, ORDER_CASH_NOT_ENOUGH, \
    ORDER_POSITION_NOT_ENOUGH
from panda_backtest.backtest_common.order.order_verify import OrderVerify
from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
class FundOrderAccountVerify(OrderVerify):

    def __init__(self):
        self.context = CoreContext.get_instance()

    def can_submit_order(self, account, order_result):
        sr_logger = RemoteLogFactory.get_sr_logger()
        strategy_context = self.context.strategy_context
        fund_account = strategy_context.fund_account_dict[account]
        if order_result.side == SIDE_BUY:
            """判断资金"""
            cash = fund_account.cash
            order_cash = order_result.purchase_amount
            if cash < order_result.purchase_amount:
                order_side = '申购'
                order_effect = '开仓'
                order_result.message = FUND_ORDER_FAILED_MESSAGE % (
                    order_result.order_book_id, str(order_result.quantity),
                    account, order_effect, order_side, order_result.order_id,
                    ORDER_CASH_NOT_ENOUGH
                    % (str(order_cash), str(cash)))
            else:
                return True
        else:
            """判断仓位"""
            position = fund_account.positions[order_result.order_book_id]
            if position.sellable < order_result.quantity:
                order_side = '赎回'
                order_effect = '平仓'
                order_result.message = FUND_ORDER_FAILED_MESSAGE % (
                    order_result.order_book_id, str(order_result.quantity),
                    account, order_effect, order_side, order_result.order_id,
                    ORDER_POSITION_NOT_ENOUGH
                    % (str(position.sellable)))
            else:
                return True
