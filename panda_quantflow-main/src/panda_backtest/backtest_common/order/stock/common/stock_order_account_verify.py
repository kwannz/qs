
from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY, CLOSE
import logging

from panda_backtest.backtest_common.constant.string_constant import STOCK_ORDER_FAILED_MESSAGE, ORDER_CASH_NOT_ENOUGH, \
    ORDER_POSITION_NOT_ENOUGH

from panda_backtest.backtest_common.system.context.core_context import CoreContext

from panda_backtest.backtest_common.order.order_verify import OrderVerify
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory
class StockOrderAccountVerify(OrderVerify):

    def __init__(self):
        self.context = CoreContext.get_instance()

    def can_submit_order(self, account, order_result):
        sr_logger = RemoteLogFactory.get_sr_logger()
        strategy_context = self.context.strategy_context

        if account not in strategy_context.stock_account_dict.keys():
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
                                                                 '不存在当前股票账号')
            return False

        stock_account = strategy_context.stock_account_dict[account]

        if order_result.side == SIDE_BUY:
            """判断资金"""
            cash = stock_account.cash
            order_cash = order_result.price * order_result.quantity + order_result.transaction_cost
            if cash < order_result.price * order_result.quantity + order_result.transaction_cost:
                if order_result.side == SIDE_BUY:
                    order_side = '买入'
                else:
                    order_side = '卖出'
                if order_result.effect == CLOSE:
                    order_effect = '平仓'
                else:
                    order_effect = '开仓'
                order_result.message = STOCK_ORDER_FAILED_MESSAGE % (
                    order_result.order_book_id, str(order_result.quantity),
                    account, order_effect, order_side, order_result.order_id,
                    ORDER_CASH_NOT_ENOUGH
                    % (str(order_cash), str(cash)))
            else:
                return True
        else:
            """判断仓位"""
            position = stock_account.positions[order_result.order_book_id]
            if position.sellable < order_result.quantity:
                if order_result.side == SIDE_BUY:
                    order_side = '买入'
                else:
                    order_side = '卖出'
                if order_result.effect == CLOSE:
                    order_effect = '平仓'
                else:
                    order_effect = '开仓'
                order_result.message = STOCK_ORDER_FAILED_MESSAGE % (
                    order_result.order_book_id, str(order_result.quantity),
                    account, order_effect, order_side, order_result.order_id,
                    ORDER_POSITION_NOT_ENOUGH
                    % (str(position.sellable)))
            else:
                return True
