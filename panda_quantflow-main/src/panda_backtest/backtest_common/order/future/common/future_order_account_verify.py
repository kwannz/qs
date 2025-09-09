
from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY, CLOSE, OPEN
import logging

from panda_backtest.backtest_common.constant.string_constant import FUTURE_ORDER_FAILED_MESSAGE, ORDER_CASH_NOT_ENOUGH, \
    ORDER_POSITION_NOT_ENOUGH

from panda_backtest.backtest_common.system.context.core_context import CoreContext

from panda_backtest.backtest_common.order.order_verify import OrderVerify

class FutureOrderAccountVerify(OrderVerify):

    def __init__(self):
        self.context = CoreContext.get_instance()

    def can_submit_order(self, account, order_result):
        strategy_context = self.context.strategy_context
        future_account = strategy_context.future_account_dict[account]

        if order_result.effect == OPEN:
            if order_result.side == SIDE_BUY:
                order_side = '买入'
            else:
                order_side = '卖出'
            if order_result.effect == CLOSE:
                order_effect = '平仓'
            else:
                order_effect = '开仓'
            if future_account.cash < order_result.margin + order_result.transaction_cost:
                order_result.message = FUTURE_ORDER_FAILED_MESSAGE % (
                    order_result.order_book_id, str(order_result.quantity),
                    account, order_effect, order_side, order_result.order_id,
                    ORDER_CASH_NOT_ENOUGH
                    % (
                        str(
                            order_result.margin + order_result.transaction_cost),
                        str(future_account.cash)))
                return False
        else:
            positions = future_account.positions
            if order_result.side == SIDE_BUY:
                order_side = '买入'
            else:
                order_side = '卖出'
            if order_result.effect == CLOSE:
                order_effect = '平仓'
            else:
                order_effect = '开仓'
            order_position = positions[order_result.order_book_id]
            if order_result.side == SIDE_BUY:
                if order_result.quantity > order_position.closable_sell_quantity \
                        or order_result.close_td_pos > order_position.closable_today_sell_quantity:
                    order_result.message = FUTURE_ORDER_FAILED_MESSAGE % (
                        order_result.order_book_id, str(order_result.quantity),
                        account, order_effect, order_side, order_result.order_id,
                        ORDER_POSITION_NOT_ENOUGH
                        % (str(order_position.closable_sell_quantity)))
                    return False
            else:
                if order_result.quantity > order_position.closable_buy_quantity \
                        or order_result.close_td_pos > order_position.closable_today_buy_quantity:
                    order_result.message = FUTURE_ORDER_FAILED_MESSAGE % (
                        order_result.order_book_id, str(order_result.quantity),
                        account, order_effect, order_side, order_result.order_id,
                        ORDER_POSITION_NOT_ENOUGH
                        % (str(order_position.closable_buy_quantity)))
                    return False

        return True
