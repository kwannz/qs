from panda_backtest.backtest_common.constant.strategy_constant import CLOSE, SIDE_BUY
import logging

from panda_backtest.backtest_common.system.context.core_context import CoreContext

class FutureOrderSplitManager(object):
    def __init__(self):
        self.context = CoreContext.get_instance()

    def split_close_today_order(self, order_result):
        if order_result.effect != CLOSE:
            return order_result
        strategy_context = self.context.strategy_context
        future_account = strategy_context.future_account_dict[order_result.account]
        future_account_positions = future_account.positions
        if order_result.is_td_close == 0:
            if order_result.side == SIDE_BUY:
                left_pos = order_result.quantity - \
                           (future_account_positions[order_result.order_book_id].closable_sell_quantity -
                            future_account_positions[order_result.order_book_id].closable_today_sell_quantity)
            else:
                left_pos = order_result.quantity - \
                           (future_account_positions[order_result.order_book_id].closable_buy_quantity -
                            future_account_positions[order_result.order_book_id].closable_today_buy_quantity)
            if left_pos > 0:
                order_result.close_td_pos = left_pos
                order_result.unfilled_close_td_pos = left_pos

        else:
            order_result.close_td_pos = order_result.quantity
            order_result.unfilled_close_td_pos = order_result.close_td_pos

        return order_result
