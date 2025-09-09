from panda_backtest.util.log.remote_log_factory import RemoteLogFactory

from panda_backtest.backtest_common.constant.strategy_constant import SIDE_BUY, CLOSE, OPEN
from panda_backtest.backtest_common.constant.string_constant import FUTURE_ORDER_FAILED_MESSAGE, ORDER_CASH_NOT_ENOUGH, \
    ORDER_POSITION_NOT_ENOUGH

from panda_backtest.backtest_common.system.context.core_context import CoreContext

from panda_backtest.backtest_common.order.order_verify import OrderVerify


class FutureOrderAccountVerify(OrderVerify):

    def __init__(self):
        self.context = CoreContext.get_instance()

    def can_submit_order(self, account, order_result):

        if order_result.side == SIDE_BUY:
            order_side = '买入'
        else:
            order_side = '卖出'
        if order_result.effect == CLOSE:
            order_effect = '平仓'
        else:
            order_effect = '开仓'

        if account not in self.context.strategy_context.future_account_dict.keys():
            order_result.message = FUTURE_ORDER_FAILED_MESSAGE % (
                order_result.order_book_id, str(order_result.quantity),
                account, order_effect, order_side, str(-1),
                '不存在当前期货账号')
            return False

        xb_back_test_account = self.context.strategy_context.future_account_dict[order_result.account]
        if xb_back_test_account.init_pos_status is False:
            order_result.message = FUTURE_ORDER_FAILED_MESSAGE % (
                order_result.order_book_id, str(order_result.quantity),
                account, order_effect, order_side, str(-1),
                '期货账号未初始化持仓')
            return False

        xb_back_test_positions = xb_back_test_account.positions[
            order_result.order_book_id]
        if order_result.effect == CLOSE:
            if order_result.side == SIDE_BUY:
                sell_quantity = xb_back_test_positions.closable_sell_quantity
                today_sell_quantity = xb_back_test_positions.closable_today_sell_quantity

                if order_result.is_td_close == 0 and order_result.quantity > sell_quantity:
                    message = FUTURE_ORDER_FAILED_MESSAGE % (order_result.order_book_id, str(order_result.quantity),
                                                             account, order_effect, order_side, order_result.order_id,
                                                             ORDER_POSITION_NOT_ENOUGH
                                                             % (str(sell_quantity)))
                    order_result.message = message
                    return False
                if order_result.is_td_close == 1 and order_result.quantity > today_sell_quantity:
                    message = FUTURE_ORDER_FAILED_MESSAGE % (order_result.order_book_id, str(order_result.quantity),
                                                             account, order_effect, order_side, order_result.order_id,
                                                             ORDER_POSITION_NOT_ENOUGH
                                                             % (str(today_sell_quantity)))
                    order_result.message = message
                    return False
            else:
                buy_quantity = xb_back_test_positions.closable_buy_quantity
                today_buy_quantity = xb_back_test_positions.closable_today_buy_quantity

                if order_result.is_td_close == 0 and order_result.quantity > buy_quantity:
                    message = FUTURE_ORDER_FAILED_MESSAGE % (order_result.order_book_id, str(order_result.quantity),
                                                             account, order_effect, order_side, order_result.order_id,
                                                             ORDER_POSITION_NOT_ENOUGH
                                                             % (str(buy_quantity)))
                    order_result.message = message
                    return False
                if order_result.is_td_close == 1 and order_result.quantity > today_buy_quantity:
                    message = FUTURE_ORDER_FAILED_MESSAGE % (order_result.order_book_id, str(order_result.quantity),
                                                             account, order_effect, order_side, order_result.order_id,
                                                             ORDER_POSITION_NOT_ENOUGH
                                                             % (str(today_buy_quantity)))
                    order_result.message = message
                    return False

        return True
