import copy

from panda_backtest.backtest_common.constant.strategy_constant import CLOSE, SIDE_BUY

from panda_backtest.backtest_common.system.context.core_context import CoreContext


class FutureOrderSplitManager(object):
    def __init__(self):
        self.context = CoreContext.get_instance()

    def split_close_today_order(self, order_result):
        order_result_list = list()
        xb_back_test_positions = self.context.strategy_context.future_account_dict[order_result.account].positions[
            order_result.order_book_id]

        if order_result.side == SIDE_BUY:
            close_yd_pos = xb_back_test_positions.closable_sell_quantity - xb_back_test_positions.closable_today_sell_quantity
        else:
            close_yd_pos = xb_back_test_positions.closable_buy_quantity - xb_back_test_positions.closable_today_buy_quantity
        if order_result.market == 'SHFE' or order_result.market == 'INE':
            # 上期所和能源交易所先平昨再平今
            if order_result.is_td_close == 0:

                if order_result.quantity <= close_yd_pos:
                    yes_order_result = copy.deepcopy(order_result)
                    yes_order_result.is_td_close = 0
                    yes_order_result.is_close_local = 0
                    order_result_list.append(yes_order_result)
                else:
                    close_td_pos = order_result.quantity - close_yd_pos
                    if close_yd_pos > 0:
                        yes_order_result = copy.deepcopy(order_result)
                        yes_order_result.is_td_close = 0
                        yes_order_result.quantity = int(close_yd_pos)
                        yes_order_result.close_td_pos = 0
                        yes_order_result.is_close_local = 0
                        order_result_list.append(yes_order_result)
                    if close_td_pos > 0:
                        td_order_result = copy.deepcopy(order_result)
                        td_order_result.is_td_close = 1
                        td_order_result.quantity = int(close_td_pos)
                        td_order_result.close_td_pos = close_td_pos
                        td_order_result.is_close_local = 0
                        order_result_list.append(td_order_result)
            else:
                # 平今
                order_result.is_close_local = 0
                order_result_list.append(order_result)

        else:
            # 其他交易所只有平仓
            order_result.is_td_close = 0
            order_result.is_close_local = 0
            order_result_list.append(order_result)

        return order_result_list
