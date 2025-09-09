from utils.thread.thread_util import ThreadUtil

class RestoreStrategy(object):

    def __init__(self):
        pass

    def init_restore_save(self, mock_id, strategy_context):
        ThreadUtil.hand_cycle_thread(self.start_restore_save, (mock_id, strategy_context), 3)

    def start_restore_save(self, mock_id, strategy_context):
        strategy_context.restore_save(mock_id)
        strategy_context.all_trade_reverse_result.standard_symbol_result.restore_save(mock_id)
        for stock_result in strategy_context.all_trade_reverse_result.stock_result_dict.values():
            stock_result.restore_save(mock_id)

        for future_result in strategy_context.all_trade_reverse_result.future_result_dict.values():
            future_result.restore_save(mock_id)

    def start_restore_read(self, mock_id, strategy_context):

        strategy_context.restore_read(mock_id)
        strategy_context.all_trade_reverse_result.standard_symbol_result.restore_read(mock_id)

        for stock_result in strategy_context.all_trade_reverse_result.stock_result_dict.values():
            stock_result.restore_read(mock_id)

        for future_result in strategy_context.all_trade_reverse_result.future_result_dict.values():
            future_result.restore_read(mock_id)
