from importlib import import_module
import logging
from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.backtest_common.system.event.engine import Engine
from panda_backtest.backtest_common.system.compile.strategy import Strategy
from panda_backtest.backtest_common.system.compile.strategy_utils import FileStrategyLoader
from panda_backtest.config.dev_init import DevInit
from panda_trading.trading.data.context.strategy_context import StrategyContext



class Run(object):

    @staticmethod
    def start(handle_message):

        run_id = handle_message.get('run_id', None)
        product_id = handle_message.get('product_id', None)
        DevInit.init_log_env(run_id)

        # 系统核心上下文 创建q
        strategy_context = StrategyContext()

        _context = CoreContext(strategy_context)

        DevInit.init_remote_sr_log(run_id, product_id, strategy_context)

        # 全局动态字典初始化
        global_args = {}

        run_type = 6

        if run_type is None:
            return

        global_args = FileStrategyLoader(
            handle_message['code'], False).load(global_args)
        handle_message['strategy_id'] = 1
        extension_module = import_module("redefine_trade.extensions.real_trade")
        extension = extension_module.load_extension()
        extension.create(_context)
        strategy = Strategy(global_args, _context.event_bus)

        # 项目启动
        Engine(_context).run(handle_message)
        log = logging.getLogger(__name__)
        log.info(f"{__name__} 项目正常启动完成 .....")


