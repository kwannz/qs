from importlib import import_module

from panda_backtest.api.api import init_sr_logger
from panda_backtest.backtest_common.system.context.core_context import CoreContext
from panda_backtest.backtest_common.system.event.engine import Engine
from panda_backtest.backtest_common.system.compile.strategy import Strategy
from panda_backtest.backtest_common.system.compile.strategy_utils import FileStrategyLoader
from panda_backtest.config.dev_init import DevInit
from panda_trading.trading.data.context.strategy_context import StrategyContext
from utils.log.log_factory import LogFactory


class Run(object):

    @staticmethod
    def start(handle_message):
        run_id = handle_message.get('run_id', None)
        DevInit.init_log_env(run_id)

        # 系统核心上下文 创建q
        strategy_context = StrategyContext()

        _context = CoreContext(strategy_context)

        # TODO
        # DevInit.init_remote_sr_log(run_id, "ver", strategy_context)
        init_sr_logger()

        # 全局动态字典初始化
        global_args = {}

        run_type = 6

        if run_type is None:
            return
        global_args = FileStrategyLoader(handle_message['file'], True).load(global_args)
        handle_message['strategy_id'] = 1
        extension_module = import_module("panda_trading.trading.extensions.real_trade")
        extension = extension_module.load_extension()
        extension.create(_context)
        strategy = Strategy(global_args, _context.event_bus)

        print_log = LogFactory.get_logger()
        print_log.info("项目正常启动")
        # 项目启动
        Engine(_context).run(handle_message)



