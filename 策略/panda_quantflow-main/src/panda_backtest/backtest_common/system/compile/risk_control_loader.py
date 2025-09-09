import sys
import traceback
import sys
import six
import codecs
from panda_backtest.backtest_common.exception.code import error_code
from panda_backtest.backtest_common.exception.error_exception import ErrorException
from panda_backtest.backtest_common.exception.risk_control_exception import RiskControlException
from panda_backtest.backtest_common.exception.risk_control_exception_builder import RiskControlExceptionBuilder
from panda_backtest.backtest_common.exception.strategy_exception_builder import StrategyExceptionBuilder


class RiskControlLoader(object):

    def __init__(self, strategy_info, local_run):
        self._local_run = local_run
        if local_run:
            self._strategy_file_path = strategy_info
        else:
            self._strategy_file_path = '<string>'
            self.code = strategy_info

    def load(self, scope, risk_control_name):
        if self._local_run:
            with codecs.open(self._strategy_file_path, encoding="utf-8") as f:
                source_code = f.read()
            return strategy_compile(source_code, self._strategy_file_path, scope, risk_control_name)
        else:
            return strategy_compile(self.code, self._strategy_file_path, scope, risk_control_name)


def strategy_compile(source_code, strategy, scope, risk_control_name):
    try:
        code = compile(source_code, strategy, 'exec')
        six.exec_(code, scope)
        return scope
    except Exception as e:
        raise RiskControlException(
            RiskControlExceptionBuilder.build_risk_control_compile_exception_msg(risk_control_name), '00001', None)
