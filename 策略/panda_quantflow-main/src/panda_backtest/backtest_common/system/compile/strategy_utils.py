# -*- coding: utf-8 -*-
"""
File: strategy_utils.py
Author: peiqi
Date: 2025/5/14
Description: 
"""
import six
import logging
import builtins

import codecs
from panda_backtest.backtest_common.exception.error_exception import ErrorException
from panda_backtest.backtest_common.exception.strategy_exception_builder import StrategyExceptionBuilder
from panda_backtest.util.log.remote_log_factory import RemoteLogFactory

def create_strategy_print():
    """
    创建策略专用的print函数，将print输出重定向到SRLogger
    """
    sr_logger = RemoteLogFactory.get_sr_logger()
    original_print = builtins.print
    
    def strategy_print(*args, **kwargs):
        # 处理print的参数，模拟标准print行为
        if args:
            # 将所有参数转换为字符串并用空格连接
            sep = kwargs.get('sep', ' ')
            message = sep.join(str(arg) for arg in args)
            
            # 使用SRLogger的info方法输出
            sr_logger.info(message)
        else:
            # 空print调用，输出空行
            sr_logger.info('')
    
    return strategy_print

class FileStrategyLoader(object):

    def __init__(self, strategy_info, local_run):
        self._local_run = local_run
        if local_run:
            self._strategy_file_path = strategy_info
        else:
            self._strategy_file_path = '<string>'
            self.code = strategy_info

    def load(self, scope):
        if self._local_run:
            with codecs.open(self._strategy_file_path, encoding="utf-8") as f:
                source_code = f.read()
            return strategy_compile(source_code, self._strategy_file_path, scope)
        else:
            return strategy_compile(self.code, self._strategy_file_path, scope)

def strategy_compile(source_code, strategy, scope):
    try:
        # 在策略作用域中注入自定义的print函数
        scope['print'] = create_strategy_print()        
        code = compile(source_code, strategy, 'exec')
        six.exec_(code, scope)
        return scope
    except Exception as e:
        raise ErrorException(StrategyExceptionBuilder.build_strategy_compile_exception_msg(), '00001', str(e))

