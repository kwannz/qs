
import sys
import logging

import traceback
import linecache

class StrategyExceptionBuilder(object):

    @staticmethod
    def build_strategy_compile_exception_msg():
        exception_type, exception_object, exception_traceback = sys.exc_info()
        if hasattr(exception_object, 'lineno'):
            line_num = exception_object.lineno
            text = exception_object.text
            err_msg = exception_object.msg
            mes = '策略代码编译异常，第%s行：%s \n异常类型：%s，异常信息：%s' % (
                str(line_num), str(text), str(exception_type), str(err_msg))
        else:
            exception_type, exception_object, exception_traceback = sys.exc_info()
            tb = exception_traceback
            last_tb = None
            while tb.tb_next is not None:
                last_tb = tb.tb_next
                tb = tb.tb_next
            line_num = last_tb.tb_lineno
            err_msg = str(exception_object)
            mes = '策略代码编译异常，第%s行， \n异常类型：%s，异常信息：%s' % (
                str(line_num), str(exception_type), str(err_msg))
        return mes

    @staticmethod
    def build_strategy_run_exception_msg():
        exception_type, exception_object, exception_traceback = sys.exc_info()
        tb = exception_traceback
        res_str = '策略运行异常,异常类型：%s' % str(exception_type)
        start_mes = False
        while tb is not None:
            file_name = tb.tb_frame.f_code.co_filename
            if file_name == '<string>' and start_mes is False:
                start_mes = True

            if start_mes:
                func = tb.tb_frame.f_code.co_name
                line_num = tb.tb_lineno
                if file_name == '<string>':
                    file_name = '策略代码'
                err_msg = file_name + ', '
                err_msg = err_msg + ('第%s行' % line_num) + (",函数：%s" % str(func))
                res_str = res_str + "\n"
                res_str = res_str + err_msg
            tb = tb.tb_next
        res_str = res_str + "\n" + "异常信息：" + str(exception_object)
        return res_str
