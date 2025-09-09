import sys
import traceback

class RiskControlExceptionBuilder(object):

    @staticmethod
    def build_risk_control_compile_exception_msg(risk_control_name):
        exception_type, exception_object, exception_traceback = sys.exc_info()
        if hasattr(exception_object, 'lineno'):
            line_num = exception_object.lineno
            text = exception_object.text
            err_msg = exception_object.msg
            mes = '[%s]:风控代码编译异常，第%s行：%s \n异常类型：%s，异常信息：%s' % (
                str(risk_control_name), str(line_num), str(text), str(exception_type), str(err_msg))
        else:
            exception_type, exception_object, exception_traceback = sys.exc_info()
            tb = exception_traceback
            last_tb = None
            while tb.tb_next is not None:
                last_tb = tb.tb_next
                tb = tb.tb_next
            line_num = last_tb.tb_lineno
            err_msg = str(exception_object)
            mes = '[%s]:风控代码编译异常，第%s行，异常类型：%s，异常信息：%s' % (
                risk_control_name, str(line_num), str(exception_type), str(err_msg))
        return mes

    @staticmethod
    def build_risk_control_run_exception_msg(risk_control_name):
        print(traceback.format_exc())
        exception_type, exception_object, exception_traceback = sys.exc_info()
        tb = exception_traceback
        last_tb = None
        while tb.tb_next is not None:
            last_tb = tb.tb_next
            tb = tb.tb_next
        line_num = last_tb.tb_lineno
        err_msg = str(exception_object)
        mes = '[%s]:风控代码运行异常，第%s行 \n异常类型：%s，异常信息：%s' % (
            risk_control_name, str(line_num), str(exception_type), str(err_msg))
        return mes
