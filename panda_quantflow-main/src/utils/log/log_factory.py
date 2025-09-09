from loguru import logger
import os

class LogFactory:

    __init_status = False
    @staticmethod
    def init_logger(file_name=None, log_dir=None, is_console=True):
        pid = os.getpid()
        if LogFactory.__init_status is False:
            if log_dir is None:
                log_dir = '/tmp/sunrise/log'
            if file_name is None:
                file_name = 'sunrise_' + str(pid) + '.log'
            else:
                file_name = file_name + '.log'

            if not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir)
                except Exception as e:
                    print("错误")

            # 禁止输出默认到sys.stderr
            if is_console is False:
                if 0 in logger._handlers.keys():
                    logger.remove(0)

            # if is_console is True:
            #     logger.add(StreamHandler(sys.stdout), colorize=True)
            logger.add(log_dir + '/' + file_name,
                       format="{time:YYYY-MM-DD HH:mm:ss} [{level}] {name}:{function}:{line} {message}",
                       rotation="00:00", level="INFO", enqueue=True, catch=False)
            # logger.add(StreamHandler(sys.stderr), format="{message}")
            LogFactory.__init_status = True

    @staticmethod
    def get_logger():
        return logger
