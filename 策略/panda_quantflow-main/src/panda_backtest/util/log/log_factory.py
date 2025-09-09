#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/6/6 下午2:48
# @Author : wlb
# @File   : log_factory.py
# @desc   :
from loguru import logger
import os


class LogFactory:

    __init_status = False

    @staticmethod
    def init_logger(file_name=None, log_dir=None, is_console=True):
        pid = os.getpid()
        if LogFactory.__init_status is False:
            if log_dir is None:
                log_dir = '~/sunrise/log'
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


# def test_print(arg):
#     logger = LogFactory.get_logger()
#     logger.info('子进程++' + str(arg))
#     while True:
#         logger.info('子进程++' + str(arg))
#         time.sleep(3)


if __name__ == '__main__':
    LogFactory.init_logger()
    LogFactory.init_logger()
    logger = LogFactory.get_logger()
    # logger.error('主进程启动')
    # for i in range(1, 10):
    #     p = Process(target=test_print, args=(i,))
    #     p.start()
    #
    # while True:
    #     logger.error('主进程启动')
    #     time.sleep(3)