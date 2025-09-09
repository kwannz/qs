#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 19-4-7 下午8:11
# @Author : wlb
# @File   : thread_util.py
# @desc   :
import threading
import time
import traceback


class ThreadUtil:

    @staticmethod
    def hand_cycle_thread(func, parmas, second):
        thread = threading.Thread(target=ThreadUtil.hand_cycle_func, name="循环线程", args=(func, parmas, second))
        thread.start()
        return thread

    @staticmethod
    def hand_cycle_func(func, params, second):
        while True:
            try:
                func(*params)
                time.sleep(second)
            except Exception as e:
                mes = traceback.format_exc()
                print('循环线程异常：', str(mes))
                time.sleep(second)

    @staticmethod
    def run_thread_func(func, params):
        thread = threading.Thread(target=func, name="循环线程", args=(params, ))
        thread.start()
