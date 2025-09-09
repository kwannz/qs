#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/6/18 下午6:55
# @Author : wlb
# @File   : local_logger.py
# @desc   :


class LocalLogger(object):

    @staticmethod
    def info(content):
        # print('================未配置远程日志，本地打印================')
        print(content)

    @staticmethod
    def error(content):
        # print('================未配置远程日志，本地打印================')
        print(content)

    @staticmethod
    def warn(content):
        # print('================未配置远程日志，本地打印================')
        print(content)

    @staticmethod
    def debug(content):
        # print('================未配置远程日志，本地打印================')
        print(content)

    @staticmethod
    def process(i, total_num):
        pass
        # print('================未配置远程日志，无法打印进度================')

    @staticmethod
    def end():
        pass
        # print('==============
        # ==未配置远程日志，日志线程结束================')

    @staticmethod
    def risk(risk_control_name, content):
        print('[%s]风控日志: %s' % (str(risk_control_name), str(content)))