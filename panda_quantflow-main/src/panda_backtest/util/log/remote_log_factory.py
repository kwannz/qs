#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/6/18 下午6:45
# @Author : wlb
# @File   : remote_log_factory.py
# @desc   :
from panda_backtest.util.log.local_logger import LocalLogger


class RemoteLogFactory(object):

    __sr_logger = None

    def __init__(self):
        pass

    @classmethod
    def init_sr_logger(cls, logger):
        if cls.__sr_logger is None:
            cls.__sr_logger = logger

    @classmethod
    def get_sr_logger(cls):
        if cls.__sr_logger is None:
            return LocalLogger
        return cls.__sr_logger
