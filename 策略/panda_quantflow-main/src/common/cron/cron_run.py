#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/5/28 下午4:02
# @Author : wlb
# @File   : cron_run.py
# @desc   :
from .crontab_manager import CrontabManager


class CronRun(object):
    def start_cron(self, module_list):
        CrontabManager.init_cron_manager()
        CrontabManager.init_all_task(module_list)
        CrontabManager.start_scheduler()
