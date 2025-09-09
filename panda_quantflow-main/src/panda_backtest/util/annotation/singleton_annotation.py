#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-1-30 下午9:46
# @Author : wlb
# @File   : singleton_annotation.py
# @desc   : 单例注解

def singleton(cls):
    instances = {}

    def _singleton(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]
    return _singleton
