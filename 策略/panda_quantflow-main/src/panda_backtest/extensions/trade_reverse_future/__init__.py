#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 18-3-27 下午4:47
# @Author : wlb
# @File   : __init__.py.py
# @desc   :

def load_extension():
    from panda_backtest.extensions.trade_reverse_future.main import FutureTradingExtension

    return FutureTradingExtension()
