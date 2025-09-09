#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 19-3-25 下午6:59
# @Author : wlb
# @File   : trade_route_server.py
# @desc   :

from tornado import web
import tornado.ioloop

from redefine_account_monitor.server.monitor_handler import MonitorHandler

if __name__ == '__main__':
    app = web.Application([
        (r"/start_monitor", MonitorHandler),
    ]) 
    app.listen(9099)
    print('===============期货账号监控服务器启动================')
    tornado.ioloop.IOLoop.instance().start()


