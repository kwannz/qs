#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/11/9 18:23
# @Author : wlb
# @File   : risk_control_manager.py
# @desc   :
import json
import logging

from panda_backtest.backtest_common.constant.strategy_constant import REJECTED
from panda_backtest.backtest_common.exception.risk_control_exception import RiskControlException
from panda_backtest.backtest_common.exception.risk_control_exception_builder import RiskControlExceptionBuilder
from panda_backtest.backtest_common.system.compile.risk_control_loader import RiskControlLoader
from panda_backtest.backtest_common.system.event.event import ConstantEvent, Event
import collections

class RiskControlManager(object):

    def __init__(self, mongo_client, event_bus, strategy_context):
        self.mongo_client = mongo_client
        self.event_bus = event_bus
        self.strategy_context = strategy_context
        self.risk_control_init_func_dict = collections.OrderedDict()
        self.risk_control_before_trading_func_dict = collections.OrderedDict()
        self.risk_control_day_before_func_dict = collections.OrderedDict()
        self.risk_control_after_trading_func_dict = collections.OrderedDict()
        self.risk_control_handle_bar_func_dict = collections.OrderedDict()
        self.risk_control_order_verify_func_dict = collections.OrderedDict()
        self.risk_control_name_dict = dict()
        self.refresh_init_list = list()

    def load_risk_control(self, run_id, run_type=None):
        print('加载风控')
        strategy_risk_control_col = self.mongo_client.strategy_risk_control
        if run_type is None:
            strategy_risk_control_cur = strategy_risk_control_col.find(
                {'run_id': str(run_id)}, {'_id': 0}).sort([('risk_priority', -1)])
        else:
            strategy_risk_control_cur = strategy_risk_control_col.find(
                {'run_id': str(run_id), 'type': run_type}, {'_id': 0}).sort([('risk_priority', -1)])

        strategy_risk_control_list = list(strategy_risk_control_cur)

        if strategy_risk_control_list is not None and len(strategy_risk_control_list) > 0:
            risk_control_data_list = list()
            risk_control_param_dict = dict()
            for strategy_risk_control in strategy_risk_control_list:
                strategy_risk_control['risk_control_id'] = str(strategy_risk_control['risk_control_id'])
                risk_code = {}
                risk_code = RiskControlLoader(strategy_risk_control['risk_control_code'], False) \
                    .load(risk_code, strategy_risk_control['risk_control_name'])
                risk_control_data = {'risk_control_id': strategy_risk_control['risk_control_id'],
                                     'risk_code': risk_code, 'update_time': strategy_risk_control['update_time']}
                risk_control_data_list.append(risk_control_data)
                self.risk_control_name_dict[strategy_risk_control['risk_control_id']] = \
                    strategy_risk_control['risk_control_name']

                risk_control_params = strategy_risk_control.get('risk_control_params', None)

                if risk_control_params is not None:
                    risk_control_param_list = json.loads(risk_control_params)
                    for risk_control_param in risk_control_param_list:
                        if risk_control_param[0] == 0:
                            risk_control_param_dict[risk_control_param[1]] = float(risk_control_param[2])
                        else:
                            risk_control_param_dict[risk_control_param[1]] = risk_control_param[2]

            self.strategy_context.init_opz_params(risk_control_param_dict)
            self.handle_risk_control(risk_control_data_list)
            self.strategy_context.enable_risk_control = True

        else:
            print('无风控，重置')
            self.risk_control_init_func_dict = collections.OrderedDict()
            self.risk_control_before_trading_func_dict = collections.OrderedDict()
            self.risk_control_day_before_func_dict = collections.OrderedDict()
            self.risk_control_after_trading_func_dict = collections.OrderedDict()
            self.risk_control_handle_bar_func_dict = collections.OrderedDict()
            self.risk_control_order_verify_func_dict = collections.OrderedDict()
            self.risk_control_name_dict = dict()
            self.refresh_init_list = list()
            self.strategy_context.enable_risk_control = False

    def load_local_risk_control(self, strategy_risk_control_list):
        if strategy_risk_control_list is not None and len(strategy_risk_control_list) > 0:
            strategy_risk_control_data_list = list()
            for strategy_risk_control in strategy_risk_control_list:
                strategy_risk_control['risk_control_id'] = str(strategy_risk_control['risk_control_id'])
                risk_code = {}
                risk_code = RiskControlLoader(strategy_risk_control['risk_control_code_file'], True) \
                    .load(risk_code, strategy_risk_control['risk_control_name'])
                strategy_risk_control_data = {'risk_control_id': strategy_risk_control['risk_control_id'],
                                              'risk_code': risk_code,
                                              'update_time': strategy_risk_control['update_time']}
                strategy_risk_control_data_list.append(strategy_risk_control_data)
                self.risk_control_name_dict[strategy_risk_control['risk_control_id']] = \
                    strategy_risk_control['risk_control_name']

            self.handle_risk_control(strategy_risk_control_data_list)
            self.strategy_context.enable_risk_control = True

        else:
            self.strategy_context.enable_risk_control = False

    def get_risk_control_name(self, risk_control_id):
        return self.risk_control_name_dict.get(risk_control_id, '未知风控')

    def handle_risk_control(self, risk_control_data_list):
        risk_control_init_func_dict_new = collections.OrderedDict()
        risk_control_before_trading_func_dict_new = collections.OrderedDict()
        risk_control_day_before_func_dict_new = collections.OrderedDict()
        risk_control_after_trading_func_dict_new = collections.OrderedDict()
        risk_control_handle_bar_func_dict_new = collections.OrderedDict()
        risk_control_order_verify_func_dict_new = collections.OrderedDict()
        for risk_control_data in risk_control_data_list:
            risk_control_id = risk_control_data.get('risk_control_id')
            risk_code = risk_control_data.get('risk_code')
            risk_update_time = risk_control_data.get('update_time')
            risk_control_init = risk_code.get('risk_control_init', None)
            if risk_control_init is not None:
                self.handle_new_risk_control_data(risk_control_init_func_dict_new, self.risk_control_init_func_dict,
                                                  risk_control_id,
                                                  risk_update_time, risk_control_init)

            risk_control_before_trading = risk_code.get('risk_control_before_trading', None)
            if risk_control_before_trading is not None:
                self.handle_new_risk_control_data(risk_control_before_trading_func_dict_new,
                                                  self.risk_control_before_trading_func_dict,
                                                  risk_control_id,
                                                  risk_update_time, risk_control_before_trading)

            risk_control_day_before = risk_code.get('risk_control_day_before', None)
            if risk_control_day_before is not None:
                self.handle_new_risk_control_data(risk_control_day_before_func_dict_new,
                                                  self.risk_control_day_before_func_dict,
                                                  risk_control_id,
                                                  risk_update_time, risk_control_day_before)

            risk_control_after_trading = risk_code.get('risk_control_after_trading', None)
            if risk_control_after_trading is not None:
                self.handle_new_risk_control_data(risk_control_after_trading_func_dict_new,
                                                  self.risk_control_after_trading_func_dict,
                                                  risk_control_id,
                                                  risk_update_time, risk_control_after_trading)

            risk_control_handle_bar = risk_code.get('risk_control_handle_bar', None)
            if risk_control_handle_bar is not None:
                self.handle_new_risk_control_data(risk_control_handle_bar_func_dict_new,
                                                  self.risk_control_handle_bar_func_dict,
                                                  risk_control_id,
                                                  risk_update_time, risk_control_handle_bar)

            risk_control_order_verify = risk_code.get('risk_control_order_verify', None)
            if risk_control_order_verify is not None:
                self.handle_new_risk_control_data(risk_control_order_verify_func_dict_new,
                                                  self.risk_control_order_verify_func_dict,
                                                  risk_control_id,
                                                  risk_update_time, risk_control_order_verify)

        self.risk_control_init_func_dict = risk_control_init_func_dict_new
        self.risk_control_before_trading_func_dict = risk_control_before_trading_func_dict_new
        self.risk_control_day_before_func_dict = risk_control_day_before_func_dict_new
        self.risk_control_after_trading_func_dict = risk_control_after_trading_func_dict_new
        self.risk_control_handle_bar_func_dict = risk_control_handle_bar_func_dict_new
        self.risk_control_order_verify_func_dict = risk_control_order_verify_func_dict_new

    def handle_new_risk_control_data(self, risk_control_func_dict_new, risk_control_func_dict,
                                     risk_control_id, risk_update_time,
                                     risk_control_func):
        if risk_control_id in risk_control_func_dict.keys() \
                and risk_update_time == risk_control_func_dict[risk_control_id]['risk_update_time']:
            risk_control_func_dict_new[risk_control_id] = risk_control_func_dict[risk_control_id]
        else:
            risk_control_func_data = {'risk_control_id': risk_control_id, 'code': risk_control_func,
                                      'risk_update_time': risk_update_time}
            risk_control_func_dict_new[risk_control_id] = risk_control_func_data
            self.refresh_init_list.append(risk_control_id)

    def init_event(self):

        self.event_bus.register_handle(
            ConstantEvent.RISK_CONTROL_RELOAD, self.load_risk_control)

        self.event_bus.register_handle(
            ConstantEvent.RISK_CONTROL_INIT, self.run_risk_control_init)

        self.event_bus.register_handle(
            ConstantEvent.RISK_CONTROL_TRADING_BEFORE, self.run_risk_control_before_trading)

        self.event_bus.register_handle(
            ConstantEvent.RISK_CONTROL_DAY_BEFORE, self.run_risk_control_day_before)

        self.event_bus.register_handle(
            ConstantEvent.RISK_CONTROL_TRADING_AFTER, self.run_risk_control_after_trading)

        self.event_bus.register_handle(
            ConstantEvent.RISK_CONTROL_HANDLE_BAR, self.run_risk_control_handle_bar)

        self.event_bus.register_handle(
            ConstantEvent.RISK_CONTROL_ORDER_VERIFY, self.run_risk_control_order_verify)

    def run_risk_control_init(self, context):
        for risk_control_init_func_dict in self.risk_control_init_func_dict.values():
            key = risk_control_init_func_dict['risk_control_id']
            if key in self.refresh_init_list:
                risk_control_init = risk_control_init_func_dict['code']
                try:
                    risk_control_init(context, key)
                except Exception as e:
                    raise RiskControlException(
                        RiskControlExceptionBuilder.build_risk_control_run_exception_msg(
                            self.risk_control_name_dict.get(key)), '00001', None)
        self.refresh_init_list = list()

    def run_risk_control_before_trading(self, context):
        for risk_control_before_trading_dict in self.risk_control_before_trading_func_dict.values():
            key = risk_control_before_trading_dict['risk_control_id']
            risk_control_before_trading = risk_control_before_trading_dict['code']
            try:
                risk_control_before_trading(context, key)
            except Exception as e:
                raise RiskControlException(
                    RiskControlExceptionBuilder.build_risk_control_run_exception_msg(
                        self.risk_control_name_dict.get(key)), '00001', None)

    def run_risk_control_day_before(self, context):
        for risk_control_day_before_dict in self.risk_control_day_before_func_dict.values():
            key = risk_control_day_before_dict['risk_control_id']
            risk_control_day_before = risk_control_day_before_dict['code']
            try:
                risk_control_day_before(context, key)
            except Exception as e:
                raise RiskControlException(
                    RiskControlExceptionBuilder.build_risk_control_run_exception_msg(
                        self.risk_control_name_dict.get(key)), '00001', None)

    def run_risk_control_after_trading(self, context):
        for risk_control_after_trading_dict in self.risk_control_after_trading_func_dict.values():
            key = risk_control_after_trading_dict['risk_control_id']
            risk_control_after_trading = risk_control_after_trading_dict['code']
            try:
                risk_control_after_trading(context, key)
            except Exception as e:
                raise RiskControlException(
                    RiskControlExceptionBuilder.build_risk_control_run_exception_msg(
                        self.risk_control_name_dict.get(key)), '00001', None)

    def run_risk_control_handle_bar(self, context, bar):
        for risk_control_handle_bar_dict in self.risk_control_handle_bar_func_dict.values():
            key = risk_control_handle_bar_dict['risk_control_id']
            risk_control_handle_bar = risk_control_handle_bar_dict['code']
            try:
                risk_control_handle_bar(context, bar, key)
            except Exception as e:
                raise RiskControlException(
                    RiskControlExceptionBuilder.build_risk_control_run_exception_msg(
                        self.risk_control_name_dict.get(key)), '00001', None)

    def run_risk_control_order_verify(self, context, order):
        # print('校验：' + str(len(self.risk_control_order_verify_func_dict)))
        for risk_control_order_verify_dict in self.risk_control_order_verify_func_dict.values():
            key = risk_control_order_verify_dict['risk_control_id']
            risk_control_order_verify = risk_control_order_verify_dict['code']
            try:
                res = risk_control_order_verify(context, order, key)
            except Exception as e:
                raise RiskControlException(
                    RiskControlExceptionBuilder.build_risk_control_run_exception_msg(
                        self.risk_control_name_dict.get(key)), '00001', None)
            if res is False:
                order.status = REJECTED
                order.message = '未通过风控【%s】下单校验' % self.risk_control_name_dict.get(key)
                break
