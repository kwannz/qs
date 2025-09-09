#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2019/6/20 下午6:15
# @Author : wlb
# @File   : dp_assest.py
# @desc   :

import json
import os
import time

from panda_backtest.backtest_commonmodel.result.xb_backtest_position import XbBacktestPosition
from redefine_trade.connector.redis_client import RedisClient
from redefine_trade.constant.redis_key import *
from tabulate import tabulate


class PrintAccount(object):

    __redis_client = RedisClient()

    @classmethod
    def start_print(cls, mock_id):
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            assets_data_json = cls.__redis_client.getHashRedis(
                real_trade_account_assets, str.encode(mock_id))
            print('#################################################账号###############################################')
            if assets_data_json:
                assets_data_list = json.loads(assets_data_json)
                print_list = list()

                for assets_data in assets_data_list:
                    print_list.append([assets_data['account_id'], assets_data['start_capital'], assets_data['total_profit'],
                                       assets_data['daily_pnl'], assets_data['holding_pnl'], assets_data['realized_pnl'], assets_data['available_funds'],
                                       assets_data['frozen_capital'], assets_data['cost'], assets_data['margin']])
                print(tabulate(print_list,
                               headers=['账号', '启动资金', '动态权益', '当日盈亏', '持仓盈亏', '平仓盈亏',
                                        '余额', '冻结资金', '手续费', '保证金']))

            print('####################################################################################################')

            print('\n' * 5)

            print('############################################持仓#####################################################')
            pos_data_json_list = cls.__redis_client.getHashRedis(
                real_trade_account_positions, str.encode(mock_id))
            # print(pos_data_json)
            pos_data_json_list = json.loads(pos_data_json_list)
            print_pos_list = list()
            if pos_data_json_list:
                for pos_data_list in pos_data_json_list:
                    # pos_data_list = json.loads(pos_data_json)

                    for pos_data in pos_data_list:
                        xb_back_test_pos = XbBacktestPosition()
                        xb_back_test_pos.__dict__.update(pos_data)
                        print_pos_list.append([xb_back_test_pos.contract_code, xb_back_test_pos.position,
                                               (pos_data[
                                                    'strategy_position'] if 'strategy_position' in pos_data.keys() else 0),
                                               xb_back_test_pos.td_position, xb_back_test_pos.yd_position,
                                               xb_back_test_pos.sellable,
                                               ('多' if 'direction' not in pos_data.keys() or pos_data[
                                                   'direction'] == 0 else '空'),
                                               xb_back_test_pos.last_price, xb_back_test_pos.cost,
                                               xb_back_test_pos.price,
                                               xb_back_test_pos.hold_price, xb_back_test_pos.margin,
                                               xb_back_test_pos.accumulate_profit])
                print(tabulate(print_pos_list,
                               headers=['合约', '仓位', '策略持仓', '今仓', '昨仓', '可平', '多空', '最新价', '手续费', '开仓价', '持仓价', '保证金',
                                        '收益']))
                print(
                    '####################################################################################################')
                time.sleep(1)


if __name__ == '__main__':
    PrintAccount.start_print('188889')