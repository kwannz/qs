import datetime
import queue
import threading
import time
import traceback
from panda_backtest.api.api import *
import pandas as pd

from panda_backtest.backtest_common.constant.string_constant import FUTURE_ORDER_WX_MESSAGE
from panda_backtest.backtest_common.system.event.event import Event, ConstantEvent
from common.connector.redis_client import RedisClient
from utils.lock.redis_lock import RedisClock

class FutureGroupOrder(object):

    def __init__(self, account, work_order):
        self.account = account
        self.is_run = False
        self.context = CoreContext.get_instance()
        self.work_order = work_order
        self.order_queue = queue.Queue()
        # 保存下单结果集合，结构：（合约_开平_买卖：（预计数量，最终完成数量，撤单数量，报单途中数量））
        self.order_result_dict = dict()
        self.total_order_num = 0
        self.finish_order_num = 0
        self.redis_client = RedisClient()
        self.lock = None

    def init_data(self):
        pass

    def start_order(self, long_symbol_dict, short_symbol_dict):
        sr_logger = RemoteLogFactory.get_sr_logger()
        sr_logger.info('开始进行批量下单')
        if self.is_run:
            sr_logger.error('当前不可进行调仓操作，原因：有调仓操作进行中')
            return
        self.order_queue = queue.Queue()
        self.order_result_dict = dict()
        self.total_order_num = 0
        self.finish_order_num = 0

        if len(self.work_order.get_wait_work_oder()) > 0:
            sr_logger.error('当前不可进行调仓操作，原因：有未完成订单进行中')
            return
        try:
            self.handle_start_group_order(long_symbol_dict, short_symbol_dict)
        except Exception as e:
            sr_logger.error('调仓异常，原因：%s' % str(traceback.format_exc()))

    def handle_start_group_order(self, long_symbol_dict, short_symbol_dict):
        strategy_context = self.context.strategy_context
        future_account = strategy_context.future_account_dict[self.account]
        if future_account.init_pos_status is False:
            sr_logger = RemoteLogFactory.get_sr_logger()
            sr_logger.error('当前不可进行调仓操作，原因：期货账号持仓未初始化')
            return
        # 根据仓位计算下单
        positions = future_account.positions

        # 本次无持仓的合约先平仓
        now_holding_symbol_set = set(positions.keys())
        for symbol in now_holding_symbol_set:
            order_position = positions[symbol]
            if symbol not in set(list(long_symbol_dict.keys())) or long_symbol_dict[symbol] == 0:
                if order_position.buy_quantity > 0:
                    order = (symbol, CLOSE, SIDE_SELL, order_position.buy_quantity, 0, 0)
                    self.order_queue.put_nowait(order)
                    self.total_order_num += 1
                    order_result = list()
                    order_result.append(order_position.buy_quantity)
                    order_result.append(0)
                    order_result.append(0)
                    order_result.append(0)
                    self.order_result_dict[symbol + '_' + str(CLOSE) + '_' + str(SIDE_SELL)] = order_result
            if symbol not in set(list(short_symbol_dict.keys())) or short_symbol_dict[symbol] == 0:
                if order_position.sell_quantity > 0:
                    order = (symbol, CLOSE, SIDE_BUY, order_position.sell_quantity, 0, 0)
                    self.order_queue.put_nowait(order)
                    self.total_order_num += 1
                    order_result = list()
                    order_result.append(order_position.sell_quantity)
                    order_result.append(0)
                    order_result.append(0)
                    order_result.append(0)
                    self.order_result_dict[symbol + '_' + str(CLOSE) + '_' + str(SIDE_BUY)] = order_result

        # 多头持仓改变
        for symbol, position in long_symbol_dict.items():
            if position == 0:
                continue

            target_position = position
            if symbol in positions.keys():
                order_position = positions[symbol]
                target_position = target_position - order_position.buy_quantity

            if target_position == 0:
                continue

            if target_position > 0:
                order = (symbol, OPEN, SIDE_BUY, target_position, 0, 0)
                self.order_queue.put_nowait(order)
                self.total_order_num += 1
                order_result = list()
                order_result.append(target_position)
                order_result.append(0)
                order_result.append(0)
                order_result.append(0)
                self.order_result_dict[symbol + '_' + str(OPEN) + '_' + str(SIDE_BUY)] = order_result
            else:
                order = (symbol, CLOSE, SIDE_SELL, -target_position, 0, 0)
                self.order_queue.put_nowait(order)
                self.total_order_num += 1
                order_result = list()
                order_result.append(-target_position)
                order_result.append(0)
                order_result.append(0)
                order_result.append(0)
                self.order_result_dict[symbol + '_' + str(CLOSE) + '_' + str(SIDE_SELL)] = order_result

        # 空头持仓改变
        for symbol, position in short_symbol_dict.items():
            if position == 0:
                continue

            target_position = position
            if symbol in positions.keys():
                order_position = positions[symbol]
                target_position = target_position - order_position.sell_quantity

            if target_position == 0:
                continue

            if target_position > 0:
                order = (symbol, OPEN, SIDE_SELL, target_position, 0, 0)
                self.order_queue.put_nowait(order)
                self.total_order_num += 1
                order_result = list()
                order_result.append(target_position)
                order_result.append(0)
                order_result.append(0)
                order_result.append(0)
                self.order_result_dict[symbol + '_' + str(OPEN) + '_' + str(SIDE_SELL)] = order_result
            else:
                order = (symbol, CLOSE, SIDE_BUY, -target_position, 0, 0)
                self.order_queue.put_nowait(order)
                self.total_order_num += 1
                order_result = list()
                order_result.append(-target_position)
                order_result.append(0)
                order_result.append(0)
                order_result.append(0)
                self.order_result_dict[symbol + '_' + str(CLOSE) + '_' + str(SIDE_BUY)] = order_result

        if not self.order_queue.empty():
            self.is_run = True
            sr_logger = RemoteLogFactory.get_sr_logger()
            sr_logger.info('调仓准备进行，尝试获取调仓锁')
            redis_lock = RedisClock(self.redis_client, "future_group_order_account_lock", expire=60)
            self.lock = redis_lock.get_lock()
            if self.lock.acquire(timeout=150):
                try:
                    content = FUTURE_ORDER_WX_MESSAGE.format(str(self.context.strategy_context.run_info.product_name),
                                                             str(self.context.strategy_context.run_info.strategy_name),
                                                             str(self.account))
                    event = Event(ConstantEvent.SYSTEM_WX_NOTIFY, content=content)
                    event_bus = self.context.event_bus
                    event_bus.publish_event(event)
                    sr_logger.info('获取调仓锁成功, 5秒后将正式进入调仓')
                    time.sleep(5)
                    self.order_queue.put_nowait("first_round_end")
                    result_thread = threading.Thread(target=self.handle_group_order)
                    result_thread.start()

                except Exception as e:
                    sr_logger.error('调仓异常，原因：%s' % str(traceback.format_exc()))
                    self.lock.release()
                    sr_logger.info('调仓锁释放')
            else:
                sr_logger.error('未成功获取到锁')
                self.is_run = False
        else:
            sr_logger = RemoteLogFactory.get_sr_logger()
            sr_logger.info("当前无仓位需要进行调整")

    def handle_group_order(self):
        sr_logger = RemoteLogFactory.get_sr_logger()
        try:
            all_order_result_list = list()
            for order_result_key, order_result_value in self.order_result_dict.items():
                order_result = dict()
                order_result_key_list = order_result_key.split('_')
                order_result['合约'] = order_result_key_list[0]
                order_result['开平'] = '开仓' if int(order_result_key_list[1]) == OPEN else '平仓'
                order_result['买卖'] = '买入' if int(order_result_key_list[2]) == SIDE_BUY else '卖出'
                order_result['计划下单数量'] = order_result_value[0]
                all_order_result_list.append(order_result)

            all_order_result_pd = pd.DataFrame(all_order_result_list)
            sr_logger.info(all_order_result_pd)

            while True:
                order = self.order_queue.get()
                if order == 'first_round_end':
                    self.lock.release()
                    sr_logger.info('调仓锁释放')
                elif order == 'end':
                    self.is_run = False
                    sr_logger.info('批量下单完成')
                    all_order_result_list = list()
                    for order_result_key, order_result_value in self.order_result_dict.items():
                        order_result = dict()
                        order_result_key_list = order_result_key.split('_')
                        order_result['合约'] = order_result_key_list[0]
                        order_result['开平'] = '开仓' if int(order_result_key_list[1]) == OPEN else '平仓'
                        order_result['买卖'] = '买入' if int(order_result_key_list[2]) == SIDE_BUY else '卖出'
                        order_result['计划下单数量'] = order_result_value[0]
                        order_result['完成数量'] = order_result_value[1]
                        order_result['未完成数量'] = order_result_value[2]
                        all_order_result_list.append(order_result)

                    all_order_result_pd = pd.DataFrame(all_order_result_list)
                    sr_logger.info(all_order_result_pd)
                    return
                else:
                    symbol = order[0]
                    effect = order[1]
                    side = order[2]
                    position = int(order[3])
                    retry_num = order[4]
                    order_time = order[5]
                    if time.time() >= order_time:
                        order_result = self.order_result_dict[symbol + '_' + str(effect) + '_' + str(side)]
                        if (order_result[1] + order_result[2] + order_result[3]) > order_result[0]:
                            sr_logger.error("订单下单情况有异常，请检查，合约：{}".format(symbol))
                            return
                        order_result[3] += position
                        # print('正式开启下单，下单时间：' + str(datetime.datetime.now().strftime('%Y.%m.%d-%H:%M:%S')))
                        if effect == OPEN:
                            if side == SIDE_BUY:
                                order_list = buy_open(self.account, symbol, position, retry_num=retry_num,
                                                      remark='group_order')
                            else:
                                order_list = sell_open(self.account, symbol, position, retry_num=retry_num,
                                                       remark='group_order')
                        else:
                            if side == SIDE_BUY:
                                order_list = buy_close(self.account, symbol, position, retry_num=retry_num,
                                                       remark='group_order')
                            else:
                                order_list = sell_close(self.account, symbol, position, retry_num=retry_num,
                                                        remark='group_order')

                        if len(order_list) == 0:
                            order_result[2] += position
                            order_result[3] -= position
                            print('返回了空数组,%s,%s' % (str(int(order_result[2] + order_result[1])), str(order_result[0])))
                            if int(order_result[2] + order_result[1]) >= int(order_result[0]):
                                self.finish_order_num += 1

                                if self.finish_order_num >= self.total_order_num:
                                    end_flag = 'end'
                                    self.order_queue.put_nowait(end_flag)

                    else:
                        self.order_queue.put_nowait(order)
        except Exception as e:
            sr_logger.error("期货批量下单异常，异常日志：" + str(e))

    def on_future_order_cancel(self, order):
        if order.remark != 'group_order':
            return

        order_result = self.order_result_dict[
            order.order_book_id + '_' + str(order.effect) + '_' + str(order.side)]
        order_result[3] -= order.unfilled_quantity
        max_retry = 5
        if hasattr(order, 'retry_num'):
            now_retry_num = order.retry_num + 1
            if now_retry_num >= max_retry:

                order_result[2] += order.unfilled_quantity
                if int(order_result[2] + order_result[1]) >= int(order_result[0]):
                    self.finish_order_num += 1

                    if self.finish_order_num == self.total_order_num:
                        end_flag = 'end'
                        self.order_queue.put_nowait(end_flag)
                return
        else:
            now_retry_num = 2

        quantity = order.unfilled_quantity

        if order.effect == OPEN:
            if order.side == SIDE_BUY:
                order = (order.order_book_id, OPEN, SIDE_BUY, quantity, now_retry_num, time.time() + 6)
                self.order_queue.put_nowait(order)
            else:
                order = (order.order_book_id, OPEN, SIDE_SELL, quantity, now_retry_num, time.time() + 6)
                self.order_queue.put_nowait(order)
        else:
            if order.side == SIDE_BUY:
                order = (order.order_book_id, CLOSE, SIDE_BUY, quantity, now_retry_num, time.time() + 6)
                self.order_queue.put_nowait(order)
            else:
                order = (order.order_book_id, CLOSE, SIDE_SELL, quantity, now_retry_num, time.time() + 6)
                self.order_queue.put_nowait(order)

    def on_future_rtn_trade(self, trade):
        if trade.order_remark != 'group_order':
            return

        if trade.contract_code + '_' + str(trade.direction) + '_' + str(
                trade.business) in self.order_result_dict.keys():
            order_result = self.order_result_dict[
                trade.contract_code + '_' + str(trade.direction) + '_' + str(trade.business)]
            order_result[1] += trade.volume
            order_result[3] -= trade.volume
            if order_result[1] + order_result[2] >= order_result[0]:
                self.finish_order_num += 1

                if self.finish_order_num == self.total_order_num:
                    end_flag = 'end'
                    self.order_queue.put_nowait(end_flag)
