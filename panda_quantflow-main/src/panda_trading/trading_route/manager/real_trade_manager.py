import multiprocessing
import os
import platform
import time
import traceback

from panda_trading.trading.main_run import Run
from common.connector.mongodb_handler import DatabaseHandler as MongoClient
from common.connector.redis_client import RedisClient
from panda_trading.trading.constant.redis_key import real_trade_progress
from common.config import config
class RealTradeManager(object):

    def __init__(self):
        pass

    def start_trade(self, run_id):
        redis_client = RedisClient()
        key = real_trade_progress + run_id
        try:
            start_time = time.time()
            run_trade = self.get_run_trade(run_id)

            self.kill_run_trade(run_id)

            p1 = multiprocessing.Process(target=self.run_trade, args=(run_trade,))
            p1.start()
            print('启动耗时：' + str(time.time() - start_time))
            print('进程id：' + str(p1.ident))
            redis_client.setHashRedis(key, 'run_process_id', p1.ident)
            redis_client.setHashRedis(key, 'status', 1)
        except Exception:
            mes = traceback.format_exc()
            print('启动策略异常：==》' + str(mes))
            redis_client.setHashRedis(key, 'status', -1)
            redis_client.setHashRedis(key, 'err_mes', mes)

    def get_run_trade(self, trade_id):
        mongo_client = MongoClient(config).get_mongo_db()
        collection = mongo_client.get_collection("")
        run_trade = collection.find_one(
            {'product_strategy_id': str(trade_id)},
            {'_id': 0})
        run_trade['id'] = trade_id
        return run_trade

    def run_trade(self, run_trade):
        redis_client = RedisClient()
        if run_trade is None or run_trade is False:
            return None
        else:

            run_id = str(run_trade['id'])

            handle_message = dict()
            handle_message['is_start_risk_control'] = 1
            handle_message['future_account_id'] = run_trade.get('future_account_id', None)
            handle_message['stock_account_id'] = run_trade.get('account_id', None)
            handle_message['start_capital'] = run_trade.get('fund_stock', None)
            if handle_message['start_capital'] is not None:
                handle_message['start_capital'] = float(handle_message['start_capital'])
            handle_message['start_future_capital'] = run_trade.get('fund_futures', None)
            if handle_message['start_future_capital'] is not None:
                handle_message['start_future_capital'] = float(handle_message['start_future_capital'])
            handle_message['standard_symbol'] = ''
            handle_message['run_type'] = 2
            handle_message['back_test_id'] = str(run_trade['strategy_id'])
            handle_message['strategy_id'] = str(run_trade['strategy_id'])
            handle_message['strategy_name'] = run_trade.get('strategy_name', None)
            handle_message['product_id'] = str(run_trade['product_id'])
            handle_message['product_name'] = run_trade.get('product_name', None)
            handle_message['run_id'] = str(run_trade['id'])
            handle_message['account_type'] = 1
            handle_message['code'] = run_trade['strategy_code'].replace('5588', handle_message['future_account_id'])


            try:
                key = real_trade_progress + run_id

                Run.start(handle_message)
                print('正常启动')
                while True:
                    try:
                        redis_client.setHashRedis(key, 'update_time', int(round(time.time() * 1000)))
                        time.sleep(10)
                    except Exception as e:
                        print('更新策略状态失败，异常:' + str(e))

            except Exception:
                key = real_trade_progress + run_id
                mes = traceback.format_exc()
                print('运行失败', mes)
                redis_client.setHashRedis(key,
                                          'status', -1)
                redis_client.setHashRedis(key, 'err_mes', mes)

    def kill_run_trade(self, run_id):
        redis_client = RedisClient()
        # 判断是否有旧进程
        key = real_trade_progress + run_id

        old_run_info = redis_client.getHashRedis(key)
        if old_run_info:
            run_process_id = old_run_info.get(b'run_process_id', None)
            if run_process_id:
                old_process_pid = bytes.decode(run_process_id)
                if platform.system() == 'Linux' or platform.system() == 'Darwin':
                    cmd = 'kill ' + str(old_process_pid)
                    print('用unix like方式结束，%s' % str(cmd))
                    try:
                        os.system(cmd)
                        print(old_process_pid, 'killed')
                        redis_client.setHashRedis(key,
                                                  'status', 2)
                        redis_client.setHashRedis(key,
                                                  'err_mes', '手动停止')
                    except Exception as e:
                        print(e)
                else:
                    cmd = 'taskkill /pid ' + str(old_process_pid) + ' /f'
                    try:
                        os.system(cmd)
                        print(old_process_pid, 'killed')
                        redis_client.setHashRedis(key,
                                                  'status', 2)
                        redis_client.setHashRedis(key,
                                                  'err_mes', '手动停止')
                    except Exception as e:
                        print(e)
