import json
import time
import traceback


import threading

from common.connector.redis_client import RedisClient
from utils.log.log_factory import LogFactory


class StrategySubPub(object):
    def __init__(self):
        self.redis_client = RedisClient()
        self.trade_signal_thread = None
        self.qry_account_thread = None
        self.logger = LogFactory.get_logger()

    def pub_data(self, sub_key, json_data):
        self.redis_client.public(sub_key, json_data)

    def init_sub_strategy_signal(self, strategy_context, sub_keys,  call_back):
        sub_thread = threading.Thread(target=self.start_sub_strategy_signal, name="订阅线程", args=(strategy_context,
                                                                                                sub_keys,
                                                                                                call_back))
        # 创建线程完毕之后，一定要启动
        sub_thread.setDaemon(True)
        sub_thread.start()
        self.trade_signal_thread = sub_thread

    def start_sub_strategy_signal(self, strategy_context, sub_keys,  call_back):
        try:
            ps = self.redis_client.subscribe_multiple(sub_keys)

            for item in ps.listen():  # 监听状态：有消息发布了就拿过来
                if item['type'] == 'pmessage':
                    body = bytes.decode(item['data'])
                    channel = bytes.decode(item['channel'])
                    call_back(strategy_context, channel, body)

        except Exception as e:
            mes = traceback.format_exc()
            self.logger.error('策略自定义监听处理异常：' + str(mes))
            time.sleep(10)
            self.start_sub_strategy_signal(strategy_context, sub_keys, call_back)

