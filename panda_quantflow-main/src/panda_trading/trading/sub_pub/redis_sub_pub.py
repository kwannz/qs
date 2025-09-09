import json
import time
import traceback
import threading
from common.connector.redis_client import RedisClient
from utils.log.log_factory import LogFactory
from utils.thread.thread_util import ThreadUtil
import datetime


class RedisSubPub(object):
    check_status_time = None
    def __init__(self):
        self.redis_client = RedisClient()
        self.trade_signal_thread = None
        self.qry_account_thread = None
        self.logger = LogFactory.get_logger()

    def init_sub_trade_signal(self, run_id, account_type, call_back):
        if account_type == 0:
            routing_key = 'stock'
        else:
            routing_key = 'future'

        print('监听的类型是：' + str(routing_key))
        sub_thread = threading.Thread(target=self.start_sub_trade_signal, name="订阅线程", args=(run_id,
                                                                                             routing_key, call_back))
        # 创建线程完毕之后，一定要启动
        sub_thread.setDaemon(True)
        sub_thread.start()
        self.trade_signal_thread = sub_thread

    def start_sub_trade_signal(self, run_id, routing_key, call_back):
        try:
            risk_routing_key = 'risk_reload_' + run_id

            if routing_key == 'stock':
                ps = self.redis_client.subscribe_multiple(['stock', risk_routing_key])
            else:
                ps = self.redis_client.subscribe_multiple(['future', risk_routing_key])

            for item in ps.listen():  # 监听状态：有消息发布了就拿过来
                print('有交易信息来了:' + str(item))
                if item['type'] == 'pmessage':
                    body = bytes.decode(item['data'])
                    print('收到请求，内容' + str(body))
                    body_dict = json.loads(body)
                    call_back(None, None, body_dict['time'], body_dict['data'])

        except Exception as e:
            mes = traceback.format_exc()
            print('交易信号监听处理异常：' + str(mes))
            time.sleep(10)
            self.start_sub_trade_signal(run_id, routing_key, call_back)

    def init_qry_account(self, call_back):
        self.qry_account_thread = ThreadUtil.hand_cycle_thread(call_back, (), 0.5)

    def init_check_thread(self):
        ThreadUtil.hand_cycle_thread(self.check_thread_status, (), 600)

    def check_thread_status(self):
        time_stamp = datetime.datetime.now()
        # 每 1 分钟输出一次
        if self.check_status_time is None or time_stamp - self.check_status_time > datetime.timedelta(minutes=1):
            self.check_status_time = time_stamp
            self.logger.info(
                "交易信号线程存活：%s, 时间：%s" % (str(self.trade_signal_thread.is_alive()), time_stamp.strftime('%Y.%m.%d-%H:%M:%S')))
            self.logger.info(
                "查询账号线程存活：%s, 时间：%s" % (str(self.qry_account_thread.is_alive()), time_stamp.strftime('%Y.%m.%d-%H:%M:%S')))


if __name__ == '__main__':
    # 测试执行框架
    redissubpub = RedisSubPub()
    run_id = '33'
    routing_key = 'future'
    call_back = None
    redissubpub.start_sub_trade_signal(run_id, routing_key, call_back)
