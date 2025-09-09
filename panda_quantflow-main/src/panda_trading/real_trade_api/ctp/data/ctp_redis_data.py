import json

from panda_backtest.backtest_common.model.result.order import Order
from common.connector.redis_client import RedisClient


class CtpRedisData(object):
    def __init__(self):
        self.redis_client = RedisClient()

    def save_work_order(self, account, order):
        work_order_key = 'account_work_order:'
        self.redis_client.setHashRedis(work_order_key + account, order.order_id, json.dumps(order.__dict__))

    def get_work_order(self, account, order_id):
        work_order_key = 'account_work_order:'
        order = Order()
        data_json = self.redis_client.getHashRedis(work_order_key + account, order_id)
        if data_json:
            order.__dict__.update(json.loads(data_json))
            return order
        else:
            return None
