from panda_backtest.backtest_common.model.result.order import Order
from common.connector.mongodb_handler import DatabaseHandler as MongoClient


class CtpMongoData(object):
    def __init__(self):
        self.ctp_mongo_db = MongoClient.get_mongo_db()

    def save_work_order(self, account, order):
        print('=================================更新订单=================================')
        print(order.__dict__)
        print('=================================更新订单=================================')
        collection = self.ctp_mongo_db.future_real_order
        key = {'order_id': order.order_id, 'account': account}
        collection.update_one(key, {"$set": order.__dict__}, upsert=True)

    def get_work_order_by_order_id(self, account, order_id):
        print('查询===》', str(order_id))
        collection = self.ctp_mongo_db.future_real_order
        order_dict = collection.find_one(
            {'order_id': order_id}, {'_id': 0})
        if order_dict:
            order = Order()
            order.__dict__ = order_dict
            return order
        else:
            return None

    def get_work_order_by_sys_id(self, account, order_sys_id, market, date):
        collection = self.ctp_mongo_db.future_real_order
        order_dict = collection.find_one(
            {'order_sys_id': order_sys_id, 'market': market, 'account': account, 'date': date}, {'_id': 0})
        if order_dict:
            order = Order()
            order.__dict__ = order_dict
            return order
        else:
            return None
