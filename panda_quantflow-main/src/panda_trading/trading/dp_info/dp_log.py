from common.connector.mongodb_handler import DatabaseHandler as MongoClient


class PrintLog(object):

    __mongo_client = MongoClient.get_mongo_db()

    @classmethod
    def start_print(cls, mock_id):
        collection = cls.__mongo_client.xb_user_strategy_log_test
        while True:
            cur = collection.find({'relation_id': mock_id}).sort({'_id': -1}).limit(1)

            print_data_list = list()
            for cur_item in cur:
                print(cur_item)

if __name__ == '__main__':
    PrintLog.start_print('5d0b2696d878f75e5c4651ac')