
import os
import time

from common.connector.mongodb_handler import DatabaseHandler as  MongoClient
from tabulate import tabulate


class PrintTrade(object):

    __mongo_client = MongoClient.get_mongo_db()

    @classmethod
    def start_print(cls, mock_id):
        collection = cls.__mongo_client.real_trade
        while True:

            cur = collection.find({'mock_id': mock_id})
            print_data_list = list()
            for cur_item in cur:
                print_data_list.append([cur_item['contract_code'], ('买' if cur_item['business'] == 0 else '卖'),
                                        cur_item['volume'],
                                        cur_item['price'], ('开' if cur_item['direction'] == 0 else '平'),
                                        str(cur_item['gmt_create']) + ' ' + str(cur_item['gmt_create_time'])])

            os.system('cls' if os.name == 'nt' else 'clear')
            print('###############################################成交记录#############################################')
            print(tabulate(print_data_list,
                           headers=['合约', '买卖', '数量', '成交价', '开平', '交易时间']))
            print('####################################################################################################')
            time.sleep(3)


if __name__ == '__main__':
    PrintTrade.start_print('43')