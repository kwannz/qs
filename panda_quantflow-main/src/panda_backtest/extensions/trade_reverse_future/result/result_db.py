import copy
import logging

import json
import queue
import threading
import time

from bson import ObjectId

from panda_backtest import project_dir
from panda_backtest.backtest_common.system.context.core_context import CoreContext
from common.connector.mongodb_handler import DatabaseHandler
from common.config.config import config
import jsonpickle

class ResultDb(object):
    def __init__(self):
        self.result_queue = queue.Queue()
        self.context = CoreContext.get_instance()
        self.save_flag = True
        self.mongo_client =DatabaseHandler(config)
        self.result_thread = threading.Thread(target=self.save_to_db)
        self.result_thread.setDaemon(True)
        self.result_thread.start()

    def save_daily_data_to_db(self, all_account_list, all_position_list, all_trade_list, all_profit_dict):
        all_position_list = json.loads(jsonpickle.encode(all_position_list, unpicklable=False))
        all_trade_list = json.loads(jsonpickle.encode(all_trade_list, unpicklable=False))
        save_all_account_list = copy.deepcopy(all_account_list)
        self.result_queue.put_nowait((0, save_all_account_list, all_position_list, all_trade_list, all_profit_dict))

    def save_result_to_db(self, last_strategy_profit, ar, last_standard_profit, sr, alpha, beta, sharpe, vol,
                          md, info_ration, sortino, annual_te, kama_ratio, dw, benchmark_name):

        self.save_flag = False

        all_account_list = list()
        all_position_list = list()
        all_trade_list = list()
        all_profit_list = list()
        while not self.result_queue.empty():
            result_item = self.result_queue.get()
            all_account_list.extend(result_item[1])
            all_position_list.extend(result_item[2])
            all_trade_list.extend(result_item[3])
            all_profit_list.append(result_item[4])

        if len(all_account_list) > 0:
            self.save_account(all_account_list)
        if len(all_trade_list) > 0:
            self.save_trade(all_trade_list)
        if len(all_position_list) > 0:
            self.save_position(all_position_list)
        if len(all_profit_list) > 0:
            self.save_profit(all_profit_list)

        strategy_context = self.context.strategy_context
        update_dict = {'back_profit': last_strategy_profit,
                       'back_profit_year': ar,
                       'benchmark_profit': last_standard_profit,
                       'benchmark_profit_year': sr,
                       'alpha': alpha,
                       'beta': beta,
                       'sharpe': sharpe,
                       'volatility': vol,
                       'max_drawdown': md,
                       'information_ratio': info_ration,
                       'sortino': sortino,
                       'tracking_error': annual_te,
                       'kama_ratio': kama_ratio,
                       'downside_risk': dw,
                       'benchmark_name': benchmark_name,
                       'time_consume': time.time() - strategy_context.run_info.start_run_time,
                       'custom_tag': strategy_context.run_info.custom_tag,
                       'run_status': 1
                       }
        # result_col = self.mongo_client.xb_back_test
        strategy_context = self.context.strategy_context
        run_id = strategy_context.run_info.run_id
        # result_col.update_one({'_id': ObjectId(run_id)},
        #                       {'$set': update_dict})
        self.mongo_client.mongo_update_one(config["MONGO_DB"],collection_name="panda_back_test",query={"_id": ObjectId(run_id)},update={"$set": update_dict},upsert=True)

    def save_to_db(self):
        while self.save_flag:
            result_item = self.result_queue.get()
            if result_item[0] == 0:
                result_item_account_list = result_item[1]
                all_position_list = result_item[2]
                all_trade_list = result_item[3]
                all_profit_dict = result_item[4]
                if len(result_item_account_list) > 0:
                    self.save_account(result_item_account_list)
                if len(all_position_list) > 0:
                    self.save_position(all_position_list)
                if len(all_trade_list) > 0:
                    self.save_trade(all_trade_list)
                if len(all_profit_dict) > 0:
                    self.save_profit(all_profit_dict)

    def save_account(self, account_list):
        # account_col = self.mongo_client.xb_backtest_account
        # account_col.insert_many(account_list)
        self.mongo_client.mongo_insert_many(db_name=config["MONGO_DB"],collection_name="panda_backtest_account",documents=account_list)

    def save_position(self, all_position_list):
        # position_col = self.mongo_client.xb_backtest_position
        # position_col.insert_many(all_position_list)
        self.mongo_client.mongo_insert_many(db_name=config["MONGO_DB"],collection_name="panda_backtest_position",documents=all_position_list)

    def save_trade(self, all_trade_list):
        # trade_col = self.mongo_client.xb_backtest_trade
        # trade_col.insert_many(all_trade_list)
        self.mongo_client.mongo_insert_many(db_name=config["MONGO_DB"], collection_name="panda_backtest_trade",
                                            documents=all_trade_list)

    def save_profit(self, all_profit_data):
        # profit_col = self.mongo_client.xb_backtest_profit
        # profit_col.insert_one(all_profit_dict)
        
        try:
            print(f"save_profit 开始处理数据，类型: {type(all_profit_data)}")
            
            # 处理不同的数据类型
            if isinstance(all_profit_data, list):
                # 如果传入的是列表，检查每个元素的类型
                # print(f"处理列表数据，长度: {len(all_profit_data)}")
                documents = []
                for i, item in enumerate(all_profit_data):
                    if isinstance(item, dict):
                        documents.append(item)
                    elif hasattr(item, '__dict__'):
                        documents.append(item.__dict__)
                    else:
                        print(f"跳过无法处理的数据类型: {type(item)}")
                        continue
            else:
                # 如果传入的不是列表
                print("处理非列表数据")
                if isinstance(all_profit_data, dict):
                    documents = [all_profit_data]
                elif hasattr(all_profit_data, '__dict__'):
                    documents = [all_profit_data.__dict__]
                else:
                    print(f"无法处理的数据类型: {type(all_profit_data)}")
                    return
                    
            # print(f"准备插入 {len(documents)} 个文档")
            
            # 确保所有documents都是字典
            validated_documents = []
            for doc in documents:
                if isinstance(doc, dict):
                    validated_documents.append(doc)
                else:
                    print(f"跳过非字典文档: {type(doc)}")
                    
            if validated_documents:
                # print("开始MongoDB插入操作")
                self.mongo_client.mongo_insert_many(db_name=config["MONGO_DB"], collection_name="panda_backtest_profit",
                                                    documents=validated_documents)
                # print("MongoDB插入操作完成")
            else:
                print("没有有效的文档可以插入")
                
        except Exception as e:
            print(f"save_profit 执行出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def save_draw(self, chart_data):
        # chart = self.mongo_client.xb_custom_chart
        # context = self.context.strategy_context
        # relation_id = context.run_info.run_id
        # query = {'relation_id': relation_id}
        # value = {'relation_id': relation_id, 'data': chart_data}
        # chart.update(query, value, upsert=True)
        context = self.context.strategy_context
        relation_id = context.run_info.run_id
        value ={
            "$set":{
                'relation_id': relation_id, 'data': chart_data
            }
        }
        self.mongo_client.mongo_update_one(config["MONGO_DB"], collection_name="panda_custom_chart",
                                       query={"relation_id":relation_id}, update=value,upsert=True)

