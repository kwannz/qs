import json
import logging

class FutureRateManager(object):
    def __init__(self, rate_file):
        self.future_cost_rate_dict = None
        self.init_future_cost_rate(rate_file)

    def init_future_cost_rate(self, rate_file):
        with open(rate_file, 'r') as load_f:
            self.future_cost_rate_dict = json.load(load_f)

    def get_future_cost_rate(self, future, volume, money, close_td_volume, close_td_money, future_type):
        if future_type in self.future_cost_rate_dict.keys():
            future_cost_rate_dict = self.future_cost_rate_dict[future_type]
            if future_cost_rate_dict['CostType'] == 0:
                return volume * future_cost_rate_dict['CostRate'] + \
                       close_td_volume * future_cost_rate_dict['CloseTdCostRate']
            else:
                return money * future_cost_rate_dict['CostRate'] + \
                       close_td_money * future_cost_rate_dict['CloseTdCostRate']
        else:
            print('获取手续费失败,期货:' + str(future))
            return 0
