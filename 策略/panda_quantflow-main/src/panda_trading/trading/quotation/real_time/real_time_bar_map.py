import six

class RealTimeBarMap(object):
    def __init__(self, stock_real_time_bar_map, future_real_time_bar_map):
        self.future_real_time_bar_map = future_real_time_bar_map
        self.stock_real_time_bar_map = stock_real_time_bar_map

    def __getitem__(self, key):

        if not isinstance(key, six.string_types):
            # TODO：抛异常
            raise Exception('获取行情数据失败')

        key_len = len(key.split('.'))
        if key_len > 1:
            exchange = key.split('.')[1]

            if exchange == 'SZ' or exchange == 'SH':
                return self.stock_real_time_bar_map[key]

            else:
                return self.future_real_time_bar_map[key]
        else:
            return self.future_real_time_bar_map[key]
