
class RiskApi(object):

    def __init__(self, mongo_db_client):
        self.mongo_db_client = mongo_db_client

    def get_risk_level(self, account):
        fields = {'_id': 0}
        qry_filter = {'account': account}
        collection = self.mongo_db_client.risk_control_level
        data_dict = collection.find_one(qry_filter, fields)
        if data_dict is None:
            return None
        else:
            return data_dict

    def save_risk_level(self, account, level, limit_rate):
        data_dict = {'account': account, 'risk_level': level, 'limit_rate': limit_rate}
        key = {'account': account}
        collection = self.mongo_db_client.risk_control_level
        collection.update(key, data_dict, upsert=True)

    def remove_pos_limit(self, account, symbol, business, custom_parameter):
        key = {'account': account}
        if symbol is not None:
            key['symbol'] = symbol

        if business is not None:
            key['business'] = business

        if custom_parameter is not None:
            key['custom_parameter'] = custom_parameter

        collection = self.mongo_db_client.position_limit
        collection.delete_many(key)

    def save_pos_limit(self, account, symbol, business, min_volume, custom_parameter):
        key = {'account': account, 'symbol': symbol, 'business': business, 'custom_parameter': custom_parameter}
        collection = self.mongo_db_client.position_limit
        cur = collection.find(key)
        cur_data = list(cur)
        if len(cur_data) > 0:
            min_volume = cur_data[0]['min_volume'] + min_volume
            data_dict = {'account': account, 'symbol': symbol, 'business': business, 'min_volume': min_volume,
                         'custom_parameter': custom_parameter}
            collection.update(key, {"$set": data_dict})
        else:
            data_dict = {'account': account, 'symbol': symbol, 'business': business, 'min_volume': min_volume,
                         'custom_parameter': custom_parameter}
            collection.insert_one(data_dict)

    def get_pos_limit(self, account, symbol, business, custom_parameter):
        fields = {'_id': 0}
        if symbol is None:
            qry_filter = {'account': account}
        else:
            qry_filter = dict()
            qry_filter['account'] = account
            qry_filter['symbol'] = symbol
            if business is not None:
                qry_filter['business'] = business

            if custom_parameter is not None:
                qry_filter['custom_parameter'] = custom_parameter
        collection = self.mongo_db_client.position_limit
        data_cur = collection.find(qry_filter, fields)
        return list(data_cur)

