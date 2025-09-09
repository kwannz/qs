import os
from pathlib import Path
from panda_trading.trading_route import project_dir

import dotenv
dotenv.load_dotenv(Path.joinpath(Path(project_dir) , ".env"))
print(os.getenv("MONGO_URI"))

import json
import time
import traceback

from common.connector.redis_client import RedisClient
from panda_trading.trading_route.manager.real_trade_manager import RealTradeManager
from common.config.project import ProjectConfig

class RedisTradeRouteServer(object):
    def __init__(self):
        self.redis_client = RedisClient()
        self.trade_manager = RealTradeManager()

    def start_listen_server(self):
        try:
            server_ip = ProjectConfig.get_config_parser(project_dir).get('server_ip', 'ip')
            route = 'real_trade_server:' + server_ip
            ps = self.redis_client.subscribe(route)
            print(f"{self.redis_client.get_config()}")
            print('RedisTradeRouteServer 成功启动实盘python服务器')
            print(f"正在监听频道：{route}")

            for item in ps.listen():  # 监听状态：有消息发布了就拿过来
                if item['type'] == 'message':
                    if bytes.decode(item['channel']) == route:
                        print(item['data'])
                        self.handle_route_sub_mes(None, None, None, bytes.decode(item['data']))
                else:
                    print(f"RedisTradeRouteServer 收到未知类型消息-{item}")
        except Exception as e:
            mes = traceback.format_exc()
            print('RedisTradeRouteServer 实盘服务器监听处理异常：' + str(mes))
            time.sleep(10)
            self.start_listen_server()

    def handle_route_sub_mes(self, ch, method, properties, body):
        print('RedisTradeRouteServer 收到请求，内容' + str(body))
        request_dict = json.loads(body)
        if request_dict['type'] == 'start_trade':
            run_id = request_dict['run_id']
            self.trade_manager.start_trade(run_id)
        else:
            run_id = request_dict['run_id']
            self.trade_manager.kill_run_trade(run_id)


if __name__ == '__main__':
    pass
    # real_trade_server = RedisTradeRouteServer()
    # real_trade_server.start_listen_server()