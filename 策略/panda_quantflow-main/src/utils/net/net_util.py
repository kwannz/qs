import json
import requests
import socket


class NetUtil:

    @staticmethod
    def is_open(ip, port):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((ip, int(port)))
            s.shutdown(2)
            # 利用shutdown()函数使socket双向数据传输变为单向数据传输。shutdown()需要一个单独的参数，
            # 该参数表示了如何关闭socket。具体为：0表示禁止将来读；1表示禁止将来写；2表示禁止将来读和写。
            return True
        except:
            return False

    @staticmethod
    def get_local_ip():
        try:
            ips = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            ips.connect(('8.8.8.8', 80))
            ip = ips.getsockname()[0]
            return ip
        finally:
            ips.close()

    @staticmethod
    def get_out_ip():
        res = requests.get('http://httpbin.org/ip')
        if res.ok:
            return json.loads(bytes.decode(res.content))['origin']
        else:
            return None


if __name__ == '__main__':
    test = NetUtil.get_local_ip()
    print(test)
