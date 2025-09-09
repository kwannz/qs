from redis import Redis
import redis_lock
from common.connector.redis_client import RedisClient

class RedisClock(object):
    def __init__(self, redis_client: RedisClient, lock_name, expire, auto_renewal=True):
        conn = redis_client()
        self.lock = redis_lock.Lock(conn.client , lock_name, expire=expire, auto_renewal=auto_renewal)

    def get_lock(self):
        return self.lock
