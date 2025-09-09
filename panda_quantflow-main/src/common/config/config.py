"""
配置模块，用于加载和管理配置信息
支持从配置文件和环境变量导入，环境变量优先级更高
"""

import os
import logging

logger = logging.getLogger(__name__)

# 初始化配置变量
config = None


def load_config():
    """加载配置文件，并从环境变量更新配置"""
    global config
    config = {}
    # MongoDB
    config["MONGO_USER"] = os.getenv("MONGO_USER", "panda")
    config["MONGO_PASSWORD"] = os.getenv("MONGO_PASSWORD", "panda")
    config["MONGO_URI"] = os.getenv("MONGO_URI", "127.0.0.1:27017")
    config["MONGO_AUTH_DB"] = os.getenv("MONGO_AUTH_DB", "admin")
    config["MONGO_DB"] = os.getenv("MONGO_DB", "panda")
    config["MONGO_TYPE"] = os.getenv("MONGO_TYPE", "replica_set")
    config["MONGO_REPLICA_SET"] = os.getenv("MONGO_REPLICA_SET", "rs0")

    # 日志配置 Logging
    config["LOG_LEVEL"] = os.getenv("LOG_LEVEL", "DEBUG")
    config["log_file"] = os.getenv("LOG_FILE", "logs/data_cleaner.log")
    config["log_rotation"] = os.getenv("LOG_ROTATION", "1 MB")
    config["LOG_PATH"] = os.getenv("LOG_PATH", "logs")

    # Redis
    config["REDIS_HOST"] = os.getenv("REDIS_HOST", "localhost")
    config["REDIS_PORT"] = int(os.getenv("REDIS_PORT", 6379))
    config["REDIS_DB"] = int(os.getenv("REDIS_DB", 0))
    config["REDIS_PASSWORD"] = os.getenv("REDIS_PASSWORD", "123456")
    config["REDIS_MAX_CONNECTIONS"] = int(os.getenv("REDIS_MAX_CONNECTIONS", 10))
    config["REDIS_SOCKET_TIMEOUT"] = int(os.getenv("REDIS_SOCKET_TIMEOUT", 5))
    config["REDIS_CONNECT_TIMEOUT"] = int(os.getenv("REDIS_CONNECT_TIMEOUT", 5))

    # MySQL
    config["MYSQL_HOST"] = os.getenv("MYSQL_HOST", "localhost")
    config["MYSQL_PORT"] = int(os.getenv("MYSQL_PORT", 3306))
    config["MYSQL_USER"] = os.getenv("MYSQL_USER", "root")
    config["MYSQL_PASSWORD"] = os.getenv("MYSQL_PASSWORD", "qweqwe")
    config["MYSQL_DATABASE"] = os.getenv("MYSQL_DATABASE", "pandaai_test")

    return config


def get_config():
    """
    获取配置对象，如果配置未加载则先加载配置

    Returns:
        dict: 配置信息字典
    """
    global config
    if config is None:
        config = load_config()
    return config


# 初始加载配置
try:
    config = load_config()
    logger.info(f"初始化配置成功: {config}")
except Exception as e:
    logger.error(f"初始化配置失败: {str(e)}")
    # 不在初始化时抛出异常，留到实际使用时再处理
