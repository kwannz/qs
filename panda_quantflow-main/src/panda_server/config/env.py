import os

# 运行模式: LOCAL 为用户本地运行, CLOUD 为 PandAaI 官网运行
RUN_MODE = os.getenv("RUN_MODE", "LOCAL")
SERVER_ROLE = os.getenv("SERVER_ROLE", "ALL")  # API, CONSUMER, ALL

# 日志配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
LOG_CONSOLE = os.getenv("LOG_CONSOLE", "true")
LOG_FILE = os.getenv("LOG_FILE", "true")
LOG_FORMAT = os.getenv("LOG_FORMAT", "plain")  # plain, json
LOG_CONSOLE_ANSI = os.getenv("LOG_CONSOLE_ANSI", "true")

# MongoDB 连接配置,其次默认从配置文件中加载
MONGO_URI = os.getenv("MONGO_URI", "localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "panda")
MONGO_USER = os.getenv("MONGO_USER", "panda")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "panda")
MONGO_AUTH_DB = os.getenv("MONGO_AUTH_DB", "admin")
MONGO_TYPE = os.getenv("MONGO_TYPE", "replica_set")  # 'single' 或 'replica_set'
MONGO_REPLICA_SET = os.getenv("MONGO_REPLICA_SET", "rs0")


# RabbitMQ 配置
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://admin:123456@localhost:5672")
RABBITMQ_MAX_RETRIES = os.getenv("RABBITMQ_MAX_RETRIES", 3)
RABBITMQ_RETRY_INTERVAL = os.getenv("RABBITMQ_RETRY_INTERVAL", 5)
RABBITMQ_PREFETCH_COUNT = os.getenv("RABBITMQ_PREFETCH_COUNT", 3)

# 工作流队列配置
WORKFLOW_EXCHANGE_NAME = os.getenv("WORKFLOW_EXCHANGE_NAME", "workflow.run")
WORKFLOW_ROUTING_KEY = os.getenv("WORKFLOW_ROUTING_KEY", "workflow.run")
WORKFLOW_RUN_QUEUE = os.getenv("WORKFLOW_RUN_QUEUE", "workflow_run")
WORKFLOW_LOG_ROUTING_KEY = os.getenv("WORKFLOW_LOG_ROUTING_KEY", "workflow.log")
WORKFLOW_LOG_QUEUE = os.getenv("WORKFLOW_LOG_QUEUE", "workflow_log")
PANDA_SERVER_WORKFLOW_WORKERS = os.getenv("PANDA_SERVER_WORKFLOW_WORKERS", 5)

# LLM 相关配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", None)
