import logging
import logging.config
from pathlib import Path
from panda_server.config.env import LOG_FORMAT, LOG_LEVEL, LOG_CONSOLE, LOG_FILE, LOG_CONSOLE_ANSI
from pythonjsonlogger import jsonlogger

# 日志格式常量
LOG_FORMAT_STRING = "%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

"""
使用说明：

1. 在 panda_server/config/env.py 中配置相关环境变量 (LOG_LEVEL, LOG_CONSOLE, LOG_FILE, LOG_FORMAT 等)
2. 在 panda_server/main.py 中配置日志配置: setup_logging()
3. 在项目各个文件中按如下方法使用日志:
    import logging
    logger = logging.getLogger(__name__)
    logger.debug("...")
    logger.info("...")
    logger.warning("...")
    logger.error("...")
    logger.critical("...")
"""


class ColoredFormatter(logging.Formatter):
    """带ANSI颜色的日志格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        "DEBUG": "\033[38;5;240m",    # 深灰色（最低优先级）
        "INFO": "\033[38;5;75m",      # 天蓝色（更柔和的蓝）
        "WARNING": "\033[1;38;5;214m", # 橙色加粗（警告）
        "ERROR": "\033[1;4;38;5;196m", # 红色加粗下划线（错误）
        "CRITICAL": "\033[48;5;88;38;5;226;1m"  # 深红底亮黄字加粗
    }
    RESET = '\033[0m'  # 重置颜色
    
    def __init__(self, fmt=None, datefmt=None, use_colors=True):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
    
    def format(self, record):
        # 先格式化原始消息
        formatted = super().format(record)
        
        # 如果不使用颜色，直接返回
        if not self.use_colors:
            return formatted
            
        # 根据日志级别给整行添加颜色
        levelname = record.levelname
        if levelname in self.COLORS:
            # 给整行日志添加颜色
            formatted = f"{self.COLORS[levelname]}{formatted}{self.RESET}"
            
        return formatted


def setup_logging_directories():
    """确保日志目录存在"""
    # 获取项目根目录 (backend/)
    backend_root = Path(__file__).parent.parent
    log_dir = backend_root / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir


def get_log_config():
    """获取日志配置"""

    # 解析配置
    log_level = LOG_LEVEL.upper()
    console_enabled = LOG_CONSOLE.lower() == "true"
    file_enabled = LOG_FILE.lower() == "true"
    console_ansi_enabled = LOG_CONSOLE_ANSI.lower() == "true"
    log_format = LOG_FORMAT.lower()

    log_level = (
        log_level
        if log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        else "DEBUG"
    )

    # 判断是否启用控制台颜色
    use_console_colors = console_ansi_enabled and log_format == "plain"

    # 构建handlers配置
    handlers_config = {}
    handlers = []

    if console_enabled:
        # 根据是否启用颜色选择formatter
        if log_format == "json":
            console_formatter = "json"
        else:
            console_formatter = "colored" if use_console_colors else "plain"
        handlers_config["console"] = {
            "class": "logging.StreamHandler",
            "level": log_level,
            "formatter": console_formatter,
            "stream": "ext://sys.stdout",
        }
        handlers.append("console")

    if file_enabled:
        log_dir = setup_logging_directories()
        file_formatter = "json" if log_format == "json" else "plain"
        handlers_config.update(
            {
                "file_debug": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": file_formatter,  # 文件日志根据 log_format 选择
                    "filename": str(log_dir / f"panda_debug.log"),
                    "maxBytes": 10485760,
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
                "file_info": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": file_formatter,  # 文件日志根据 log_format 选择
                    "filename": str(log_dir / f"panda_info.log"),
                    "maxBytes": 10485760,
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
                "file_error": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": file_formatter,  # 文件日志根据 log_format 选择
                    "filename": str(log_dir / f"panda_error.log"),
                    "maxBytes": 10485760,
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
            }
        )
        handlers.extend(["file_debug", "file_info", "file_error"])

    # 构建formatters配置
    formatters_config = {
        "plain": {
            "format": LOG_FORMAT_STRING,
            "datefmt": LOG_DATE_FORMAT,
        },
        "json": {
            '()': jsonlogger.JsonFormatter,
            'fmt': '%(asctime)s %(levelname)s %(name)s %(message)s',
            'datefmt': LOG_DATE_FORMAT,
        },
    }
    
    # 如果启用控制台颜色，添加colored formatter
    if use_console_colors:
        formatters_config["colored"] = {
            "()": ColoredFormatter,
            "fmt": LOG_FORMAT_STRING,
            "datefmt": LOG_DATE_FORMAT,
            "use_colors": True,
        }

    config = {
        "version": 1,
        "disable_existing_loggers": False,  # keep third-party loggers
        "formatters": formatters_config,
        "handlers": handlers_config,
        "root": {
            "level": log_level,
            "handlers": handlers,
        },
        "loggers": {
            # 各个子包日志配置
            "panda_server": {
                "level": log_level,
                "handlers": handlers,
                "propagate": False,
            },
            "panda_trading": {
                "level": log_level,
                "handlers": handlers,
                "propagate": False,
            },
            "panda_ml": {"level": log_level, "handlers": handlers, "propagate": False},
            "panda_backtest": {
                "level": log_level,
                "handlers": handlers,
                "propagate": False,
            },
            "panda_quantflow": {
                "level": log_level,
                "handlers": handlers,
                "propagate": False,
            },
            "panda_plugins": {
                "level": log_level,
                "handlers": handlers,
                "propagate": False,
            },
            "common": {"level": log_level, "handlers": handlers, "propagate": False},
            
            # 其它 panda_ai 相关库日志配置
            "panda_common": {
                "level": log_level,
                "handlers": handlers,
                "propagate": False,
            },
            "panda_factor": {
                "level": log_level,
                "handlers": handlers,
                "propagate": False,
            },
            
            # 其它第三方库日志配置
            "uvicorn": {"level": log_level, "handlers": handlers, "propagate": False},
            "fastapi": {"level": log_level, "handlers": handlers, "propagate": False},
            "motor": {"level": "WARNING", "handlers": handlers, "propagate": False},
            "pymongo": {"level": "WARNING", "handlers": handlers, "propagate": False},
            "redis": {"level": "WARNING", "handlers": handlers, "propagate": False},
            "sklearn": {"level": "WARNING", "handlers": handlers, "propagate": False},
            "xgboost": {"level": "WARNING", "handlers": handlers, "propagate": False},
            "lightgbm": {"level": "WARNING", "handlers": handlers, "propagate": False},
            "torch": {"level": "WARNING", "handlers": handlers, "propagate": False},
            "httpx": {"level": "WARNING", "handlers": handlers, "propagate": False},
            "requests": {"level": "WARNING", "handlers": handlers, "propagate": False},
            "urllib3": {"level": "WARNING", "handlers": handlers, "propagate": False},
            "aiormq": {"level": "WARNING", "handlers": handlers, "propagate": False},
            "aio_pika": {"level": "WARNING", "handlers": handlers, "propagate": False},
        },
    }

    return config


def setup_logging():
    """初始化统一日志配置"""
    try:
        # 清理现有的handlers，防止累积
        root_logger = logging.getLogger()
        if root_logger.handlers:
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                if hasattr(handler, "close"):
                    handler.close()

        # 应用新配置
        config = get_log_config()
        logging.config.dictConfig(config)
        return True
    except Exception as e:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger = logging.getLogger(__name__)
        logger.error(f"日志配置初始化失败，使用基本配置: {e}")
        return False


# 在 import 时自动执行一次日志配置
setup_logging() 