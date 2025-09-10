import logging
import os
import sys
import inspect
from datetime import datetime
from logging.handlers import RotatingFileHandler
from functools import wraps

def _ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _build_formatter(fmt: str = 'plain') -> logging.Formatter:
    if fmt.lower() == 'json':
        try:
            from pythonjsonlogger import jsonlogger  # type: ignore
            return jsonlogger.JsonFormatter('%(asctime)s %(name)s %(levelname)s %(message)s')
        except Exception:
            pass
    return logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                             datefmt='%Y-%m-%d %H:%M:%S')


def _init_root_logger():
    # Read env for runtime control
    level = os.getenv('LOG_LEVEL', 'INFO').upper()
    fmt = os.getenv('LOG_FORMAT', 'plain')
    console_on = os.getenv('LOG_CONSOLE', '1') in ('1', 'true', 'True')
    file_on = os.getenv('LOG_FILE', '1') in ('1', 'true', 'True')
    log_dir = os.getenv('LOG_DIR', 'logs')
    rotate_size = os.getenv('LOG_ROTATE_SIZE', '1048576')  # bytes, default 1MB
    backup_count = int(os.getenv('LOG_BACKUP_COUNT', '5'))

    try:
        lvl = getattr(logging, level)
    except Exception:
        lvl = logging.INFO

    formatter = _build_formatter(fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(lvl)

    # Avoid duplicate handlers on re-import
    if not getattr(root_logger, '_panda_logger_inited', False):
        if console_on:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

        if file_on:
            _ensure_dir(log_dir)
            current_date = datetime.now().strftime('%Y%m%d')
            info_path = os.path.join(log_dir, f'panda_info_{current_date}.log')
            err_path = os.path.join(log_dir, f'panda_error_{current_date}.log')

            try:
                max_bytes = int(rotate_size)
            except Exception:
                max_bytes = 1_048_576

            file_handler = RotatingFileHandler(info_path, maxBytes=max_bytes, backupCount=backup_count)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

            error_file_handler = RotatingFileHandler(err_path, maxBytes=max_bytes, backupCount=backup_count)
            error_file_handler.setFormatter(formatter)
            error_file_handler.setLevel(logging.ERROR)
            root_logger.addHandler(error_file_handler)

        setattr(root_logger, '_panda_logger_inited', True)


_init_root_logger()

def get_logger(module_name):
    """
    Get logger for specified module
    
    Args:
        module_name: Module name, typically use __name__
        
    Returns:
        Logger: Configured logger instance
    """
    logger = logging.getLogger(module_name)
    # 级别跟随根 logger，可由环境变量控制
    # 单独模块如需更细粒度，可自行 setLevel
    return logger

# Global cache for created loggers to avoid duplicates
_module_loggers = {}

class LoggerInjector:
    @staticmethod
    def _get_caller_module_name():
        """Get the caller's module name"""
        frame = inspect.currentframe()
        if frame is None:
            return "__unknown__"
            
        # Get two levels up call frame (skip get_logger and current method)
        caller_frame = None
        try:
            # Need deeper frame level since this is accessed as property
            if frame.f_back is not None and frame.f_back.f_back is not None:
                caller_frame = frame.f_back.f_back
            module = inspect.getmodule(caller_frame)
            return module.__name__ if module else "__main__"
        except (AttributeError, TypeError):
            return "__unknown__"
        finally:
            # Clean up references to prevent memory leaks
            del frame
            if caller_frame:
                del caller_frame
    
    @classmethod
    def get_logger(cls):
        """Get logger for the module calling this method"""
        module_name = cls._get_caller_module_name()
        if module_name not in _module_loggers:
            _module_loggers[module_name] = get_logger(module_name)
        return _module_loggers[module_name]

# Module-level logger that forwards all calls to the appropriate module logger
class Logger:
    """
    A proxy class that forwards all logging methods to the appropriate module logger.
    This allows using 'logger.info()' directly without instantiation.
    """
    def debug(self, msg, *args, **kwargs):
        return LoggerInjector.get_logger().debug(msg, *args, **kwargs)
        
    def info(self, msg, *args, **kwargs):
        return LoggerInjector.get_logger().info(msg, *args, **kwargs)
        
    def warning(self, msg, *args, **kwargs):
        return LoggerInjector.get_logger().warning(msg, *args, **kwargs)
        
    def error(self, msg, *args, **kwargs):
        return LoggerInjector.get_logger().error(msg, *args, **kwargs)
        
    def critical(self, msg, *args, **kwargs):
        return LoggerInjector.get_logger().critical(msg, *args, **kwargs)
        
    def exception(self, msg, *args, **kwargs):
        return LoggerInjector.get_logger().exception(msg, *args, **kwargs)
        
    def log(self, level, msg, *args, **kwargs):
        return LoggerInjector.get_logger().log(level, msg, *args, **kwargs)
    
    # Additional logger methods
    def isEnabledFor(self, level):
        return LoggerInjector.get_logger().isEnabledFor(level)
    
    def getEffectiveLevel(self):
        return LoggerInjector.get_logger().getEffectiveLevel()
    
    def setLevel(self, level):
        return LoggerInjector.get_logger().setLevel(level)

# Create a single instance to be imported elsewhere
logger = Logger()

# Try attach context filter if available (for cross-project correlation IDs)
try:
    from common.logging.log_context import ContextFilter  # type: ignore
    logging.getLogger().addFilter(ContextFilter())
except Exception:
    pass
