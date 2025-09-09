from abc import ABC, abstractmethod
import logging
import queue
import time
from pydantic import BaseModel
from typing import Type, Optional
import panda_plugins.base.jsonschema_patches  # 必须保留
from common.logging.user_logger import UserLogger


class BaseWorkNode(ABC):
    """
    Base Work Node
    Base class to be inherited when developing panda_plugins
    """

    # internal class attributes
    __work_node_name__: str
    __work_node_display_name__: str
    __work_node_group__: str
    __work_node_order__: int
    __work_node_type__: str
    __short_description__: str = ""  # html rich text
    __long_description__: str = ""  # html rich text

    def __init__(self):
        # 日志记录器，由工作流执行器设置
        self._user_logger = None
        self._workflow_id = None  # 存储workflow_id
        self._sys_logger = logging.getLogger(self.__class__.__name__)
        self._log_queue = queue.Queue()  # 用于缓存日志消息

        # 创建标准 logger 接口
        self.logger = self.LoggerWrapper(self)

    class LoggerWrapper:
        """
        Logger wrapper class that provides standard logger interface
        """

        def __init__(self, work_node):
            self._work_node = work_node

        def debug(self, message: str, **kwargs):
            """记录调试级别日志"""
            self._work_node.log_debug(message, **kwargs)

        def info(self, message: str, **kwargs):
            """记录信息级别日志"""
            self._work_node.log_info(message, **kwargs)

        def warning(self, message: str, **kwargs):
            """记录警告级别日志"""
            self._work_node.log_warning(message, **kwargs)

        def warn(self, message: str, **kwargs):
            """记录警告级别日志 (别名)"""
            self._work_node.log_warning(message, **kwargs)

        def error(self, message: str, **kwargs):
            """记录错误级别日志"""
            self._work_node.log_error(message, **kwargs)

        def critical(self, message: str, **kwargs):
            """记录严重错误级别日志"""
            self._work_node.log_critical(message, **kwargs)

        def fatal(self, message: str, **kwargs):
            """记录严重错误级别日志 (别名)"""
            self._work_node.log_critical(message, **kwargs)

    def _setup_logging_context(
        self,
        user_id: str,
        workflow_run_id: str,
        work_node_id: str,
        workflow_id: str = None,
    ):
        """
        设置日志上下文，由工作流执行器调用

        Args:
            user_id: 用户ID
            workflow_run_id: 工作流运行ID
            work_node_id: 工作节点ID
            workflow_id: 工作流ID
        """
        try:
            self._user_logger = UserLogger(
                user_id=user_id,
                workflow_run_id=workflow_run_id,
                work_node_id=work_node_id,
            )
            self._workflow_id = workflow_id  # 存储workflow_id以备后用
        except Exception as e:
            self._sys_logger.warning(f"Failed to setup logging context: {e}")

    def _queue_log(self, level: str, message: str, **kwargs):
        """
        将日志消息放入队列，避免在同步上下文中直接调用异步方法
        """
        if self._user_logger:
            # 将日志信息放入队列
            log_entry = {
                "level": level,
                "message": message,
                "workflow_id": self._workflow_id,
                "timestamp": time.time(),
                "kwargs": kwargs,
            }
            try:
                self._log_queue.put_nowait(log_entry)
            except queue.Full:
                self._sys_logger.warning("Log queue is full, dropping log message")
        else:
            # 如果没有用户日志记录器，使用系统日志
            getattr(self._sys_logger, level.lower(), self._sys_logger.info)(
                f"[USER_LOG] {message} (metadata: {kwargs})"
            )

    async def _process_queued_logs(self):
        """
        处理队列中的日志消息（异步方法，由工作流执行器调用）
        """
        if not self._user_logger:
            return

        while not self._log_queue.empty():
            try:
                log_entry = self._log_queue.get_nowait()
                level = log_entry["level"]
                message = log_entry["message"]
                workflow_id = log_entry["workflow_id"]
                kwargs = log_entry["kwargs"]

                # 调用对应的异步日志方法
                if hasattr(self._user_logger, level.lower()):
                    log_method = getattr(self._user_logger, level.lower())
                    await log_method(message, workflow_id=workflow_id, **kwargs)

            except queue.Empty:
                break
            except Exception as e:
                self._sys_logger.warning(f"Failed to process queued log: {e}")

    def log_debug(self, message: str, **kwargs):
        """
        记录调试级别日志

        Args:
            message: 日志消息
            **kwargs: 额外的元数据
        """
        self._queue_log("DEBUG", message, **kwargs)

    def log_info(self, message: str, **kwargs):
        """
        记录信息级别日志

        Args:
            message: 日志消息
            **kwargs: 额外的元数据
        """
        self._queue_log("INFO", message, **kwargs)

    def log_warning(self, message: str, **kwargs):
        """
        记录警告级别日志

        Args:
            message: 日志消息
            **kwargs: 额外的元数据
        """
        self._queue_log("WARNING", message, **kwargs)

    def log_error(self, message: str, **kwargs):
        """
        记录错误级别日志

        Args:
            message: 日志消息
            **kwargs: 额外的元数据
        """
        self._queue_log("ERROR", message, **kwargs)

    def log_critical(self, message: str, **kwargs):
        """
        记录严重错误级别日志

        Args:
            message: 日志消息
            **kwargs: 额外的元数据
        """
        self._queue_log("CRITICAL", message, **kwargs)

    @classmethod
    @abstractmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        """
        Returns the Pydantic model class that defines the input.
        IMPORTANT: Plugin developers must implement this method.
        """
        pass

    @classmethod
    @abstractmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        """
        Returns the Pydantic model class that defines the output.
        IMPORTANT: Plugin developers must implement this method.
        """
        pass

    @abstractmethod
    def run(self, input: BaseModel) -> Optional[BaseModel]:
        """
        Executes node logic, receives input model as parameter and returns output model.
        IMPORTANT: Plugin developers must implement this method.

        Example usage in plugin:
            def run(self, input: InputModel) -> OutputModel:
                self.log_info("开始处理数据", input_size=len(input.data))
                try:
                    # 处理逻辑
                    result = process_data(input.data)
                    self.log_info("数据处理完成", output_size=len(result))
                    return OutputModel(data=result)
                except Exception as e:
                    self.log_error("数据处理失败", error=str(e))
                    raise
        """
        pass

    @classmethod
    def set_short_description(cls, html: str):
        cls.__short_description__ = html.strip()

    @classmethod
    def set_long_description(cls, html: str):
        cls.__long_description__ = html.strip()
