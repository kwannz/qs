from datetime import datetime
import logging
from typing import Optional, Any
from pydantic import BaseModel, Field
from bson import ObjectId
from panda_server.utils.time_utils import get_beijing_time

# 添加 logger
logger = logging.getLogger(__name__)

class UserLog(BaseModel):
    """
    通用用户日志数据模型
    
    字段说明：
    - id: MongoDB 的 ObjectId，会自动生成
    - user_id: 用户ID
    - timestamp: 日志产生的时间戳
    - level: 日志级别（INFO, WARNING, ERROR 等）
    - message: 日志消息内容
    - type: 日志类型，默认为user_action
    - workflow_run_id: 工作流运行ID（可选）
    - work_node_id: 工作节点ID（可选）
    - sequence: 日志在同一workflow中的序列号，用于排序和分页
    - workflow_id: 工作流ID
    """
    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    user_id: str
    timestamp: datetime = Field(default_factory=get_beijing_time)
    level: str = "INFO"
    message: str
    type: str = "user_action"
    workflow_run_id: Optional[str] = None
    work_node_id: Optional[str] = None
    sequence: int = Field(default=0, description="日志在同一workflow中的序列号，从1开始自增，0表示待自动分配")
    workflow_id: Optional[str] = None
    error_detail: Optional[Any] = None
    req_id: Optional[str] = Field(default=None, description="HTTP 请求ID（如果可用）")
    backtest_id: Optional[str] = Field(default=None, description="回测ID（如果适用）")

    class Config:
        populate_by_name = True
        json_encoders = {
            ObjectId: str
        }

# 使用示例：
"""
# 创建用户日志示例
from common.logging.user_log_model import UserLog

user_log = UserLog(
    user_id="user123",
    message="工作流节点执行完成", 
    level="INFO",
    type="workflow_run",
    workflow_run_id="workflow456",
    work_node_id="node789",
    workflow_id="workflow_id_123"
)
""" 
