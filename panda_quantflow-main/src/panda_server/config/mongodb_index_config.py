"""
MongoDB索引配置文件
定义所有集合的索引结构
"""
import logging
from pymongo import ASCENDING
from panda_server.config.mongodb_index_manager import sync_collection_indexes

logger = logging.getLogger(__name__)

# 工作流日志索引定义
WORKFLOW_LOGS_INDEXES = [
    {
        "name": "workflow_logs_by_user_workflow_time_asc_idx",
        "keys": [("user_id", 1), ("workflow_run_id", 1), ("_id", 1)],
        "options": {}
    },
    {
        "name": "workflow_logs_by_user_workflow_node_time_asc_idx", 
        "keys": [("user_id", 1), ("workflow_run_id", 1), ("work_node_id", 1), ("_id", 1)],
        "options": {}
    },
    {
        "name": "workflow_logs_by_user_workflow_level_time_asc_idx",
        "keys": [("user_id", 1), ("workflow_run_id", 1), ("level", 1), ("_id", 1)],
        "options": {}
    },
    {
        "name": "workflow_logs_by_user_workflow_sequence_asc_idx",
        "keys": [("user_id", 1), ("workflow_run_id", 1), ("sequence", 1)],
        "options": {}
    },
    {
        "name": "workflow_logs_by_workflow_sequence_asc_idx",
        "keys": [("workflow_run_id", 1), ("sequence", 1)],
        "options": {}
    },
    {
        "name": "workflow_logs_by_user_sequence_asc_idx",
        "keys": [("user_id", 1), ("sequence", 1)],
        "options": {}
    }
]

# 工作流序列计数器索引定义
WORKFLOW_COUNTERS_INDEXES = [
    {
        "name": "workflow_sequence_counters_by_workflow_unique_idx",
        "keys": [("workflow_run_id", ASCENDING)],
        "options": {"unique": True}
    }
]

async def init_workflow_logs_indexes(db_instance):
    """初始化工作流日志相关的所有索引"""
    await sync_collection_indexes(
        db_instance=db_instance,
        collection_name="workflow_logs",
        indexes_to_create=WORKFLOW_LOGS_INDEXES,
        collection_display_name="workflow logs"
    )

async def init_workflow_counters_indexes(db_instance):
    """初始化工作流序列计数器相关的索引"""
    # 再走原有的索引同步逻辑（如果有其他索引定义也能同步）
    await sync_collection_indexes(
        db_instance=db_instance,
        collection_name="workflow_sequence_counters",
        indexes_to_create=WORKFLOW_COUNTERS_INDEXES,
        collection_display_name="workflow sequence counters"
    )

async def init_all_indexes(db_instance):
    """初始化所有数据库索引"""
    await init_workflow_logs_indexes(db_instance)
    await init_workflow_counters_indexes(db_instance)