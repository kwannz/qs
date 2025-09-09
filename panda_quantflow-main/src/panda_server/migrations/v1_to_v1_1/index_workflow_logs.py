"""
工作流日志MongoDB索引设置脚本
包含：workflow_logs集合的所有索引
用于生产环境的增量索引变更
"""
import asyncio
import logging
from panda_server.migrations.v1_to_v1_1.index_common_manager import (
    DatabaseManager,
    create_collection_indexes, 
    drop_collection_indexes
)

logger = logging.getLogger(__name__)

# 需要新增的索引列表
NEW_WORKFLOW_LOGS_INDEXES = [
    {
        # 用户工作流时间索引：用于按用户和工作流运行ID查询日志，按时间(_id)排序
        "name": "workflow_logs_by_user_workflow_time_asc_idx",
        "keys": [("user_id", 1), ("workflow_run_id", 1), ("_id", 1)],
        "options": {}
    },
    {
        # 用户工作流节点时间索引：用于按用户、工作流运行ID和节点ID查询特定节点的日志
        "name": "workflow_logs_by_user_workflow_node_time_asc_idx", 
        "keys": [("user_id", 1), ("workflow_run_id", 1), ("work_node_id", 1), ("_id", 1)],
        "options": {}
    },
    {
        # 用户工作流日志级别索引：用于按用户、工作流运行ID和日志级别查询日志
        "name": "workflow_logs_by_user_workflow_level_time_asc_idx",
        "keys": [("user_id", 1), ("workflow_run_id", 1), ("level", 1), ("_id", 1)],
        "options": {}
    },
    {
        # 用户工作流序列索引：用于按用户、工作流运行ID和序列号查询和排序日志
        "name": "workflow_logs_by_user_workflow_sequence_asc_idx",
        "keys": [("user_id", 1), ("workflow_run_id", 1), ("sequence", 1)],
        "options": {}
    },
    {
        # 工作流序列索引：用于按workflow分页查询
        "name": "workflow_logs_by_workflow_sequence_asc_idx",
        "keys": [("workflow_run_id", 1), ("sequence", 1)],
        "options": {}
    },
    {
        # 用户序列索引：用于用户全局日志查询
        "name": "workflow_logs_by_user_sequence_asc_idx",
        "keys": [("user_id", 1), ("sequence", 1)],
        "options": {}
    }
]

# 需要删除的旧索引列表
DEPRECATED_WORKFLOW_LOGS_INDEXES = []

async def create_workflow_logs_indexes(db):
    """创建工作流日志相关的新索引"""
    try:
        # 获取集合
        logs_collection = db.get_collection("workflow_logs")
        return await create_collection_indexes(
            collection=logs_collection,
            indexes_to_create=NEW_WORKFLOW_LOGS_INDEXES,
            collection_display_name="workflow logs"
        )
    except Exception as e:
        logger.error(f"Failed to create workflow logs indexes: {e}")
        return False

async def drop_workflow_logs_indexes(db):
    """删除指定的旧索引"""
    try:
        logs_collection = db.get_collection("workflow_logs")
        return await drop_collection_indexes(
            collection=logs_collection,
            indexes_to_drop=DEPRECATED_WORKFLOW_LOGS_INDEXES,
            collection_display_name="workflow logs"
        )
    except Exception as e:
        logger.error(f"Failed to drop workflow logs indexes: {e}")
        return False

async def setup_workflow_logs_collection(mode: str = 'both'):
    """设置工作流日志集合的索引
    
    Args:
        mode: 执行模式
            - 'both': 先创建新索引，后删除旧索引（默认）
            - 'create': 只创建新索引
            - 'drop': 只删除旧索引
    """
    logger.info(f"Setting up workflow logs collection indexes in {mode} mode...")
    
    async with DatabaseManager() as db:
        if mode in ('both', 'create'):
            # 创建新索引
            create_success = await create_workflow_logs_indexes(db)
            if not create_success:
                logger.error("Failed to create new indexes, aborting operation")
                return False
            logger.info("Successfully created new indexes")
            
        if mode in ('both', 'drop'):
            # 删除旧索引
            drop_success = await drop_workflow_logs_indexes(db)
            if not drop_success:
                logger.error("Failed to drop old indexes")
                return False
            logger.info("Successfully dropped old indexes")
        
        logger.info("Workflow logs collection setup completed successfully")
        return True

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 执行索引设置
    success = asyncio.run(setup_workflow_logs_collection('both'))
    if not success:
        logger.error("Workflow logs indexes setup failed")
        exit(1)
    logger.info("Workflow logs indexes setup completed") 