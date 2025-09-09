"""
工作流序列计数器MongoDB索引设置脚本
包含：workflow_sequence_counters集合的索引
用于生产环境的增量索引变更
"""
import asyncio
import logging
from pymongo import ASCENDING
from panda_server.migrations.v1_to_v1_1.index_common_manager import (
    DatabaseManager,
    create_collection_indexes, 
    drop_collection_indexes
)

logger = logging.getLogger(__name__)

# 需要新增的索引列表
NEW_WORKFLOW_COUNTERS_INDEXES = [
    {
        # 工作流运行ID唯一索引：确保每个工作流只有一个计数器
        "name": "workflow_sequence_counters_by_workflow_unique_idx",
        "keys": [("workflow_run_id", ASCENDING)],
        "options": {"unique": True}
    }
]

# 需要删除的旧索引列表
DEPRECATED_WORKFLOW_COUNTERS_INDEXES = []

async def create_workflow_counters_indexes(db):
    """创建工作流序列计数器相关的新索引"""
    try:
        # 获取集合
        counter_collection = db.get_collection("workflow_sequence_counters")
        
        return await create_collection_indexes(
            collection=counter_collection,
            indexes_to_create=NEW_WORKFLOW_COUNTERS_INDEXES,
            collection_display_name="workflow sequence counters"
        )
    except Exception as e:
        logger.error(f"Failed to create workflow counters indexes: {e}")
        return False

async def drop_workflow_counters_indexes(db):
    """删除指定的旧索引"""
    try:
        counter_collection = db.get_collection("workflow_sequence_counters")
        return await drop_collection_indexes(
            collection=counter_collection,
            indexes_to_drop=DEPRECATED_WORKFLOW_COUNTERS_INDEXES,
            collection_display_name="workflow sequence counters"
        )
    except Exception as e:
        logger.error(f"Failed to drop workflow counters indexes: {e}")
        return False

async def setup_workflow_counters_collection(mode: str = 'both'):
    """设置工作流序列计数器集合的索引
    
    Args:
        mode: 执行模式
            - 'both': 先创建新索引，后删除旧索引（默认）
            - 'create': 只创建新索引
            - 'drop': 只删除旧索引
    """
    logger.info(f"Setting up workflow sequence counters collection indexes in {mode} mode...")
    
    async with DatabaseManager() as db:
        if mode in ('both', 'create'):
            # 创建新索引
            create_success = await create_workflow_counters_indexes(db)
            if not create_success:
                logger.error("Failed to create new indexes, aborting operation")
                return False
            logger.info("Successfully created new indexes")
            
        if mode in ('both', 'drop'):
            # 删除旧索引
            drop_success = await drop_workflow_counters_indexes(db)
            if not drop_success:
                logger.error("Failed to drop old indexes")
                return False
            logger.info("Successfully dropped old indexes")
        
        logger.info("Workflow sequence counters collection setup completed successfully")
        return True

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 执行索引设置
    success = asyncio.run(setup_workflow_counters_collection('both'))
    if not success:
        logger.error("Workflow sequence counters indexes setup failed")
        exit(1)
    logger.info("Workflow sequence counters indexes setup completed") 