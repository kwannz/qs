"""
MongoDB索引管理工具
提供通用的索引创建和管理功能
"""
import logging

logger = logging.getLogger(__name__)

async def sync_collection_indexes(db_instance, collection_name: str, indexes_to_create: list, collection_display_name: str = None):
    """
    同步集合的索引配置，确保所需的索引都已创建
    
    Args:
        db_instance: MongoDB实例
        collection_name: 集合名称
        indexes_to_create: 需要创建的索引列表，每个索引是一个字典，包含name、keys和options
        collection_display_name: 用于日志显示的集合名称，如果为None则使用collection_name
    """
    try:
        # 获取集合
        collection = db_instance.get_collection(collection_name)
        display_name = collection_display_name or collection_name
        
        # 获取现有索引
        existing_indexes = set()
        try:
            indexes = await collection.list_indexes().to_list(length=None)
            existing_indexes = {index["name"] for index in indexes}
            logger.info(f"Existing {display_name} indexes: {existing_indexes}")
        except Exception as e:
            logger.warning(f"Failed to list existing indexes for {display_name}: {e}")
        
        # 创建缺失的索引
        created_count = 0
        for index_def in indexes_to_create:
            if index_def["name"] not in existing_indexes:
                try:
                    await collection.create_index(
                        index_def["keys"],
                        name=index_def["name"],
                        **index_def["options"]
                    )
                    logger.info(f"Created index: {index_def['name']}")
                    created_count += 1
                except Exception as e:
                    logger.error(f"Failed to create index {index_def['name']}: {e}")
            else:
                logger.debug(f"Index already exists: {index_def['name']}")
                
        if created_count > 0:
            logger.info(f"Created {created_count} new {display_name} indexes")
        else:
            logger.info(f"All {display_name} indexes already exist")
            
    except Exception as e:
        logger.error(f"Failed to initialize {collection_display_name} indexes: {e}") 