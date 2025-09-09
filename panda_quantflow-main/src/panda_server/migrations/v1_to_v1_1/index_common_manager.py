"""
迁移脚本使用的MongoDB索引管理工具
提供迁移过程中的索引创建和管理功能
专门用于生产环境的增量索引变更
"""
import logging
import asyncio
import yaml
from typing import List, Dict, Any, Tuple
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import OperationFailure
from panda_server.config.env import (
    MONGODB_URL,
    DATABASE_NAME,
    MONGO_USER,
    MONGO_PASSWORD,
    MONGO_AUTH_DB,
    MONGO_TYPE,
    MONGO_REPLICA_SET
)

logger = logging.getLogger(__name__)

# 生产环境配置（暂时注释掉）
"""
with open('src/common/config/config_pro.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
    MONGO_USER = config['MONGO_USER']
    MONGO_PASSWORD = config['MONGO_PASSWORD']
    MONGO_URI = config['MONGO_URI']
    MONGO_AUTH_DB = config['MONGO_AUTH_DB']
    MONGO_DB = config['MONGO_DB']
    MONGO_REPLICA_SET = config['MONGO_REPLICA_SET']

# 构建MongoDB连接URI
MONGODB_URI = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_URI}/{MONGO_DB}?replicaSet={MONGO_REPLICA_SET}&authSource={MONGO_AUTH_DB}"
"""

# 本地测试环境配置
def get_mongodb_uri():
    """构建MongoDB连接URI"""
    if MONGO_USER and MONGO_PASSWORD:
        auth_str = f"{MONGO_USER}:{MONGO_PASSWORD}@"
        base_uri = f"mongodb://{auth_str}{MONGODB_URL}/{DATABASE_NAME}"
    else:
        base_uri = f"{MONGODB_URL}/{DATABASE_NAME}"

    if MONGO_TYPE == 'replica_set' and MONGO_REPLICA_SET:
        base_uri += f"?replicaSet={MONGO_REPLICA_SET}"
        if MONGO_AUTH_DB:
            base_uri += f"&authSource={MONGO_AUTH_DB}"
    elif MONGO_AUTH_DB:
        base_uri += f"?authSource={MONGO_AUTH_DB}"

    return base_uri

class DatabaseManager:
    """数据库连接管理器"""
    def __init__(self):
        self.client = None
        self.db = None

    async def __aenter__(self):
        """创建数据库连接"""
        try:
            mongodb_uri = get_mongodb_uri()
            self.client = AsyncIOMotorClient(mongodb_uri)
            self.db = self.client[DATABASE_NAME]
            
            # 验证连接
            await self.db.command('ping')
            logger.info("Successfully connected to MongoDB")
            
            return self.db
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """关闭数据库连接"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

async def check_existing_indexes(collection, indexes_to_check: List[Dict[str, Any]], collection_display_name: str) -> Tuple[bool, List[str]]:
    """检查索引是否已存在
    
    Args:
        collection: MongoDB集合对象
        indexes_to_check: 要检查的索引列表
        collection_display_name: 集合显示名称（用于日志）
    
    Returns:
        Tuple[bool, List[str]]: (是否有索引已存在, 已存在的索引名称列表)
    """
    try:
        existing_indexes = await collection.list_indexes().to_list(None)
        existing_index_names = {idx.get('name') for idx in existing_indexes}
        
        existing_indexes = []
        for index in indexes_to_check:
            if index["name"] in existing_index_names:
                existing_indexes.append(index["name"])
        
        return bool(existing_indexes), existing_indexes
    except Exception as e:
        logger.error(f"Failed to check existing indexes for {collection_display_name}: {e}")
        raise

async def create_collection_indexes(collection, indexes_to_create: List[Dict[str, Any]], collection_display_name: str, session=None) -> bool:
    """创建集合索引（事务型，支持session）
    
    Args:
        collection: MongoDB集合对象
        indexes_to_create: 要创建的索引列表
        collection_display_name: 集合显示名称（用于日志）
        session: MongoDB会话对象（可选）
    
    Returns:
        bool: 是否成功创建所有索引
    """
    has_existing, existing_indexes = await check_existing_indexes(collection, indexes_to_create, collection_display_name)
    if has_existing:
        logger.error(f"Following indexes already exist in {collection_display_name}: {', '.join(existing_indexes)}")
        return False

    from motor.motor_asyncio import AsyncIOMotorClient
    client: AsyncIOMotorClient = collection.database.client
    own_session = False
    if session is None:
        session = await client.start_session()
        own_session = True

    try:
        async with session:
            async with session.start_transaction():
                for index in indexes_to_create:
                    await collection.create_index(
                        index["keys"],
                        name=index["name"],
                        session=session,
                        **index.get("options", {})
                    )
                    logger.info(f"Created index {index['name']} for {collection_display_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to create indexes for {collection_display_name}: {e}")
        return False
    finally:
        if own_session:
            await session.end_session()

async def verify_indexes_exist(collection, expected_indexes: List[Dict[str, Any]], collection_display_name: str) -> bool:
    """验证索引是否存在且可用
    
    Args:
        collection: MongoDB集合对象
        expected_indexes: 期望存在的索引列表
        collection_display_name: 集合显示名称（用于日志）
    
    Returns:
        bool: 是否所有期望的索引都存在且可用
    """
    try:
        existing_indexes = await collection.list_indexes().to_list(None)
        existing_index_names = {idx.get('name') for idx in existing_indexes}
        
        missing_indexes = []
        for index in expected_indexes:
            if index["name"] not in existing_index_names:
                missing_indexes.append(index["name"])
                logger.error(f"Index {index['name']} not found in {collection_display_name}")
            else:
                # 验证索引是否可用
                try:
                    await collection.find().hint(index["name"]).limit(1).explain()
                    logger.info(f"Verified index {index['name']} exists and is usable in {collection_display_name}")
                except OperationFailure as e:
                    missing_indexes.append(index["name"])
                    logger.error(f"Index {index['name']} exists but is not usable: {e}")
        
        return not bool(missing_indexes)
    except Exception as e:
        logger.error(f"Failed to verify indexes for {collection_display_name}: {e}")
        return False

async def drop_collection_indexes(collection, indexes_to_drop: List[Dict[str, Any]], collection_display_name: str, session=None) -> bool:
    """删除集合索引（事务型，支持session）
    
    Args:
        collection: MongoDB集合对象
        indexes_to_drop: 要删除的索引列表
        collection_display_name: 集合显示名称（用于日志）
        session: MongoDB会话对象（可选）
    
    Returns:
        bool: 是否成功删除所有指定的索引
    """
    if not await verify_indexes_exist(collection, indexes_to_drop, collection_display_name):
        logger.error(f"Some indexes to be dropped from {collection_display_name} do not exist or are not usable")
        return False

    from motor.motor_asyncio import AsyncIOMotorClient
    client: AsyncIOMotorClient = collection.database.client
    own_session = False
    if session is None:
        session = await client.start_session()
        own_session = True

    try:
        async with session:
            async with session.start_transaction():
                for index in indexes_to_drop:
                    await collection.drop_index(index["name"], session=session)
                    logger.info(f"Dropped index {index['name']} from {collection_display_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to drop indexes for {collection_display_name}: {e}")
        return False
    finally:
        if own_session:
            await session.end_session()
