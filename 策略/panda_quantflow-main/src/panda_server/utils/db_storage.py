import cloudpickle
import logging
import bson
import gridfs

from typing import Any
from bson import Binary
from panda_server.config.database import  mongodb
from bson import ObjectId
import logging
from motor.motor_asyncio import AsyncIOMotorGridFSBucket

logger = logging.getLogger(__name__)

async def save_to_collection(collection_name: str, obj: Any, extra: dict = {}) -> str:
    """
    将对象 pickle 后作为 Binary 存储到指定集合 (有 16M 的大小限制)，返回新文档的 _id 字符串。
    """
    try:
        raw_bytes = cloudpickle.dumps(obj)
        extra["size"] = round(len(raw_bytes) / 1024, 0)  # KB
        doc = {
            "version": 1,
            "_class": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
            "binary": Binary(raw_bytes),
            "extra": extra,
        }
        result = await mongodb.get_collection(collection_name).insert_one(doc)
        logger.info(
            f"db save success, collction: {collection_name}, version: {doc['version']}, class: {doc['_class']}, extra: {doc['extra']}, id: {result.inserted_id}"
        )
    except Exception as e:
        logger.error(f"db save failed, collction: {collection_name}, error: {e}")
        return None
    return str(result.inserted_id)

async def get_from_collection(collection_name: str, obj_id: str) -> Any:
    collection = mongodb.get_collection(collection_name)
    obj = await collection.find_one({"_id": ObjectId(obj_id)})
    if not obj:
        raise Exception(f"No object found, id: {obj_id}")
    return cloudpickle.loads(obj["binary"])

# 使用 GridFS 存储大对象
async def save_to_gridfs(bucket_name: str, obj: Any, filename: str = None, extra: dict = {}) -> str:
    """
    将整个 Pydantic 对象 pickle 后通过 GridFS 存储到 MongoDB，支持超过 16MB 的大文件。
    """
    raw_bytes = cloudpickle.dumps(obj)
    fs = AsyncIOMotorGridFSBucket(mongodb.db, bucket_name=bucket_name)
    file_name = filename or f"{obj.__class__.__name__}.pkl"
    # 上传数据
    file_id = await fs.upload_from_stream(
        file_name,
        raw_bytes,
        metadata={"_class": f"{obj.__class__.__module__}.{obj.__class__.__name__}", **extra},
    )
    logger.info(f"GridFS save success, bucket: {bucket_name}, id: {file_id}")
    return str(file_id)


# 使用 GridFS 获取大对象
async def get_from_gridfs(bucket_name: str, file_id: str) -> tuple[Any, Exception | None]:
    """
    从 GridFS 按 file_id 获取并反序列化对象。
    """
    try:
        fs = AsyncIOMotorGridFSBucket(mongodb.db, bucket_name=bucket_name)
        oid = ObjectId(file_id)
        stream = await fs.open_download_stream(oid)
        raw_bytes = await stream.read()
        return cloudpickle.loads(raw_bytes), None
    except gridfs.errors.NoFile:
        logger.info(f"GridFS file not found: {file_id}")
        return None, gridfs.errors.NoFile
    except bson.errors.InvalidId:
        logger.warning(f"Invalid GridFS file ID format: {file_id}")
        return None, bson.errors.InvalidId
    except Exception as e:
        logger.error(f"Unexpected error when getting GridFS file: {str(e)}")
        return None, e

# 获取 GridFS 文件的 metadata
async def get_gridfs_metadata(bucket_name: str, file_id: str) -> dict | None:
    """
    从 GridFS 按 file_id 获取文件的 metadata。

    Args:
        bucket_name: GridFS bucket名称
        file_id: 文件ID

    Returns:
        dict: 文件的metadata字典，如果文件不存在返回 None, 如果文件存在但没有metadata则返回 {}
    """
    try:
        fs = AsyncIOMotorGridFSBucket(mongodb.db, bucket_name=bucket_name)
        oid = ObjectId(file_id)
        grid_out = await fs.open_download_stream(oid)
        return grid_out.metadata or {}
    except gridfs.errors.NoFile:
        logger.info(f"GridFS file not found: {file_id}")
        return None
    except bson.errors.InvalidId:
        logger.warning(f"Invalid GridFS file ID format: {file_id}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error when getting GridFS metadata: {str(e)}")
        return None

if __name__ == "__main__":
    from panda_plugins.base.base_work_node import BaseWorkNode
    import asyncio
    from pydantic import BaseModel
    import pandas as pd

    class TestModel(BaseModel):
        df: pd.DataFrame
        other: int

    async def main():
        await mongodb.connect_db()

        df = pd.DataFrame({'x': [1, 2, 3], 'y': ['a', 'b', 'c']})
        test_obj = TestModel(df=df, other=1)
        collection = "workflow_node_output"
        
        # # test save
        obj_id_str = await save_to_collection(collection, test_obj, extra={'test': 'collection'})
        print(f"Saved object ID: {obj_id_str}")

        # test read
        retrieved_obj = await get_from_collection(collection, obj_id_str)
        print("Retrieved object:", retrieved_obj)
        print("Type:", type(retrieved_obj))
        print("Nested DataFrame:")
        print(retrieved_obj.df)
        print("Other:", retrieved_obj.other)

        # test GridFS save
        bucket = f"{collection}_fs"
        fs_file_id = await save_to_gridfs(bucket, test_obj, filename="TestModel.pkl", extra={'test':'gridfs'})
        print(f"GridFS Saved file ID: {fs_file_id}")

        # test GridFS metadata
        metadata = await get_gridfs_metadata(bucket, fs_file_id)
        print("GridFS Metadata:", metadata)

        # test GridFS read
        retrieved_fs_obj, error = await get_from_gridfs(bucket, fs_file_id)
        if error:
            print("GridFS read error:", error)
            return
        print("GridFS Retrieved object:", retrieved_fs_obj)
        print("Type:", type(retrieved_fs_obj))
        print("Nested DataFrame:")
        print(retrieved_fs_obj.df)
        print("Other:", retrieved_fs_obj.other)

        await mongodb.close_db()

    asyncio.run(main())
    