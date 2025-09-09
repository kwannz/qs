import logging

from panda_server.models.base_api_response import BaseAPIResponse

logger = logging.getLogger(__name__)


# 创建路由实例，设置前缀和标签
from fastapi import APIRouter
router = APIRouter(tags=["base"])

# 定义 collection 名称
COLLECTION_NAME = "base"

@router.get("/", response_model=BaseAPIResponse)
async def root():
    """Root path"""
    logger.info("Received root path request")
    return BaseAPIResponse(
        data={
            "status": "ok",
            "message": "PandaAI QuantFlow API is running",
        }
    )