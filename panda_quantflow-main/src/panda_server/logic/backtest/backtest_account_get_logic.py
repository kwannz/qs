import logging
from bson import ObjectId
from fastapi import HTTPException, status
from panda_server.config.database import mongodb
from common.backtest.model.backtest_account import BacktestAccountModel
from panda_server.models.backtest.query_account_response import QueryBacktestAccountListResponse, QueryBacktestAccountListResponseData

logger = logging.getLogger(__name__)

# 定义 collection 名称
COLLECTION_NAME = "panda_backtest_account"

async def backtest_account_get_logic(
    back_id: str,
    page: int = 1,
    page_size: int = 10
) -> QueryBacktestAccountListResponse:
    """
    根据回测ID分页获取回测账户信息，并做模型校验，返回统一结构
    """
    collection = mongodb.get_collection(COLLECTION_NAME)
    skip = (page - 1) * page_size
    total_count = await collection.count_documents({"back_id": back_id})
    cursor = collection.find({"back_id": back_id}).skip(skip).limit(page_size)
    data_list = await cursor.to_list(length=None)
    validated_items = []
    for data in data_list:
        try:
            validated = BacktestAccountModel.model_validate(data)
            validated_items.append(validated)
        except Exception as e:
            logger.warning(f"Account data validation failed: {e}, raw: {data}")
            # 失败时可选：跳过或原样返回dict
            # validated_items.append(data)
    pagination = {
        "total": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": (total_count + page_size - 1) // page_size
    }
    response_data = QueryBacktestAccountListResponseData(items=validated_items, pagination=pagination)
    return QueryBacktestAccountListResponse(data=response_data) 