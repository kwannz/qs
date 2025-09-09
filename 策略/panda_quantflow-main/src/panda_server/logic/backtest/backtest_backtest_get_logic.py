import logging
from bson import ObjectId
from fastapi import HTTPException, status
from panda_server.config.database import mongodb
from common.backtest.model.backtest_backtest import BacktestModel, BacktestBaseModel
from panda_server.models.backtest.query_backtest_response import QueryBacktestResultResponseData, QueryBacktestBacktestResponse
from panda_server.models.base_api_response import BaseAPIResponse

logger = logging.getLogger(__name__)

# 定义 collection 名称
COLLECTION_NAME = "panda_back_test"

async def backtest_backtest_get_logic(
    back_id: str
) -> QueryBacktestBacktestResponse:
    """
    根据回测ID获取回测结果
    参数：
        back_id: 回测ID
    返回：
        dict: 回测结果原始响应
    """
    collection = mongodb.get_collection(COLLECTION_NAME)
    result = await collection.find_one({"_id": ObjectId(back_id)})
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest ID not found: {back_id}",
        )
    response_data = QueryBacktestResultResponseData.model_validate(result)
    return QueryBacktestBacktestResponse(data=response_data) 