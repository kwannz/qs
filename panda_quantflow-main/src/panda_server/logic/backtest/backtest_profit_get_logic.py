import logging
from fastapi import HTTPException, status
from panda_server.config.database import mongodb
from common.backtest.model.backtest_profit import BacktestProfitModel
from panda_server.models.backtest.query_profit_response import QueryBacktestProfitListResponse, QueryBacktestProfitListResponseData

logger = logging.getLogger(__name__)

COLLECTION_NAME = "panda_backtest_profit"

async def backtest_profit_get_logic(
    back_id: str,
    page: int = 1,
    page_size: int = 10
) -> QueryBacktestProfitListResponse:
    """
    根据回测ID分页获取回测收益信息，并做模型校验，返回统一结构
    """
    collection = mongodb.get_collection(COLLECTION_NAME)
    skip = (page - 1) * page_size
    total_count = await collection.count_documents({"back_id": back_id})
    cursor = collection.find({"back_id": back_id}).skip(skip).limit(page_size)
    data_list = await cursor.to_list(length=None)
    validated_items = []
    for data in data_list:
        try:
            validated = BacktestProfitModel.model_validate(data)
            validated_items.append(validated)
        except Exception as e:
            logger.warning(f"Profit data validation failed: {e}, raw: {data}")
    pagination = {
        "total": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": (total_count + page_size - 1) // page_size
    }
    response_data = QueryBacktestProfitListResponseData(items=validated_items, pagination=pagination)
    return QueryBacktestProfitListResponse(data=response_data) 