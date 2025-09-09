import logging
from fastapi import HTTPException, status
from panda_server.config.database import mongodb
from common.backtest.model.backtest_trade import BacktestTradeModel
from panda_server.models.backtest.query_trade_response import QueryBacktestTradeListResponse, QueryBacktestTradeListResponseData

logger = logging.getLogger(__name__)

COLLECTION_NAME = "panda_backtest_trade"

async def backtest_trade_get_logic(
    back_id: str,
    page: int = 1,
    page_size: int = 10
) -> QueryBacktestTradeListResponse:
    """
    根据回测ID分页获取回测交易信息，并做模型校验，返回统一结构
    """
    collection = mongodb.get_collection(COLLECTION_NAME)
    skip = (page - 1) * page_size
    total_count = await collection.count_documents({"back_id": back_id})
    cursor = collection.find({"back_id": back_id}).skip(skip).limit(page_size)
    data_list = await cursor.to_list(length=None)
    validated_items = []
    for data in data_list:
        try:
            validated = BacktestTradeModel.model_validate(data)
            validated_items.append(validated)
        except Exception as e:
            logger.warning(f"Trade data validation failed: {e}, raw: {data}")
    pagination = {
        "total": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": (total_count + page_size - 1) // page_size
    }
    response_data = QueryBacktestTradeListResponseData(items=validated_items, pagination=pagination)
    return QueryBacktestTradeListResponse(data=response_data) 