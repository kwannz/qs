import logging
from fastapi import HTTPException, status
from panda_server.config.database import mongodb
from common.backtest.model.backtest_position import BacktestPositionModel
from panda_server.models.backtest.query_position_response import QueryBacktestPositionListResponse, QueryBacktestPositionListResponseData
from typing import Optional

logger = logging.getLogger(__name__)

COLLECTION_NAME = "panda_backtest_position"

async def backtest_position_get_logic(
    back_id: str,
    page: int = 1,
    page_size: int = 10,
    date: Optional[str] = None
) -> QueryBacktestPositionListResponse:
    """
    根据回测ID分页获取回测持仓信息，并做模型校验，返回统一结构
    支持可选日期过滤
    """
    collection = mongodb.get_collection(COLLECTION_NAME)
    query = {"back_id": back_id}
    if date:
        try:
            formatted_date = date.replace("-", "")
            query["gmt_create"] = formatted_date
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format, please use YYYY-MM-DD format",
            )
    skip = (page - 1) * page_size
    total_count = await collection.count_documents(query)
    cursor = collection.find(query).skip(skip).limit(page_size)
    data_list = await cursor.to_list(length=None)
    validated_items = []
    for data in data_list:
        try:
            validated = BacktestPositionModel.model_validate(data)
            validated_items.append(validated)
        except Exception as e:
            logger.warning(f"Position data validation failed: {e}, raw: {data}")
    pagination = {
        "total": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": (total_count + page_size - 1) // page_size
    }
    response_data = QueryBacktestPositionListResponseData(items=validated_items, pagination=pagination)
    return QueryBacktestPositionListResponse(data=response_data) 