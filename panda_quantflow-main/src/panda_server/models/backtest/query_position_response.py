from typing import List, Dict, Any
from pydantic import BaseModel
from panda_server.models.base_api_response import BaseAPIResponse
from common.backtest.model.backtest_position import BacktestPositionModel

class QueryBacktestPositionListResponseData(BaseModel):
    items: List[BacktestPositionModel]
    pagination: Dict[str, Any]

class QueryBacktestPositionListResponse(BaseAPIResponse[QueryBacktestPositionListResponseData]):
    """Backtest position list query response"""
    pass 