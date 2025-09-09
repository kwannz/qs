from typing import List, Dict, Any
from pydantic import BaseModel
from panda_server.models.base_api_response import BaseAPIResponse
from common.backtest.model.backtest_user_strategy_log import BacktestUserStrategyLogModel

class QueryBacktestUserStrategyLogListResponseData(BaseModel):
    items: List[BacktestUserStrategyLogModel]
    cursor: Dict[str, Any]

class QueryBacktestUserStrategyLogListResponse(BaseAPIResponse[QueryBacktestUserStrategyLogListResponseData]):
    """User strategy log list query response"""
    pass 