from typing import List, Dict, Any
from pydantic import BaseModel
from panda_server.models.base_api_response import BaseAPIResponse
from common.backtest.model.backtest_trade import BacktestTradeModel

class QueryBacktestTradeListResponseData(BaseModel):
    items: List[BacktestTradeModel]
    pagination: Dict[str, Any]

class QueryBacktestTradeListResponse(BaseAPIResponse[QueryBacktestTradeListResponseData]):
    """Backtest trade list query response"""
    pass 