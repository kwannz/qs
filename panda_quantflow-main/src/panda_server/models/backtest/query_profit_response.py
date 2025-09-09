from typing import List, Dict, Any
from pydantic import BaseModel
from panda_server.models.base_api_response import BaseAPIResponse
from common.backtest.model.backtest_profit import BacktestProfitModel

class QueryBacktestProfitListResponseData(BaseModel):
    items: List[BacktestProfitModel]
    pagination: Dict[str, Any]

class QueryBacktestProfitListResponse(BaseAPIResponse[QueryBacktestProfitListResponseData]):
    """Backtest profit list query response"""
    pass 