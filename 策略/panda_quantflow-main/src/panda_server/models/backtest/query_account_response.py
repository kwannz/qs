from typing import List, Dict, Any
from pydantic import BaseModel
from panda_server.models.base_api_response import BaseAPIResponse
from common.backtest.model.backtest_account import BacktestAccountModel

class QueryBacktestAccountListResponseData(BaseModel):
    items: List[BacktestAccountModel]
    pagination: Dict[str, Any]

class QueryBacktestAccountListResponse(BaseAPIResponse[QueryBacktestAccountListResponseData]):
    """Backtest account list query response"""
    pass 