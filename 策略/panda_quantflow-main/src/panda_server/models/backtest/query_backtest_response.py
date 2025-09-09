from common.backtest.model.backtest_backtest import BacktestBaseModel, PyObjectId
from typing import Optional
from pydantic import Field
from panda_server.models.base_api_response import BaseAPIResponse

class QueryBacktestResultResponseData(BacktestBaseModel):
    id: Optional[PyObjectId] = Field(None, alias="_id", description="Backtest ID")

class QueryBacktestBacktestResponse(BaseAPIResponse[QueryBacktestResultResponseData]):
    """Backtest query response"""
    pass 