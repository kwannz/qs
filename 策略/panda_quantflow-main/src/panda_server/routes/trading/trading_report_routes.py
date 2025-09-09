from fastapi import APIRouter, HTTPException, status
from typing import List, Optional

from panda_server.logic.trading.real_trade_order_logic import (
    get_future_trade_record_by_id,
    create_future_trade_record,
    update_future_trade_record,
    delete_future_trade_record,
    get_all_future_trade_orders
)
from panda_server.models.base_api_response import BaseAPIResponse
from panda_trading.models.trading.trading_real_order import FutureTradeOrderModel

router = APIRouter(prefix="/api/trade_report", tags=["trade records"])


@router.get("/order", response_model=BaseAPIResponse[List[FutureTradeOrderModel]])
async def list_all_trade_records():
    try:
        records = await get_all_future_trade_orders()
        return BaseAPIResponse.success(data=records, message="获取记录成功")
    except Exception as e:
        return BaseAPIResponse.error(code=500, message=str(e))


@router.get("/order/{order_id}", response_model=BaseAPIResponse[Optional[FutureTradeOrderModel]])
async def get_trade_record(order_id: str):
    try:
        record = await get_future_trade_record_by_id(order_id)
        if not record:
            return BaseAPIResponse.error(code=404, message="记录未找到")
        return BaseAPIResponse.success(data=record, message="获取记录成功")
    except Exception as e:
        return BaseAPIResponse.error(code=500, message=str(e))


@router.post("/order", response_model=BaseAPIResponse[FutureTradeOrderModel], status_code=status.HTTP_201_CREATED)
async def create_trade_record(record: FutureTradeOrderModel):
    try:
        created_record = await create_future_trade_record(record)
        return BaseAPIResponse.success(data=created_record, message="创建记录成功")
    except Exception as e:
        return BaseAPIResponse.error(code=500, message=str(e))


@router.put("/order/{order_id}", response_model=BaseAPIResponse[FutureTradeOrderModel])
async def update_trade_record(order_id: str, update_data: FutureTradeOrderModel):
    try:
        updated_record = await update_future_trade_record(order_id, update_data.model_dump(exclude_unset=True))
        if not updated_record:
            return BaseAPIResponse.error(code=404, message="记录未找到或未更新")
        return BaseAPIResponse.success(data=updated_record, message="更新记录成功")
    except Exception as e:
        return BaseAPIResponse.error(code=500, message=str(e))


@router.delete("/order/{order_id}", response_model=BaseAPIResponse)
async def delete_trade_record(order_id: str):
    try:
        success = await delete_future_trade_record(order_id)
        if not success:
            return BaseAPIResponse.error(code=404, message="记录未找到")
        return BaseAPIResponse.success(message="记录已删除")
    except Exception as e:
        return BaseAPIResponse.error(code=500, message=str(e))