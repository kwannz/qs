from bson import ObjectId
from pydantic import BaseModel, Field
from typing import Optional, Annotated
from datetime import datetime

class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, *_):
        from pydantic_core import core_schema

        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(ObjectId),
                    core_schema.chain_schema(
                        [
                            core_schema.str_schema(),
                            core_schema.no_info_plain_validator_function(cls.validate),
                        ]
                    ),
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x),
                return_schema=core_schema.str_schema(),
                when_used="json",
            ),
        )

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

class BacktestTradeBaseModel(BaseModel):
    """
    回测交易基础模型类
    用于表示回测中的交易信息（不含id）
    """
    type: Optional[int] = Field(default=None, description="交易类型")
    back_id: Optional[str] = Field(default=None, description="回测ID")
    now_system_order: Optional[int] = Field(default=None, description="当前系统订单")
    client_id: Optional[str] = Field(default=None, description="客户端ID")
    account_id: Optional[str] = Field(default=None, description="账户ID")
    contract_code: Optional[str] = Field(default=None, description="合约代码")
    contract_name: Optional[str] = Field(default=None, description="合约名称")
    trade_id: Optional[str] = Field(default=None, description="交易ID")
    order_id: Optional[str] = Field(default=None, description="订单ID")
    direction: Optional[int] = Field(default=None, description="交易方向")
    price: Optional[float] = Field(default=None, description="交易价格")
    business: Optional[int] = Field(default=None, description="业务类型")
    volume: Optional[int] = Field(default=None, description="交易数量")
    gmt_create: Optional[str] = Field(default=None, description="创建日期")
    trade_date: Optional[str] = Field(default=None, description="交易日期")
    gmt_create_time: Optional[str] = Field(default=None, description="创建时间")
    round_lot: Optional[int] = Field(default=None, description="手数")
    close_td_pos: Optional[int] = Field(default=None, description="平今仓位")
    order_remark: Optional[str] = Field(default=None, description="订单备注")
    margin: Optional[float] = Field(default=None, description="保证金")
    cost: Optional[float] = Field(default=None, description="成本")

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True
        from_attributes = True
        arbitrary_types_allowed = True
        json_schema_extra = {"description": "Backtest Trade Base Model"}

class BacktestTradeModel(BacktestTradeBaseModel):
    id: Annotated[
        PyObjectId,
        Field(
            default_factory=PyObjectId,
            alias="_id",
            description="Backtest Trade unique identifier, consistent with MongoDB's ObjectId",
        ),
    ] 