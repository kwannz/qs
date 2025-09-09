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

class BacktestPositionBaseModel(BaseModel):
    """
    回测持仓基础模型类
    用于表示回测中的持仓信息（不含id）
    """
    back_id: Optional[str] = Field(default=None, description="回测ID")
    account_id: Optional[str] = Field(default=None, description="账户ID")
    contract_code: Optional[str] = Field(default=None, description="合约代码")
    contract_name: Optional[str] = Field(default=None, description="合约名称")
    type: Optional[int] = Field(default=None, description="类型")
    direction: Optional[int] = Field(default=None, description="方向")
    price: Optional[float] = Field(default=None, description="价格")
    settlement: Optional[float] = Field(default=None, description="结算价")
    hold_price: Optional[float] = Field(default=None, description="持仓价")
    position: Optional[int] = Field(default=None, description="持仓量")
    td_position: Optional[int] = Field(default=None, description="今日持仓")
    cost: Optional[float] = Field(default=None, description="成本")
    margin: Optional[float] = Field(default=None, description="保证金")
    last_price: Optional[float] = Field(default=None, description="最新价")
    gmt_create: Optional[str] = Field(default=None, description="创建时间")
    round_lot: Optional[int] = Field(default=None, description="手数")
    market_value: Optional[float] = Field(default=None, description="市值")
    accumulate_profit: Optional[float] = Field(default=None, description="累计盈亏")
    holding_pnl: Optional[float] = Field(default=None, description="持仓盈亏")

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True
        from_attributes = True
        arbitrary_types_allowed = True
        json_schema_extra = {"description": "Backtest Position Base Model"}

class BacktestPositionModel(BacktestPositionBaseModel):
    id: Annotated[
        PyObjectId,
        Field(
            default_factory=PyObjectId,
            alias="_id",
            description="Backtest Position unique identifier, consistent with MongoDB's ObjectId",
        ),
    ]
