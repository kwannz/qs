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

class BacktestAccountBaseModel(BaseModel):
    """
    回测账户基础模型类
    用于表示回测账户的基本信息
    """
    type: Optional[int] = Field(default=None, description="账户类型")
    back_id: Optional[str] = Field(default=None, description="回测ID")
    account_id: Optional[str] = Field(default=None, description="账户ID")
    total_profit: Optional[float] = Field(default=None, description="总收益")
    start_capital: Optional[float] = Field(default=None, description="初始资金")
    available_funds: Optional[float] = Field(default=None, description="可用资金")
    yes_total_capital: Optional[float] = Field(default=None, description="昨日总资金")
    no_settle_total_capital: Optional[float] = Field(default=None, description="未结算总资金")
    gmt_create: Optional[str] = Field(default=None, description="创建时间")
    today_withdraw: Optional[int] = Field(default=0, description="今日提现")
    today_deposit: Optional[int] = Field(default=0, description="今日存款")
    frozen_capital: Optional[float] = Field(default=0, description="冻结资金")
    margin: Optional[float] = Field(default=0, description="保证金")
    market_value: Optional[float] = Field(default=0, description="市值")
    cost: Optional[float] = Field(default=0, description="成本")
    holding_pnl: Optional[float] = Field(default=0, description="持仓盈亏")
    add_profit: Optional[float] = Field(default=0, description="追加收益")
    daily_pnl: Optional[float] = Field(default=0, description="日收益")

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True
        from_attributes = True
        arbitrary_types_allowed = True
        json_schema_extra = {"description": "Backtest Account Base Model"}

class BacktestAccountModel(BacktestAccountBaseModel):
    id: Annotated[
        PyObjectId,
        Field(
            default_factory=PyObjectId,
            alias="_id",
            description="Backtest Account unique identifier, consistent with MongoDB's ObjectId",
        ),
    ]
