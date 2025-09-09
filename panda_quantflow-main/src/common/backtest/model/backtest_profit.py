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

class BacktestProfitBaseModel(BaseModel):
    """
    回测收益基础模型类
    用于表示回测收益的信息（不含id）
    """
    day_purchase: Optional[float] = Field(default=None, description="日买入")
    day_put: Optional[float] = Field(default=None, description="日卖出")
    gmt_create: Optional[str] = Field(default=None, description="创建日期")
    gmt_create_time: Optional[str] = Field(default=None, description="创建时间")
    back_id: Optional[str] = Field(default=None, description="回测ID")
    csi_stock: Optional[float] = Field(default=None, description="CSI股票指数")
    strategy_profit: Optional[float] = Field(default=None, description="策略收益")
    day_profit: Optional[float] = Field(default=None, description="日收益")
    overful_profit: Optional[float] = Field(default=None, description="超额收益")

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True
        from_attributes = True
        arbitrary_types_allowed = True
        json_schema_extra = {"description": "Backtest Profit Base Model"}

class BacktestProfitModel(BacktestProfitBaseModel):
    id: Annotated[
        PyObjectId,
        Field(
            default_factory=PyObjectId,
            alias="_id",
            description="Backtest Profit unique identifier, consistent with MongoDB's ObjectId",
        ),
    ]
