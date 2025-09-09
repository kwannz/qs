import ipaddress
import re
from typing import Optional, List

from bson import ObjectId
from pydantic import BaseModel, Field, field_validator, conint, ConfigDict
from datetime import datetime

from panda_trading.models.trading.trading_constant import STRATEGY_TYPE_FUTURES


class PyObjectId(str):
    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return {"type": "string"}

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
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return str(v)
# 数据模型
class RealTradeBindingModel(BaseModel):
    id: Optional[PyObjectId] = Field(None, alias="_id", description="绑定记录ID")
    user_id: PyObjectId = Field(...,  description="用户ID")
    workflow_id: PyObjectId = Field(..., description="工作流ID")
    strategy_id: str = Field(..., description="策略节点ID")
    name: str = Field(..., description="实盘策略名称")
    version: str = Field("20250622", description="实盘策略版本名称")
    future_account: str = Field(..., min_length=6, max_length=50, description="实盘账号")
    strategy_server: str = Field(..., description="策略执行服务器IP")
    strategy_type: conint(ge=0, le=1) = Field(STRATEGY_TYPE_FUTURES, description="策略类型 (0:期货, 1:股票)")
    create_time: Optional[datetime] = Field(None, description="创建时间")
    update_time: Optional[datetime] = Field(None, description="更新时间")
    is_deleted: int = Field(0, description="删除标记 (0:未删除, 1:已删除)")
    class Config:
        from_attributes = True
        populate_by_name = True
        json_schema_extra = {
            "description": "RealTradeBinding Model",
            "example": {
                "user_id": "5f9d1a2b3c4d5e6f7a8b9c0d",
                "workflow_id": "684e79f468db4e40b155ad3e",
                "strategy_id": "a9996ccc-9b53-459d-8f0f-467ace7277b",
                "name": "期货回测示例",
                "version": "20250622",
                "future_account": "242943",
                "strategy_server": "192.168.3.123",
                "strategy_type": 0
            }
        }


class RealTradeBindingCreate(BaseModel):
    user_id: Optional[PyObjectId] = Field(...,  description="用户ID")
    workflow_id: PyObjectId = Field(...,  description="工作流ID")
    strategy_id: str = Field(...,description="策略节点ID")
    name: str = Field(..., description="实盘策略名称")
    version: str = Field("20250622", description="实盘策略版本名称")
    future_account: str = Field(..., min_length=6, max_length=50, description="实盘账号")
    strategy_server: str = Field(..., description="策略执行服务器IP")
    strategy_type: conint(ge=0, le=1) = Field(STRATEGY_TYPE_FUTURES, description="策略类型 (0:期货, 1:股票)")
