from bson import ObjectId
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Annotated
from datetime import datetime

class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, *_):
        from pydantic_core import core_schema
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema([
                core_schema.is_instance_schema(ObjectId),
                core_schema.chain_schema([
                    core_schema.str_schema(),
                    core_schema.no_info_plain_validator_function(cls.validate),
                ])
            ]),
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

class BacktestUserStrategyLogBaseModel(BaseModel):
    """
    用户策略日志基础模型类
    用于表示用户策略日志的信息（不含id）
    """
    content_type: Optional[str] = Field(default=None, description="内容类型")
    exhibit_time: Optional[str] = Field(default=None, description="展示时间")
    insert_time: Optional[str] = Field(default=None, description="插入时间")
    level: Optional[str] = Field(default=None, description="日志级别")
    opz_params_str: Optional[str] = Field(default=None, description="操作参数字符串")
    relation_id: Optional[str] = Field(default=None, description="关联ID")
    run_info: Optional[str] = Field(default=None, description="运行信息")
    sort: Optional[str] = Field(default=None, description="排序字段")
    source: Optional[str] = Field(default=None, description="来源")

    @field_validator('content_type', 'level', 'sort', 'source', mode='before')
    def int_to_str(cls, v):
        if v is not None:
            return str(v)
        return v

    @field_validator('exhibit_time', 'insert_time', mode='before')
    def datetime_to_str(cls, v):
        if isinstance(v, datetime):
            return v.isoformat()
        return v

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True
        from_attributes = True
        arbitrary_types_allowed = True
        json_schema_extra = {"description": "Backtest User Strategy Log Base Model"}

class BacktestUserStrategyLogModel(BacktestUserStrategyLogBaseModel):
    id: Annotated[
        PyObjectId,
        Field(
            default_factory=PyObjectId,
            alias="_id",
            description="Backtest User Strategy Log unique identifier, consistent with MongoDB's ObjectId",
        ),
    ] 