from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from bson import ObjectId

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

# 主模型
class FutureAccountModel(BaseModel):
    """
    期货账户模型类（对应 redefine_future_account）
    """
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    is_deleted: Optional[int] = Field(default=0, description="是否删除")
    gmt_create: Optional[datetime] = Field(default=None, description="创建时间")
    gmt_modified: Optional[datetime] = Field(default=None, description="更新时间")
    modify_user: Optional[str] = Field(default=None, description="修改人")
    account_id: Optional[str] = Field(default=None, description="账号")
    password: Optional[str] = Field(default=None, description="密码")
    trade_server_url: Optional[str] = Field(default=None, description="CTP交易服务器URL")
    market_server_url: Optional[str] = Field(default=None, description="CTP行情服务器URL")
    product_info: Optional[str] = Field(default=None, description="CTP系统标识")
    auth_code: Optional[str] = Field(default=None, description="CTP认证码")
    broker_id: Optional[str] = Field(default=None, description="BrokerID")
    remark: Optional[str] = Field(default=None, description="备注")
    name: Optional[str] = Field(default=None, description="账号名称")
    user_id: int = Field(default=0, description="用户ID")
    trade_adapter: Optional[int] = Field(default="ctp", description="交易适配器（0:CTP）")

    class Config:
        from_attributes = True
        populate_by_name = True
        json_schema_extra = {
            "description": "Future Account Model",
            "example": {
                "is_deleted": 0,
                "gmt_create": "2021-01-07T10:49:36",
                "gmt_modified": "2024-06-14T09:28:58",
                "modify_user": None,
                "account_id": "108404",
                "password": "Xb8888888!",
                "trade_server_url": "tcp://180.168.146.187:10201",
                "market_server_url": "tcp://180.168.146.187:10211",
                "product_info": "test",
                "auth_code": "0000000000000000",
                "broker_id": "9999",
                "remark": None,
                "name": "期货测试",
                "user_id": 1,
                "trade_adapter": 0
            }
        }

# 创建模型：所有字段都是可选的（除了非空字段）
class FutureAccountCreateModel(FutureAccountModel):
    account_id: str = Field(..., description="账号")
    password: str = Field(..., description="密码")
    trade_server_url: str = Field(..., description="CTP交易服务器URL")
    market_server_url: str = Field(..., description="CTP行情服务器URL")
    product_info: str = Field(..., description="CTP系统标识")
    auth_code: str = Field(..., description="CTP认证码")
    broker_id: str = Field(..., description="BrokerID")
    name: str = Field(..., description="账号名称")
    user_id: int = Field(..., description="企业ID")
    trade_adapter: int = Field(..., description="交易适配器（0:CTP）")

# 更新模型：所有字段都可选
class FutureAccountUpdateModel(BaseModel):
    is_deleted: Optional[PyObjectId] = None
    gmt_modified: Optional[datetime] = None
    modify_user: Optional[str] = None
    account_id: Optional[str] = None
    password: Optional[str] = None
    trade_server_url: Optional[str] = None
    market_server_url: Optional[str] = None
    product_info: Optional[str] = None
    auth_code: Optional[str] = None
    broker_id: Optional[str] = None
    remark: Optional[str] = None
    name: Optional[str] = None
    user_id: Optional[int] = None
    trade_adapter: Optional[int] = None