import ipaddress
import re
from typing import Optional, List

from bson import ObjectId
from pydantic import BaseModel, Field, field_validator, conint, ConfigDict
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
        return str(v)

# 定义Pydantic模型
class RealTradStrategyServerModel(BaseModel):
    id: Optional[PyObjectId] = Field(..., alias="_id", description="文档ID")
    server_ip: Optional[str] = Field(None, description="服务器IP")
    name: Optional[str] = Field(None, description="服务器名称")
    remark: Optional[str] = Field(None, description="服务器备注")
    status: Optional[int] = Field(-1, description="设置服务器状态 实盘服务器状态:-2异常、-1待运行、0停止、1运行中")

class PageResponse(BaseModel):
    total: int
    page: int
    size: int
    items: List[RealTradStrategyServerModel]


class BaseAPIResponse(BaseModel):
    code: int
    message: str
    data: PageResponse



# 注册请求模型
class RegisterServerRequest(BaseModel):
    server_ip: str = Field(..., min_length=7, max_length=15, description="服务器IP地址")
    name: str = Field(..., min_length=2, max_length=100, description="服务器名称")
    remark: Optional[str] = Field(None, max_length=200, description="服务器备注")

    # IP格式校验
    @field_validator('server_ip')
    def validate_ip(cls, v):
        try:
            ipaddress.ip_address(v)
            return v
        except ValueError:
            raise ValueError("无效的IP地址格式")

    # 名称格式校验（只允许中文、字母、数字、下划线和短横线）
    @field_validator('name')
    def validate_name(cls, v):
        if not re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9_-]{2,100}$', v):
            raise ValueError("名称只能包含中文、字母、数字、下划线和短横线")
        return v


# 注册响应模型
class RegisterResponse(BaseModel):
    server_id: str = Field(..., description="注册成功的服务器ID")


# 状态模型
class ServerStatus(BaseModel):
    id: Optional[PyObjectId] = Field(..., description="服务器ID")
    status: int = Field(-1, description="实盘服务器状态:-2异常、-1待运行、1运行中、0停止")


# 存在性检查响应模型
class ServerExistenceResponse(BaseModel):
    exists: bool = Field(-1, description="服务器是否存在")
    status: Optional[int] = Field(None, description="当前实盘服务器状态:-2异常、-1待运行、1运行中、0停止")


