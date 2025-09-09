from typing import Optional, List, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, conint, ConfigDict, model_validator
from bson import ObjectId

# 自定义 PyObjectId 类型
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
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return str(v)


# 期货成交记录模型
class FutureTradeOrderModel(BaseModel):
    id: Optional[PyObjectId] = Field(None, alias="_id", description="成交记录ID (MongoDB ObjectId)")
    account_id: str = Field(..., description="账户编号")
    business: conint(ge=0, le=1) = Field(..., description="业务类型 (0:开仓, 1:平仓)")
    client_id: str = Field(..., description="客户编号")
    contract_code: str = Field(..., description="合约代码 (如 IF2506.CFE)")
    direction: conint(ge=0, le=1) = Field(..., description="交易方向 (0:买入, 1:卖出)")
    gmt_create: str = Field(..., pattern=r"^\d{8}$", description="订单创建日期 (YYYYMMDD)")
    gmt_create_time: str = Field(..., pattern=r"^\d{2}:\d{2}:\d{2}$", description="订单创建时间 (HH:MM:SS)")
    market: str = Field(..., description="市场名称 (CFFEX, SHFE 等)")
    now_system_order: conint(ge=0, le=1) = Field(..., description="是否为系统订单 (0:否, 1:是)")
    order_id: str = Field(..., description="报单编号")
    order_remark: Optional[str] = Field("", description="订单备注 (如 group_order)")
    order_sys_id: str = Field(..., description="系统订单编号")
    price: float = Field(..., ge=0.0, description="成交价格")
    run_id: str = Field(..., description="策略运行实例ID")
    run_type: int = Field(..., description="运行类型")
    trade_date: str = Field(..., pattern=r"^\d{8}$", description="成交日期 (YYYYMMDD)")
    trade_id: str = Field(..., description="成交ID")
    type_: int = Field(..., alias="type", description="订单类型")
    volume: int = Field(..., ge=1, description="成交手数")

    # 计算属性（可以加方法或 property 来补充缺失字段）
    @property
    def futures_direction(self) -> str:
        """根据 business 和 direction 推导出期货方向"""
        if self.business == 0 and self.direction == 0:
            return "看多"
        elif self.business == 0 and self.direction == 1:
            return "看空"
        elif self.business == 1 and self.direction == 0:
            return "平空"
        else:
            return "平多"

    @property
    def order_status(self) -> str:
        """挂单状态"""
        return "已成交" if self.now_system_order == 1 else "未成交"

    @property
    def unexecuted_volume(self) -> int:
        """未成交手数，默认为0（因为所有数据都是成交单）"""
        return 0

    @property
    def detail_status(self) -> str:
        """详细状态"""
        return self.order_remark or "正常"

    # 额外配置
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_schema_extra={
            "description": "Future Trade Record Model",
            "example": {
                "_id": "682d5dd4a8feb89488821f4c",
                "account_id": "58000186",
                "business": 0,
                "client_id": "0000001",
                "contract_code": "IF2506.CFE",
                "direction": 0,
                "gmt_create": "20250521",
                "gmt_create_time": "13:00:05",
                "market": "CFFEX",
                "now_system_order": 0,
                "order_id": "86363658037",
                "order_remark": "group_order",
                "order_sys_id": "15049009",
                "price": 3886.6,
                "run_id": "2457",
                "run_type": 2,
                "trade_date": "20250521",
                "trade_id": "346007",
                "type": 1,
                "volume": 1
            }
        },
    )