from bson import ObjectId
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Annotated, List, Optional, Dict


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

class BacktestBaseModel(BaseModel):
    """
    回测模型类
    用于表示回测的基本信息和结果
    """
    run_type: Optional[str] = Field(default=None, description="运行类型")
    strategy_id: Optional[str] = Field(default=None, description="策略ID")
    fund_stock: Optional[str] = Field(default=None, description="股票资金")
    fund_futures: Optional[str] = Field(default=None, description="期货资金")
    fund_funds: Optional[str] = Field(default=None, description="基金资金")
    benchmark: Optional[str] = Field(default=None, description="基准指数")
    commission: Optional[str] = Field(default=None, description="手续费率")
    margin: Optional[str] = Field(default=None, description="保证金率")
    slippage: Optional[str] = Field(default=None, description="滑点")
    account_type: Optional[str] = Field(default=None, description="账户类型")
    start_date: Optional[str] = Field(default=None, description="开始日期")
    end_date: Optional[str] = Field(default=None, description="结束日期")
    back_interval: Optional[str] = Field(default=None, description="回测间隔")
    bar_match: Optional[str] = Field(default=None, description="Bar匹配方式")
    strategy_code: Optional[str] = Field(default=None, description="策略代码")
    account_id: Optional[str] = Field(default=None, description="账户ID")
    future_account_id: Optional[str] = Field(default=None, description="期货账户ID")
    fund_account_id: Optional[str] = Field(default=None, description="基金账户ID")
    date_type: Optional[str] = Field(default=None, description="日期类型")
    fund_rate_data: Optional[str] = Field(default=None, description="资金费率数据")
    run_params: Optional[str] = Field(default=None, description="运行参数")
    run_status: Optional[str] = Field(default=None, description="运行状态")
    gmt_create: Optional[str] = Field(default=None, description="创建时间")
    alpha: Optional[str] = Field(default=None, description="Alpha值")
    back_profit: Optional[str] = Field(default=None, description="回测收益")
    back_profit_year: Optional[str] = Field(default=None, description="年化回测收益")
    benchmark_name: Optional[str] = Field(default=None, description="基准名称")
    benchmark_profit: Optional[str] = Field(default=None, description="基准收益")
    benchmark_profit_year: Optional[str] = Field(default=None, description="年化基准收益")
    beta: Optional[str] = Field(default=None, description="Beta值")
    custom_tag: Optional[str] = Field(default=None, description="自定义标签")
    downside_risk: Optional[str] = Field(default=None, description="下行风险")
    information_ratio: Optional[str] = Field(default=None, description="信息比率")
    kama_ratio: Optional[str] = Field(default=None, description="卡玛比率")
    max_drawdown: Optional[str] = Field(default=None, description="最大回撤")
    sharpe: Optional[str] = Field(default=None, description="夏普比率")
    sortino: Optional[str] = Field(default=None, description="索提诺比率")
    time_consume: Optional[str] = Field(default=None, description="耗时")
    tracking_error: Optional[str] = Field(default=None, description="跟踪误差")
    volatility: Optional[str] = Field(default=None, description="波动率")

    @field_validator(
        'run_status', 'alpha', 'back_profit', 'back_profit_year', 'benchmark_profit', 'benchmark_profit_year',
        'beta', 'downside_risk', 'information_ratio', 'kama_ratio', 'max_drawdown', 'sharpe', 'sortino',
        'time_consume', 'tracking_error', 'volatility', mode='before'
    )
    def num_to_str(cls, v):
        if v is not None:
            return str(v)
        return v

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True
        from_attributes = True
        arbitrary_types_allowed = True
        json_schema_extra = {"description": "Backtest Model"}

    
class BacktestModel(BacktestBaseModel):
    id: Annotated[
        PyObjectId,
        Field(
            default_factory=PyObjectId,
            alias="_id",
            description="Backtest unique identifier, consistent with MongoDB's ObjectId",
        ),
    ]    