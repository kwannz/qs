import traceback
from typing import Optional, Type
import logging

from panda_plugins.base import BaseWorkNode, work_node,ui
from pydantic import BaseModel,Field, field_validator
import pandas as pd
from panda_backtest.main_workflow_future import start
logger = logging.getLogger(__name__)
strategy_code_default = '''
from panda_backtest.api.api import *
import pandas as pd
import numpy as np
import copy
import datetime
import re
import pickle
import sys

def initialize(context):
    SRLogger.info("策略初始化")
    context.account = '5588'
    context.trading_code="AG2509.SHF"

def before_trading(context):
    SRLogger.info("交易前")

def handle_data(context, bar_dict):
    if (int(context.now) % 2) == 0:
        buy_open(account_id=context.account,id_or_ins=context.trading_code,amount=1)
    else:
        sell_open(account_id=context.account,id_or_ins=context.trading_code,amount=-1)

def after_trading(context):
    SRLogger.info("交易后")
'''.strip()
@ui(
    code={"input_type":"text_field","placeholder": "策略代码"},
    factors={"input_type": "None"},
    start_future_capital={"input_type": "number_field","placeholder": "请输入初始资金", "allow_link": False},
    # future_account_id={"input_type":"text_field","placeholder": "请输入期货账户（非真实）", "allow_link": False},
    commission_rate={"input_type": "number_field","allow_link": False,},
    margin_rate={"input_type": "number_field","allow_link": False,},
    frequency={"input_type": "combobox","options": ["1d", ],"placeholder": "回测频率","allow_link": False},
    start_date={"input_type":"date_field","placeholder": "开始日期", "allow_link": False},
    end_date={"input_type":"date_field","placeholder": "结束日期", "allow_link": False}
)
class FutureBacktestInputModel(BaseModel):

    code: str=Field(default="",title="策略代码")
    factors: pd.DataFrame = Field(default_factory=pd.DataFrame, title="因子值")
    start_future_capital:int=Field(default=10000000, title="初始资金")
    commission_rate:int=Field(default=1,title="佣金倍率")
    margin_rate:int=Field(default=1,title="保证金倍率")
    frequency:str=Field(default="1d",title="回测频率")
    start_date:str=Field(default="20241022",title="开始日期")
    end_date:str=Field(default="20241231",title="结束日期")

    @field_validator('factors')
    def validate_df_factor(cls, v):
        if not isinstance(v, pd.DataFrame):
            raise ValueError('factors must be a pandas DataFrame')
        return v

class FutureBacktestOutputModel(BaseModel):

    backtest_id: str = Field(default="error",title="回测任务")

@work_node(name="期货回测", group="05-回测相关",type="general", box_color="yellow")
class FutureBacktestControl(BaseWorkNode):
    # Return the input model
    # 返回输入模型
    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return FutureBacktestInputModel

    # Return the output model
    # 返回输出模型
    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return FutureBacktestOutputModel

    # Node running logic
    # 节点运行逻辑
    def run(self, input: BaseModel) -> BaseModel:
        try:
            backtest_id=start(code=input.code,start_future_capital=input.start_future_capital,future_account_id="8888",start_date=input.start_date,end_date=input.end_date,commission_rate=input.commission_rate,margin_rate=input.margin_rate,frequency=input.frequency,df_factor=input.factors)
        except Exception as e:
            if hasattr(e, "message"):
                # 提取前两行和最后一行
                error_lines = e.message.splitlines()
                self.log_error(error_lines[0] + "\n" + error_lines[1] + "\n异常信息：" + error_lines[2])
                # backtest_id="error"
            else:
                self.log_error(str(e))
        return FutureBacktestOutputModel(backtest_id=str(backtest_id))

if __name__ == "__main__":
    node = FutureBacktestControl()
    # df = pd.read_csv(
    #     '/Users/peiqi/code/python/panda_workflow/src/panda_plugins/internal/test_factor.csv',
    #     usecols=["date", "symbol", "CZ1a14685"],  # 只读取需要的列，节省内存
    #     dtype={"date": str}  # 明确指定date列为字符串类型
    # )
    # df["abcdef"]=df["CZ1a14685"]
    input = FutureBacktestInputModel()
    print(node.run(input))

