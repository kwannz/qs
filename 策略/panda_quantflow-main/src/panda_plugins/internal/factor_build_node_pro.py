from typing import Optional, Type, Union
import logging

logger = logging.getLogger(__name__)
from panda_plugins.base import BaseWorkNode, work_node, ui
from pydantic import BaseModel, Field
from pandas import DataFrame
from panda_factor.generate.macro_factor import MacroFactor

"""
综合因子构建节点
"""
@ui(
    start_date={"input_type": "date_picker","allow_link": False},
    end_date={"input_type": "date_picker","allow_link": False},
    code={"input_type": "None"},
    market={"input_type": "combobox","options": ["股票", "期货"],"placeholder": "因子类型","allow_link": False},
    type={"input_type": "combobox","options": ["Python", "公式"],"placeholder": "编码方式","allow_link": False},
    direction={"input_type": "combobox","options": ["正向", "负向"],"placeholder": "因子方向","allow_link": False}
)
class FactorBuildProInputModel(BaseModel):
    start_date: str = Field(default="20250101",title="开始时间",)
    end_date: str = Field(default="20250301",title="结束时间",)
    code: str = Field(default="",title="因子代码",)
    market: str = Field(default="股票",title="因子类型",)
    type: str = Field(default="Python",title="编码方式",)
    direction:str = Field(default="正向",title="因子方向",)

class FactorBuildProOutputModel(BaseModel):
    factor: DataFrame = Field(..., title="因子值")
    class Config:
        arbitrary_types_allowed = True

@work_node(name="综合因子构建节点", group="04-因子相关", type="general", box_color="blue")
class FactorBuildProControl(BaseWorkNode):
    def __init__(self):
        super().__init__()
        # 添加适配器方法
        self.info = self.log_info
        self.error = self.log_error
        self.warning = self.log_warning
        self.debug = self.log_debug

    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return FactorBuildProInputModel

    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return FactorBuildProOutputModel

    def run(self, input: BaseModel) -> Optional[Type[BaseModel]]:
        macro_factor = MacroFactor()
        if input.market == "股票":
            symbol_type = "stock"
        elif input.market == "期货":
            symbol_type = "future"
            
        if input.type == "Python":
            factor_values = macro_factor.create_factor_from_class(
                factor_logger=self,
                class_code=input.code,
                start_date=input.start_date,
                end_date=input.end_date,
                symbol_type=symbol_type,
            )
        elif input.type == "公式":
            try:
                factor_values = macro_factor.create_factor_from_formula_pro(
                    factor_logger=self._user_logger,
                    formulas=input.code.split("\n"),
                    start_date=input.start_date,
                    end_date=input.end_date,
                    symbol_type=symbol_type,
                )
            except Exception as e:
                error_message = str(e)
                self.error(f"异常信息：{error_message}")
                raise
                
        if input.direction == "负向":
            factor_values.iloc[:, -1] = factor_values.iloc[:, -1] * -1

        print("因子值:")
        print(factor_values)
        if factor_values is None:
            raise ValueError("因子值为空")        
        # 重置索引并修改列名
        factor_df = factor_values.reset_index()
        if 'value' in factor_df.columns:
            factor_df = factor_df.rename(columns={'value': 'factor_value'})
        
        return FactorBuildProOutputModel(factor=factor_df)

if __name__ == "__main__":
    node = FactorBuildProControl()
    formulas = "CLOSE\nLOW"
    input = FactorBuildProInputModel(start_date="20250101",end_date="20250301",formulas=formulas)
    res = node.run(input)
    print(res)
