# from typing import Optional, Type
# import logging

# import pandas as pd
# from pydantic import BaseModel, Field
# from panda_plugins.base.base_work_node import BaseWorkNode
# from panda_plugins.base.work_node_registery import work_node

# class ReadCSVInput(BaseModel):
#     """Input model for ReadCSV node"""
#     file_path: str = Field(..., description="Path to the CSV file")
#     encoding: str = Field(default="utf-8", description="File encoding")
#     sep: str = Field(default=",", description="Delimiter to use")   # 分隔符（默认逗号）
#     header: Optional[int] = Field(default=0, description="Row number to use as column names") #列名所在行（默认第0行）

#     class Config:
#         arbitrary_types_allowed = True

# class ReadCSVOutput(BaseModel):
#     """Output model for ReadCSV node"""
#     data: pd.DataFrame = Field(..., description="DataFrame containing the CSV data")

#     class Config:
#         arbitrary_types_allowed = True

# @work_node(
#     name="read_csv",
#     group="01-基础工具",
#     order=1,
#     type="data_input"
# )
# class ReadCSVNode(BaseWorkNode):
#     """Node for reading CSV files into pandas DataFrame"""

#     @classmethod
#     def input_model(cls) -> Type[BaseModel]:
#         return ReadCSVInput

#     @classmethod
#     def output_model(cls) -> Type[BaseModel]:
#         return ReadCSVOutput

#     def run(self, input: ReadCSVInput) -> ReadCSVOutput:
#         """
#         Read CSV file into pandas DataFrame
        
#         Args:
#             input: ReadCSVInput model containing file path and reading parameters
            
#         Returns:
#             ReadCSVOutput model containing the DataFrame
#         """
#         df = pd.read_csv(
#             input.file_path,
#             encoding=input.encoding,
#             sep=input.sep,
#             header=input.header
#         )
#         return ReadCSVOutput(data=df) 