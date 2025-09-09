# from typing import Optional, Type
# import logging

# import pandas as pd
# from pydantic import BaseModel, Field
# from panda_plugins.base.base_work_node import BaseWorkNode
# from panda_plugins.base.work_node_registery import work_node

# class SaveCSVInput(BaseModel):
#     """Input model for SaveCSV node"""
#     data: pd.DataFrame = Field(..., description="DataFrame to save")
#     file_path: str = Field(..., description="Path where to save the CSV file")
#     encoding: str = Field(default="utf-8", description="File encoding")
#     sep: str = Field(default=",", description="Delimiter to use")
#     index: bool = Field(default=False, description="Whether to write row names (index)") #是否保存索引（默认False）

#     model_config = {'arbitrary_types_allowed': True}

# class SaveCSVOutput(BaseModel):
#     """Output model for SaveCSV node"""
#     success: bool = Field(..., description="Whether the save operation was successful")
#     file_path: str = Field(..., description="Path where the file was saved")

#     model_config = {'arbitrary_types_allowed': True}

# @work_node(
#     name="save_csv",
#     group="01-基础工具",
#     order=1,
#     type="data_output"
# )
# class SaveCSVNode(BaseWorkNode):
#     """Node for saving pandas DataFrame to CSV files"""

#     @classmethod
#     def input_model(cls) -> Type[BaseModel]:
#         return SaveCSVInput

#     @classmethod
#     def output_model(cls) -> Type[BaseModel]:
#         return SaveCSVOutput

#     def run(self, input: SaveCSVInput) -> SaveCSVOutput:
#         """
#         Save DataFrame to CSV file
        
#         Args:
#             input: SaveCSVInput model containing DataFrame and saving parameters
            
#         Returns:
#             SaveCSVOutput model containing the save operation result
#         """
#         input.data.to_csv(
#             input.file_path,
#             encoding=input.encoding,
#             sep=input.sep,
#             index=input.index
#         )
#         return SaveCSVOutput(success=True, file_path=input.file_path) 