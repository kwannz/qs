# from typing import Any, Optional, Type
# from panda_plugins.base import BaseWorkNode, work_node
# from panda_plugins.base.ui_control import ui
# from pydantic import BaseModel, Field


# @ui(
#     list1={"input_type": "None"},
#     list2={"input_type": "None"},
#     list3={"input_type": "None"},
#     list4={"input_type": "None"},
#     list5={"input_type": "None"},
# )
# class InputModel(BaseModel):
#     list1: list
#     list2: list = Field(default_factory=list)
#     list3: list = Field(default_factory=list)
#     list4: list = Field(default_factory=list)
#     list5: list = Field(default_factory=list)


# class OutputModel(BaseModel):
#     result: list


# @work_node(name="多列表合并", group="语法节点")
# class MergeLists(BaseWorkNode):
#     """
#     Merge multiple  lists into a Python list type.

#     将多个 list 合并成一个 python list 类型.
#     """

#     # Return the input model
#     # 返回输入模型
#     @classmethod
#     def input_model(cls) -> Optional[Type[BaseModel]]:
#         return InputModel

#     # Return the output model
#     # 返回输出模型
#     @classmethod
#     def output_model(cls) -> Optional[Type[BaseModel]]:
#         return OutputModel

#     # Node running logic
#     # 节点运行逻辑
#     def run(self, input: InputModel) -> BaseModel:
#         result = [
#             *input.list1,
#             *input.list2,
#             *input.list3,
#             *input.list4,
#             *input.list5,
#         ]
#         return OutputModel(result=result)


# if __name__ == "__main__":
#     node = MergeLists()
#     input = InputModel(list1=[1,2], list2=[3])
#     print(node.run(input))
