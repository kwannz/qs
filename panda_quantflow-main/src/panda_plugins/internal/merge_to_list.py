# from typing import Any, Optional, Type
# from panda_plugins.base import BaseWorkNode, work_node
# from panda_plugins.base.ui_control import ui
# from pydantic import BaseModel, Field


# @ui(
#     element1={"input_type": "None"},
#     element2={"input_type": "None"},
#     element3={"input_type": "None"},
#     element4={"input_type": "None"},
#     element5={"input_type": "None"},
#     element6={"input_type": "None"},
#     element7={"input_type": "None"},
#     element8={"input_type": "None"},
#     element9={"input_type": "None"},
#     element10={"input_type": "None"},
# )
# class InputModel(BaseModel):
#     element1: Any
#     element2: Any = Field(default=None)
#     element3: Any = Field(default=None)
#     element4: Any = Field(default=None)
#     element5: Any = Field(default=None)
#     element6: Any = Field(default=None)
#     element7: Any = Field(default=None)
#     element8: Any = Field(default=None)
#     element9: Any = Field(default=None)
#     element10: Any = Field(default=None)


# class OutputModel(BaseModel):
#     result: list


# @work_node(name="多元素合并列表", group="语法节点")
# class MergeToList(BaseWorkNode):
#     """
#     Merge multiple elements into a Python list type.

#     将多个元素合并成一个 python list 类型.
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
#         result = []
#         for i in range(1, 11):
#             element = getattr(input, f"element{i}")
#             if element is not None:
#                 result.append(element)
#         return OutputModel(result=result)


# if __name__ == "__main__":
#     node = MergeToList()
#     input = InputModel(element1=1, element2=2)
#     print(node.run(input))
