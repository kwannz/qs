from typing import Optional, Type
from panda_plugins.base import BaseWorkNode, work_node
from pydantic import BaseModel

class InputModel(BaseModel):
    """
    Define the input model for the node.
    Use pydantic to define, which is a library for data validation and parsing.
    Reference: https://pydantic-docs.helpmanual.io
    
    
    为工作节点定义输入模型.
    使用 Pydantic 定义, Pydantic 是一个用于数据验证和解析的库.
    参考文档: https://pydantic-docs.helpmanual.io
    """
    number1: int
    number2: int

class OutputModel(BaseModel):
    """
    Define the output model for the node.
    Use pydantic to define, which is a library for data validation and parsing.
    Reference: https://pydantic-docs.helpmanual.io
    
    为工作节点定义输出模型.
    使用 Pydantic 定义, Pydantic 是一个用于数据验证和解析的库.
    参考文档: https://pydantic-docs.helpmanual.io
    """
    result: int

@work_node(name="示例-两数求和", group="测试节点")
class ExamplePluginAddition(BaseWorkNode):
    """
    Implement a example node, which can add two numbers and return the result.
    实现一个示例节点, 完成一个简单的加法运算, 输入 2 个数值, 输出 2 个数值的和.
    """

    # Return the input model
    # 返回输入模型
    @classmethod
    def input_model(cls) -> Optional[Type[BaseModel]]:
        return InputModel

    # Return the output model
    # 返回输出模型
    @classmethod
    def output_model(cls) -> Optional[Type[BaseModel]]:
        return OutputModel

    # Node running logic
    # 节点运行逻辑
    def run(self, input: BaseModel) -> BaseModel:
        result = input.number1 + input.number2
        return OutputModel(result=result)

# 设置简短预览介绍 (HTML 富文本, 推荐使用工具: https://html5-editor.net) 
ExamplePluginAddition.set_short_description('''<p>这是一个简单的加法运算, &nbsp;输入 2 个数值, 输出 <strong><span style="color: rgb(209, 72, 65);">2 个数值的和</span></strong>.</p>''')

# 设置完整预览介绍 (HTML 富文本, 推荐使用工具: https://html5-editor.net)
ExamplePluginAddition.set_long_description('''<p>这是一个简单的加法运算, &nbsp;输入 2 个数值, 输出 <strong><span style="color: rgb(209, 72, 65);">2 个数值的和</span></strong>.</p>
<p>这里是一张图片说明</p>
<p><img src="https://code2048.com/images/betacat/betacat02/20.png" alt="示例图片"></pp>
<p>这里是一段视频演示</p>
<p><iframe width="640" height="360" src="https://www.youtube.com/embed/PPbvLGgkn44?&wmode=opaque&rel=0" frameborder="0" allowfullscreen=""></iframe></p>''')

if __name__ == "__main__":
    node = ExamplePluginAddition()
    input = InputModel(number1=1, number2=2)
    print(node.run(input))
        
    
