from pydantic import BaseModel, Field, field_validator
from typing import Optional


class SaveUserPluginRequest(BaseModel):
    """保存用户插件请求模型"""
    
    plugin_id: Optional[str] = Field(None, description="插件ID，如果提供则为更新，否则为新建")
    name: str = Field(..., description="插件名称")
    code: str = Field(..., description="插件代码")
    
    # 使用 model_config 来禁止额外字段并提供 JSON schema 示例
    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "name": "CustomDataProcessor",
                    "code": '''from panda_plugins.base import BaseWorkNode, work_node
from pydantic import BaseModel
from typing import Optional

class InputModel(BaseModel):
    data: str = Field(..., description="输入数据")

class OutputModel(BaseModel):
    result: str = Field(..., description="处理结果")

@work_node(name="自定义数据处理器", group="自定义节点")
class CustomDataProcessor(BaseWorkNode):
    
    @classmethod
    def input_model(cls) -> Optional[type[BaseModel]]:
        return InputModel
    
    @classmethod  
    def output_model(cls) -> Optional[type[BaseModel]]:
        return OutputModel
    
    def run(self, input: InputModel) -> OutputModel:
        result = f"处理结果: {input.data}"
        return OutputModel(result=result)
'''
                },
                {
                    "name": "EmptyPlugin",
                    "code": ""
                },
                {
                    "plugin_id": "64f8d2b3e4b0a1c2d3e4f5a6",
                    "name": "UpdatedPlugin",
                    "code": "# 更新的插件代码"
                }
            ]
        }
    }
    
    @field_validator('plugin_id')
    @classmethod
    def validate_plugin_id(cls, v):
        """验证插件ID"""
        if v is not None:
            if not isinstance(v, str) or not v.strip():
                raise ValueError("插件ID必须是非空字符串")
            # 验证ObjectId格式
            from bson import ObjectId
            try:
                ObjectId(v)
            except Exception:
                raise ValueError("插件ID必须是有效的ObjectId格式")
        return v
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """验证插件名称"""
        if not v or not v.strip():
            raise ValueError("插件名称不能为空")
        return v.strip()
    
    @field_validator('code')
    @classmethod
    def validate_code(cls, v):
        """验证插件代码"""
        if v is None:
            raise ValueError("插件代码字段不能为None")
        # 允许空字符串，但如果有内容则去掉前后空格
        return v.strip() if v else "" 