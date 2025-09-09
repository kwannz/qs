from pydantic import BaseModel, Field
from panda_server.models.base_api_response import BaseAPIResponse


class SaveUserPluginResponseData(BaseModel):
    """保存用户插件响应数据模型"""
    
    plugin_id: str = Field(..., description="插件ID")


class SaveUserPluginResponse(BaseAPIResponse[SaveUserPluginResponseData]):
    """保存用户插件响应模型"""
    pass
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": 201,
                "message": "用户插件创建成功",
                "data": {
                    "plugin_id": "64f8d2b3e4b0a1c2d3e4f5a6"
                }
            }
        } 