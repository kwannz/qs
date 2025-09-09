from pydantic import BaseModel
from typing import List, Dict, Any, Literal, Optional
from .base_api_response import BaseAPIResponse

class PluginInfo(BaseModel):
    """单个插件的信息"""
    object_type: str = "plugin"
    name: str
    display_name: str
    group: str
    type: str
    box_color: Optional[Literal["red", "brown", "green", "blue", "cyan", "purple", "yellow", "black"]] = "black"
    short_description: str
    long_description: str
    input_schema: Optional[Dict[str, Any]] = {}
    output_schema: Optional[Dict[str, Any]] = {}

class PluginGroup(BaseModel):
    object_type: Optional[str] = "group"
    name: str
    group: Optional[str] = None
    children: List[Any]

class AllPluginsResponse(BaseAPIResponse[List[PluginGroup | PluginInfo]]):
    """所有插件的响应模型"""
    pass