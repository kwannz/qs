from pydantic import BaseModel, Field, field_validator, model_validator
import logging

from typing import Dict, Any, Optional, List
from panda_server.models.work_node_model import WorkNodeModel
from panda_server.models.link_model import LinkModel

class SaveWorkflowRequest(BaseModel):
    """保存工作流请求验证模型"""
    
    id: Optional[str] = Field(default=None, description="工作流ID，如果提供则为修改，否则为新建")
    name: str = Field(description="工作流名称", max_length=255)
    description: Optional[str] = Field(default=None, description="工作流描述")
    format_version: Optional[str] = Field(default=None, description="工作流格式版本")

    # 工作流核心数据
    nodes: Optional[List[WorkNodeModel]] = Field(default=None, description="工作流节点列表")
    links: Optional[List[LinkModel]] = Field(default=None, description="节点间连接列表")
    litegraph: Optional[dict] = Field(default=None, description="Litegraph兼容数据")
    
    # 使用 model_config 来禁止额外字段，确保数据结构严格
    model_config = {"extra": "forbid"}
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v):
        """验证工作流ID"""
        if v is not None:
            if not isinstance(v, str) or not v.strip():
                raise ValueError("工作流ID必须是非空字符串")
            # 验证ObjectId格式
            from bson import ObjectId
            try:
                ObjectId(v)
            except Exception:
                raise ValueError("工作流ID必须是有效的ObjectId格式")
        return v
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """验证工作流名称"""
        if not v or not v.strip():
            raise ValueError("工作流名称不能为空")
        return v.strip()
    
    @field_validator('description')
    @classmethod
    def validate_description(cls, v):
        """验证工作流描述"""
        if v is not None:
            return v.strip()
        return v
    
    @field_validator('format_version')
    @classmethod
    def validate_format_version(cls, v):
        """验证格式版本"""
        if v is not None:
            if not v.strip():
                raise ValueError("格式版本不能为空字符串")
            return v.strip()
        return v
    
    @field_validator('nodes')
    @classmethod
    def validate_nodes(cls, v):
        """验证节点列表"""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("nodes必须是列表类型")
            
            # 检查节点UUID的唯一性
            uuids = [node.uuid for node in v]
            if len(uuids) != len(set(uuids)):
                raise ValueError("节点UUID必须唯一")
            
            # 检查litegraph_id的唯一性
            litegraph_ids = [node.litegraph_id for node in v]
            if len(litegraph_ids) != len(set(litegraph_ids)):
                raise ValueError("节点litegraph_id必须唯一")
        
        return v
    
    @field_validator('links')
    @classmethod
    def validate_links(cls, v):
        """验证连接列表"""
        if v is not None:
            if not isinstance(v, list):
                raise ValueError("links必须是列表类型")
            
            # 检查连接UUID的唯一性
            uuids = [link.uuid for link in v]
            if len(uuids) != len(set(uuids)):
                raise ValueError("连接UUID必须唯一")
        
        return v
    
    @field_validator('litegraph')
    @classmethod
    def validate_litegraph(cls, v):
        """验证litegraph数据"""
        if v is not None:
            if not isinstance(v, dict):
                raise ValueError("litegraph必须是字典类型")
        return v
    
    @model_validator(mode='after')
    def validate_workflow_consistency(self):
        """验证工作流数据的一致性"""
        # 对于新建工作流，必须提供必要字段
        if self.id is None:  # 新建
            if self.litegraph is None:
                raise ValueError("新建工作流时必须提供litegraph字段")
        
        # 如果提供了nodes和links，检查连接引用的节点是否存在
        if self.nodes is not None and self.links is not None:
            node_uuids = {node.uuid for node in self.nodes}
            
            for link in self.links:
                if link.previous_node_uuid not in node_uuids:
                    raise ValueError(f"连接引用的前置节点UUID '{link.previous_node_uuid}' 不存在")
                if link.next_node_uuid not in node_uuids:
                    raise ValueError(f"连接引用的后续节点UUID '{link.next_node_uuid}' 不存在")
        
        return self
    
    def get_workflow_data(self) -> Dict[str, Any]:
        """获取工作流数据（排除id字段，过滤None值）"""
        data = self.model_dump(exclude={'id'}, exclude_none=True)
        return data 