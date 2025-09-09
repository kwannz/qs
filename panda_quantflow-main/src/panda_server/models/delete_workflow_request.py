from pydantic import BaseModel, Field, field_validator
from typing import List

class DeleteWorkflowRequest(BaseModel):
    """批量删除工作流请求模型"""
    
    workflow_id_list: List[str] = Field(description="要删除的工作流ID列表")
    
    @field_validator('workflow_id_list')
    @classmethod
    def validate_workflow_id_list(cls, v):
        """验证工作流ID列表"""
        if not v or len(v) == 0:
            raise ValueError("工作流ID列表不能为空")
        
        if not isinstance(v, list):
            raise ValueError("workflow_id_list必须是列表类型")
        
        # 验证每个ID都是有效的ObjectId格式
        from bson import ObjectId
        for workflow_id in v:
            if not isinstance(workflow_id, str) or not workflow_id.strip():
                raise ValueError("工作流ID必须是非空字符串")
            try:
                ObjectId(workflow_id)
            except Exception:
                raise ValueError(f"工作流ID '{workflow_id}' 不是有效的ObjectId格式")
        
        # 检查是否有重复的ID
        if len(v) != len(set(v)):
            raise ValueError("工作流ID列表中不能有重复的ID")
        
        return v 