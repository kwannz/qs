from pydantic import BaseModel, Field


class RunWorkflowRequest(BaseModel):
    """运行工作流请求模型"""
    
    workflow_id: str = Field(..., description="工作流ID")


class TerminateWorkflowRunRequest(BaseModel):
    """终止工作流运行请求模型"""
    
    workflow_run_id: str = Field(..., description="工作流运行ID") 