from pydantic import BaseModel, Field
import logging

from typing import List, Dict, Annotated, Optional
from bson import ObjectId

from panda_server.enums.workflow_run_status import WorkflowStatus

class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(cls, *_):
        from pydantic_core import core_schema

        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema(
                [
                    core_schema.is_instance_schema(ObjectId),
                    core_schema.chain_schema(
                        [
                            core_schema.str_schema(),
                            core_schema.no_info_plain_validator_function(cls.validate),
                        ]
                    ),
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x),
                return_schema=core_schema.str_schema(),
                when_used="json",
            ),
        )

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

class WorkflowRunBaseModel(BaseModel):
    """
    工作流运行模型
    表示一次工作流的运行实例，包括工作流ID、状态、进度、各节点运行情况、已通过的连线、节点输出数据等。用于追踪和管理工作流的执行过程。
    """
    workflow_id: str = Field(
        description="Unique identifier of the associated workflow"
    )
    owner: str = Field(
        description="User ID of the workflow run initiator"
    )
    status: WorkflowStatus = Field(
        description="Current status of the workflow run (WorkflowStatus enum)"
    )
    progress: float = Field(
        description="Workflow run progress, range 0-100, as a percentage"
    )
    running_node_ids: List[str] = Field(
        description="List of node IDs currently running"
    )
    success_node_ids: List[str] = Field(
        description="List of node IDs that have completed successfully"
    )
    failed_node_ids: List[str] = Field(
        description="List of node IDs that have failed"
    )
    passed_link_ids: List[str] = Field(
        description="List of link IDs that have been traversed"
    )
    output_data_obj: Dict[str, str] = Field(
        description="Node output data: key: uuid of the node, value: output_db_id"
    )

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True
        from_attributes = True
        arbitrary_types_allowed = True

class WorkflowRunModel(WorkflowRunBaseModel):
    id: Annotated[PyObjectId, Field(default_factory=PyObjectId, alias="_id")]

class WorkflowRunCreateModel(WorkflowRunBaseModel):
    pass

class WorkflowRunUpdateModel(WorkflowRunBaseModel):
    workflow_id: Optional[str] = None
    owner: Optional[str] = Field(default=None, exclude=True)
    status: Optional[WorkflowStatus] = None
    progress: Optional[float] = None
    running_node_ids: Optional[List[str]] = None
    success_node_ids: Optional[List[str]] = None
    failed_node_ids: Optional[List[str]] = None
    passed_link_ids: Optional[List[str]] = None
    output_data_obj: Optional[Dict[str, str]] = None
