from bson import ObjectId
from panda_server.enums.feature_tag import FeatureTag, FeatureTagNodeNames
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Annotated, List, Optional, Dict
from panda_server.enums.feature_tag import FeatureTag, FeatureTagNodeNames
from pydantic import BaseModel, Field, model_validator
from typing import Annotated, List, Optional, Dict
from .work_node_model import WorkNodeModel
from .link_model import LinkModel
from .feature_tag_detail_model import FeatureTagDetailModel
import time
from .feature_tag_detail_model import FeatureTagDetailModel


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


class WorkflowBaseModel(BaseModel):
    """
    Workflow Model
    表示用户的一个工作流程, 包含: 所有节点类型信息, 节点间的连线信息, 节点运行的中间结果数据等. 要求可以以json格式导入和导出.

    Mark: 早期阶段前端较为依赖 Litegraph，使用 litegraph 字段保存其数据, 方便前端读取样式, 将来逐步减少对这个字段的依赖直至移除.
    譬如: 节点坐标, 节点大小, 节点之间的连线信息等均在 litegraph 字段中有表达, 目前阶段前端依赖 litegraph 中的字段来展现 UI.
    """

    # Metadata
    format_version: str = Field(
        default="1.0", description="Workflow format document version"
    )
    name: str
    description: str = Field(
        default="", description="Reserved field for workflow description"
    )
    owner: str = Field(
        default="", description="Reserved field for workflow owner user id"
    )
    create_at: int = Field(
        description="The unix timestamp of the workflow creation time, unit: milliseconds"
    )
    update_at: int = Field(
        description="The unix timestamp of the workflow update time, unit: milliseconds"
    )
    feature_tag: Dict[FeatureTag, List[FeatureTagDetailModel]] = Field(
        default_factory=dict,
        description="Dictionary of feature tags to their details, where keys are FeatureTag enum values",
    )
    last_run_id: str | None = Field(description="last workflow run id", default=None)

    # Litegraph Related Attributes
    litegraph: dict = Field(
        description="Litegraph compatible data, used to store the workflow in the Litegraph format"
    )

    # Nodes Related Attributes
    nodes: List[WorkNodeModel] = Field(
        default_factory=list, description="List of work nodes in the workflow"
    )

    # Links Related Attributes
    links: List[LinkModel] = Field(
        default_factory=list,
        description="List of links between work nodes in the workflow",
    )

    # Set values for create_at and update_at
    @model_validator(mode="before")
    @classmethod
    def set_defaults(cls, data):
        now = int(time.time() * 1000)
        if "create_at" not in data:
            data["create_at"] = now
        if "update_at" not in data:
            data["update_at"] = now
        return data

    # Set feature tag based on nodes data
    @model_validator(mode="after")
    def set_feature_tag(self):
        if self.nodes:
            self.feature_tag = {}
            for tag_name in FeatureTag:
                tag_detail = [
                    FeatureTagDetailModel(node_id=node.uuid, node_title=node.title)
                    for node in self.nodes
                    if node.name in FeatureTagNodeNames[tag_name]
                ]
                if tag_detail:
                    self.feature_tag[tag_name] = tag_detail
        return self

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True
        from_attributes = True
        arbitrary_types_allowed = True
        json_schema_extra = {"description": "Workflow Model"}


class WorkflowModel(WorkflowBaseModel):
    id: Annotated[
        PyObjectId,
        Field(
            default_factory=PyObjectId,
            alias="_id",
            description="Workflow run unique identifier, consistent with MongoDB's ObjectId",
        ),
    ]


class WorkflowCreateModel(WorkflowBaseModel):
    pass


class WorkflowUpdateModel(WorkflowBaseModel):
    format_version: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    litegraph: Optional[dict] = None
    nodes: Optional[List[WorkNodeModel]] = None
    links: Optional[List[LinkModel]] = None
    owner: Optional[str] = Field(default=None, exclude=True)
    feature_tag: Optional[Dict[FeatureTag, List[FeatureTagDetailModel]]] = None
    update_at: Optional[int] = None


if __name__ == "__main__":
    import time

    node1 = WorkNodeModel(
        uuid="node1-uuid",
        name="node1-name",
        type="node1-type",
        litegraph_id=1,
        positionX=10,
        positionY=10,
        width=100,
        height=100,
        static_input_data={
            "input_1": "node1-input_1_value",
            "input_2": "node1-input_2_value",
        },
        output_object_id="507f1f77bcf86cd799439011",
    )

    node2 = WorkNodeModel(
        uuid="node2-uuid",
        name="node2-name",
        type="node2-type",
        litegraph_id=2,
        positionX=20,
        positionY=20,
        width=100,
        height=100,
        static_input_data={},
        output_db_id=None,
    )

    link = LinkModel(
        uuid="link-uuid",
        litegraph_id=1,
        status=1,
        previous_node_uuid="node1-uuid",
        input_field_name="node1_output_1",
        next_node_uuid="node2-uuid",
        output_field_name="node2_input_1",
    )

    workflow = WorkflowModel(
        db_id="507f1f77bcf86cd799439011",
        format_version="1.0",
        name="test-workflow",
        description="test-workflow-description",
        owner="test-user-id",
        create_at=int(time.time() * 1000),
        update_at=int(time.time() * 1000),
        nodes=[node1, node2],
        links=[link],
        litegraph={},
    )

    print(workflow.model_dump_json())
    print(workflow.model_json_schema())
