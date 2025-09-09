from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional


class WorkNodeModel(BaseModel):
    """
    Work Node Model
    表示一个工作节点的信息, 包含: 节点类型信息, 节点坐标, 节点大小, 节点样式, 节点输入数据, 节点输出数据等.
    但是不包括节点内的运行逻辑, 节点内的运行逻辑由对应开发的 plugins 的 .py 文件决定(在 panda_plugins/internal 或 panda_plugins/custom 中).

    Mark: 早期阶段前端较为依赖 Litegraph，此处部分字段与 Litegraph 的数据有重复, 先做为冗余字段保存. 将来逐步减少对 Litegraph 的依赖, 改为依赖这里的字段.
    重点关注:
        static_input_data 表示那些用户手动输入的数据(非连线输入), litegraph 中将对这部分的数据的保存有 bug, 我们需要实现保存.
        output_db_id 用于保存节点运行结果, 在 litegraph 中没有保存这部分数据, 而我们的需求场景需要保存这部分数据到数据库. 以 MongoDB 的 ObjectId 格式保存引用.
    """

    # Metadata
    uuid: str = Field(description="Work Node unique identifier")
    name: str = Field(
        description="Work Node name, consistent with the class attribute __work_node_name in BaseWorkNode"
    )
    title: str = Field(description="Work Node title, defaults to name if not provided")
    type: str = Field(
        description="Work Node type, consistent with the class attribute __work_node_type in BaseWorkNode"
    )

    # Litegraph Related Attributes
    litegraph_id: int = Field(description="Litegraph node id")

    # Style Related Attributes
    positionX: float = Field(
        description="The x coordinate of the node in the workflow canvas"
    )
    positionY: float = Field(
        description="The y coordinate of the node in the workflow canvas"
    )
    width: float = Field(description="The width of the node in the workflow canvas")
    height: float = Field(description="The height of the node in the workflow canvas")

    # Data Related Attributes
    static_input_data: dict = Field(
        default={},
        description="Static input data for the node, must be values manually entered by users, not values linked from the output of previous work nodes",
    )
    output_db_id: Optional[str] = Field(
        default=None,
        description="The object id of the output data for the node, consistent with MongoDB's ObjectId",
    )

    @model_validator(mode="before")
    @classmethod
    def set_defaults(cls, data):
        # set default title by name
        if "name" in data and (data.get("title") is None):
            data["title"] = data["name"]
        return data

    class Config:
        json_schema_extra = {"description": "Work Node Model"}
