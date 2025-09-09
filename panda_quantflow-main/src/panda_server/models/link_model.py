from pydantic import BaseModel, Field
import logging

class LinkModel(BaseModel):
    """
    Link Model

    表示2个 Work Node 之间的连线, 包含: 连线状态, 连线来源节点, 连线来源节点输出字段, 连线目标节点, 连线目标节点输入字段.
    
    Mark: 早期阶段前端较为依赖 Litegraph，在 litegraph 格式的数据中的 links 字段已经包含连线的大部分信息, 但是不包括连线的运行状态.
    """

    uuid: str = Field(description="Link unique identifier")
    litegraph_id: int = Field(description="index in litegraph links properity")
    status: int = Field(
        description="link status, 0: disabled, 1: enabled, 2: running, 3: run_success, 4: run_failed"
    )
    previous_node_uuid: str = Field(description="previous work node uuid")
    input_field_name: str = Field(description="input field name from previous work node")
    next_node_uuid: str = Field(description="next work node uuid")
    output_field_name: str = Field(description="output field name to next work node")

    class Config:
        json_schema_extra = {"description": "Liknk Model"}
