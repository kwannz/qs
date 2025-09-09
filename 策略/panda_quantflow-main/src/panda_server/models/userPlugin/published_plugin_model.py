from bson import ObjectId
from pydantic import BaseModel, Field, model_validator
from typing import Annotated, Optional
import time


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


class PublishedPluginBaseModel(BaseModel):
    """
    Published Plugin Model
    Published workflow corresponding custom nodes table, invisible to users
    """
    
    name: str = Field(description="Plugin name")
    creator: str = Field(description="Plugin creator")
    code: str = Field(description="Plugin code")
    in_published_workflow_id: str = Field(
        description="Which workflow publication caused this node to be automatically published"
    )
    in_published_workflow_version: str = Field(
        description="Which workflow version publication caused this node to be automatically published"
    )
    create_at: int = Field(
        description="Plugin publish time, unix timestamp, unit: milliseconds"
    )

    # Set values for create_at
    @model_validator(mode="before")
    @classmethod
    def set_defaults(cls, data):
        now = int(time.time() * 1000)
        if "create_at" not in data:
            data["create_at"] = now
        return data

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True
        from_attributes = True
        arbitrary_types_allowed = True
        json_schema_extra = {"description": "Published Plugin Model"}


class PublishedPluginModel(PublishedPluginBaseModel):
    published_plugin_id: Annotated[
        PyObjectId,
        Field(
            default_factory=PyObjectId,
            alias="_id",
            description="Published plugin unique identifier, consistent with MongoDB's ObjectId",
        ),
    ]


class PublishedPluginCreateModel(PublishedPluginBaseModel):
    pass


class PublishedPluginUpdateModel(PublishedPluginBaseModel):
    in_published_workflow_id: Optional[str] = None
    in_published_workflow_version: Optional[str] = None
    name: Optional[str] = None
    creator: Optional[str] = None
    code: Optional[str] = None
    create_at: Optional[int] = None
