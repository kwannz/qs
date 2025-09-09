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


class UserPluginBaseModel(BaseModel):
    """
    User Plugin Model
    用户的自定义节点表
    """
    
    name: str = Field(description="Plugin name")
    creator: str = Field(description="Plugin creator")
    code: str = Field(description="Plugin code")
    create_at: int = Field(
        description="Plugin creation time, unix timestamp, unit: milliseconds"
    )
    update_at: int = Field(
        description="Plugin last update time, unix timestamp, unit: milliseconds"
    )

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True
        from_attributes = True
        arbitrary_types_allowed = True
        json_schema_extra = {"description": "User Plugin Model"}


class UserPluginModel(UserPluginBaseModel):
    plugin_id: Annotated[
        PyObjectId,
        Field(
            default_factory=PyObjectId,
            alias="_id",
            description="User plugin unique identifier, consistent with MongoDB's ObjectId",
        ),
    ]


class UserPluginCreateModel(UserPluginBaseModel):
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


class UserPluginUpdateModel(UserPluginBaseModel):
    name: Optional[str] = None
    creator: Optional[str] = None
    code: Optional[str] = None
    create_at: Optional[int] = None  # 覆盖为可选字段
    update_at: Optional[int] = None
    
    # Set value for update_at only
    @model_validator(mode="before")
    @classmethod
    def set_update_at(cls, data):
        now = int(time.time() * 1000)
        if "update_at" not in data:
            data["update_at"] = now
        return data
