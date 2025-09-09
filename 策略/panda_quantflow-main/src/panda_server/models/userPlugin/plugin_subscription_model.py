from bson import ObjectId
from pydantic import BaseModel, Field, model_validator
from typing import Annotated, Optional
from datetime import date
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


class PluginSubscriptionBaseModel(BaseModel):
    """
    Plugin Subscription Model
    User subscription table
    """
    
    published_plugin_id: str = Field(description="Published plugin ID")
    subscriber_id: str = Field(description="Subscriber user ID (uid)")
    startdate: date = Field(
        description="Start subscription date"
    )
    enddate: date = Field(
        description="End subscription date"
    )
    expiration: int = Field(
        description="Subscription expiration date, unix timestamp, unit: milliseconds"
    )

    # Set values for startdate if not provided
    @model_validator(mode="before")
    @classmethod
    def set_defaults(cls, data):
        from datetime import date
        if "startdate" not in data:
            data["startdate"] = date.today()
        return data

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True
        from_attributes = True
        arbitrary_types_allowed = True
        json_schema_extra = {"description": "Plugin Subscription Model"}


class PluginSubscriptionModel(PluginSubscriptionBaseModel):
    subscription_id: Annotated[
        PyObjectId,
        Field(
            default_factory=PyObjectId,
            alias="_id",
            description="Plugin subscription unique identifier, consistent with MongoDB's ObjectId",
        ),
    ]


class PluginSubscriptionCreateModel(PluginSubscriptionBaseModel):
    pass


class PluginSubscriptionUpdateModel(PluginSubscriptionBaseModel):
    subscriber_id: Optional[str] = None
    startdate: Optional[date] = None
    enddate: Optional[date] = None
    expiration: Optional[int] = None
