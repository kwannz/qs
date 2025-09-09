from pydantic import BaseModel, Field

class FeatureTagDetailModel(BaseModel):
    """
    Feature Tag Detail Model
    """
    node_id: str = Field(description="Node ID")
    node_title: str = Field(description="Node Title")