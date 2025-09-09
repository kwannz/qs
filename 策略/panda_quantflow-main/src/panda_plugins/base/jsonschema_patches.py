import pandas as pd
import logging

import pydantic
import pydantic_core.core_schema as core_schema

# 设置 pydantic 默认 arbitrary_types_allowed = True
pydantic.BaseModel.model_config["arbitrary_types_allowed"] = True

# 修复 dataframe 类型在 jsonschema 中的兼容的问题
def _dataframe_core_schema(cls, source_type, handler):
    # 验证时直接接受任意类型，跳过 DataFrame 验证
    return core_schema.any_schema()

def _dataframe_json_schema(cls, core_schema, handler):
    # 输出自定义 JSON Schema，type 为 dataframe
    return {"type": "dataframe"}

pd.DataFrame.__get_pydantic_core_schema__ = classmethod(_dataframe_core_schema)
pd.DataFrame.__get_pydantic_json_schema__ = classmethod(_dataframe_json_schema)
