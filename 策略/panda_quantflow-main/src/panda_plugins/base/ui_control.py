from typing import Type
import logging

from functools import wraps
from pydantic import BaseModel

def ui(**kwargs):
    """
    类装饰器，为 InputModel 的 Pydantic 模型的特定字段添加UI元数据以控制工作节点的UI样式.    
    
    Decorator for adding UI metadata to the specific fields of the Pydantic model of InputModel to control the UI style of the work node.
    """
    
    def decorator(cls: Type[BaseModel]) -> Type[BaseModel]:
        original_schema_method = cls.model_json_schema
        
        @wraps(original_schema_method)
        def updated_schema(*args, **schema_kwargs):
            schema = original_schema_method(*args, **schema_kwargs)
            
            for field_name, ui_options in kwargs.items():
                if 'properties' in schema and field_name in schema['properties']:
                    props = schema['properties'][field_name]
                    props['ui'] = {**props.get('ui', {}), **ui_options}
            
            return schema
        
        cls.model_json_schema = updated_schema
        
        return cls
    
    return decorator
