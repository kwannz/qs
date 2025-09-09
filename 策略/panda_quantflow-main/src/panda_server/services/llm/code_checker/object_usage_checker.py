import ast
from typing import List, Dict, Union, Set, Optional


class ObjectUsageChecker:
    """
    检查对象的使用是否合法
    Check whether the usage of objects is legal
    """

    def __init__(self, arg_name: str, allowed_attrs: List[str], allowed_methods: List[str], global_custom_attributes: Optional[Set[str]] = None):
        self.arg_name = arg_name
        self.allowed_attributes = set(allowed_attrs)
        self.allowed_methods = set(allowed_methods)
        self.violations = []
        
        # 如果提供了全局自定义属性，将其添加到允许的属性中
        if global_custom_attributes:
            self.allowed_attributes.update(global_custom_attributes)

    def check(self, function_node: ast.FunctionDef | ast.AsyncFunctionDef):
        method_call_positions = set()
        assignment_targets = set()

        # 收集所有赋值操作的左边属性，并动态添加到允许的属性中
        for node in ast.walk(function_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == self.arg_name:
                        assignment_targets.add((target.lineno, target.attr))
                        # 动态添加用户自定义的属性到允许列表中
                        self.allowed_attributes.add(target.attr)

        # 收集所有方法调用的位置
        for node in ast.walk(function_node):
            if isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == self.arg_name
                ):

                    method_name = node.func.attr
                    method_call_positions.add((node.func.lineno, method_name))

                    # 检查方法是否合法
                    if method_name not in self.allowed_methods:
                        self.violations.append(
                            {
                                "type": "illegal_method",
                                "name": f"{self.arg_name}.{method_name}()",
                                "line": node.lineno,
                                "description": f"Illegal method used: {method_name}",
                            }
                        )

        # 检查属性访问（排除方法调用和赋值操作的属性访问）
        for node in ast.walk(function_node):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == self.arg_name:

                    attr_name = node.attr
                    # 只检查不是方法调用一部分且不是赋值操作左边的属性访问
                    if (node.lineno, attr_name) not in method_call_positions and (node.lineno, attr_name) not in assignment_targets:
                        if attr_name not in self.allowed_attributes:
                            self.violations.append(
                                {
                                    "type": "illegal_attribute",
                                    "name": f"{self.arg_name}.{attr_name}",
                                    "line": node.lineno,
                                    "description": f"Illegal attribute used: {attr_name}",
                                }
                            ) 