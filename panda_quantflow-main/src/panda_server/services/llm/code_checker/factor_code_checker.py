import ast
from typing import List, Dict, Union, Any
from panda_server.services.llm.code_checker.base_code_checker import BaseCodeChecker
from panda_server.services.llm.code_checker.rules.factor_code_rules import *


class FactorCodeChecker(BaseCodeChecker):
    """
    因子代码检查器
    Factor code checker
    
    因子代码要求 ref: https://www.pandaai.online/community/article/72
    """

    def __init__(self, code: str):
        """
        初始化因子代码检查器
        Initialize factor code checker
        
        Args:
            code: 因子代码
        """
        super().__init__(code)

    def complete_check(self) -> List[Dict[str, Any]]:
        """
        完整检查，包括语法检查、危险代码检查、导入检查等
        Complete check, including syntax check, dangerous code check, import check, etc.

        Returns:
            List[Dict[str, Any]]: 检查结果(错误或风险信息)
            [
                {
                    "type": "syntax_error",
                    "msg": "error message"
                },
                {
                    "type": "dangerous_code", 
                    "msg": [{"type": "...", "name": "...", "line": ..., "description": "..."}]
                }
            ]

        如果检查通过，则返回空列表
        """
        result = []
        
        # 1. 检查语法
        syntax_error = self.check_syntax()
        if syntax_error:
            result.append({"type": "syntax_error", "msg": syntax_error})
            return result

        # 2. 检查危险代码
        dangerous_violations = self.danger_code_check()
        if dangerous_violations:
            result.append({"type": "dangerous_code", "msg": dangerous_violations})

        # 3. 检查导入语句
        import_error = self.check_imports()
        if import_error:
            result.append({"type": "illegal_import", "msg": import_error})

        # 4. 检查必要的方法
        missing_method_error = self.check_mandatory_methods()
        if missing_method_error:
            result.append({"type": "missing_method", "msg": missing_method_error})

        # 5. 检查关键对象的使用
        key_object_violations = self.check_key_object_usage()
        if key_object_violations:
            all_violations = []
            for violations in key_object_violations:
                all_violations.extend(violations)
            
            if all_violations:
                result.append({"type": "illegal_key_object_usage", "msg": all_violations})

        # 6. 检查因子引擎内置方法的使用
        key_method_violations = self.check_key_method_usage()
        if key_method_violations:
            all_violations = []
            for violations in key_method_violations:
                all_violations.extend(violations)
            
            if all_violations:
                result.append({"type": "illegal_key_method_usage", "msg": all_violations})

        # 7. 检查禁用因子的使用
        forbidden_factor_error = self.check_forbidden_factors()
        if forbidden_factor_error:
            result.append({"type": "forbidden_factor_usage", "msg": forbidden_factor_error})

        return result

    def check_imports(self, mode:str="blacklist") -> str | None:
        """
        检查导入语句
        Check import statements
        
        Args:
            mode: 检查模式, 默认是 blacklist, 也可以是 whitelist
            - blacklist: 黑名单模式, 只检查非法导入, 其它导入都视为合法
            - whitelist: 白名单模式, 只检查合法导入, 其它导入都视为非法
        """
        module_names = self.extract_root_module_names()
        if mode == "blacklist":
            illegal_imports = [
                module_name
                for module_name in module_names
                if module_name in not_allowed_imports
            ]
        elif mode == "whitelist":
            illegal_imports = [
                module_name
                for module_name in module_names
                if module_name not in allowed_imports
            ]
        if illegal_imports:
            error_msg = "Not allowed imports: " + ", ".join(illegal_imports)
            return error_msg
        return None

    def check_mandatory_methods(self) -> str | None:
        """
        检查必要的方法
        Check mandatory methods
        """
        methods_info = self.extract_all_function_defs()
        method_names = [
            method_info["name"] 
            for method_info in methods_info 
            if method_info["type"] == "method"  # 只检查类方法
        ]
        
        missing_methods = [
            method_name
            for method_name in mandatory_methods
            if method_name not in method_names
        ]
        if missing_methods:
            error_msg = "Missing methods: " + ", ".join(missing_methods)
            return error_msg
        return None

    def check_key_object_usage(self) -> List[List[Dict[str, Union[str, int]]]]:
        """
        检查关键对象的使用
        Check key object usage
        
        Returns:
            List[List[Dict]]: 每个规则的违规列表
        """
        all_violations = []
        for obj_rules in key_object_rules:
            check_result = super().check_key_object_usage(
                in_function=obj_rules["in_function"],
                arg_name=obj_rules["arg_name"],
                allowed_attributes=obj_rules["allowed_attributes"],
                allowed_methods=obj_rules["allowed_methods"],
                allow_custom_attributes=obj_rules["allow_custom_attributes"],
                track_across_functions=obj_rules["track_across_functions"],
            )
            if check_result:
                all_violations.append(check_result)
        return all_violations

    def check_key_method_usage(self) -> List[List[Dict[str, Union[str, int]]]]:
        """
        检查因子引擎内置方法的使用
        Check factor engine builtin methods usage
        
        Returns:
            List[List[Dict]]: 每个方法规则的违规列表
        """
        all_violations = []
        for method_rules in key_method_rules:
            check_result = super().check_function_args(
                function_name=method_rules["function_name"],
                required_args=method_rules["required_args"],
                optional_args=method_rules["optional_args"],
                supports_varargs=method_rules["supports_varargs"],
                supports_kwargs=method_rules["supports_kwargs"],
            )
            if check_result:
                all_violations.append(check_result)
        return all_violations

    def check_forbidden_factors(self) -> str | None:
        """
        检查代码中是否使用了禁用的因子
        Check if forbidden factors are used in the code
        
        Returns:
            str | None: 错误信息，如果没有使用禁用因子则返回None
        """
        try:
            tree = ast.parse(self.code)
        except SyntaxError:
            # 如果语法错误，其他地方会处理，这里直接返回
            return None
        
        forbidden_used = []
        
        for node in ast.walk(tree):
            # 检查 factors['forbidden_factor'] 模式
            if isinstance(node, ast.Subscript):
                if (isinstance(node.value, ast.Name) and 
                    node.value.id == 'factors' and
                    isinstance(node.slice, ast.Constant)):
                    factor_name = node.slice.value
                    if factor_name in forbidden_factors:
                        forbidden_used.append(factor_name)
        
        if forbidden_used:
            unique_forbidden = list(set(forbidden_used))
            error_msg = (f"使用了不存在的因子: {', '.join(unique_forbidden)}。"
                        f"只能使用以下9个基础因子: {', '.join(base_factors)}")
            return error_msg
        
        return None 