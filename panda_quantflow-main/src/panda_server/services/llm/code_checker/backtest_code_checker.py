import ast
from typing import List, Dict, Union, Any
from panda_server.services.llm.code_checker.base_code_checker import BaseCodeChecker
from panda_server.services.llm.code_checker.rules.backtest_code_rules import *


class BacktestCodeChecker(BaseCodeChecker):
    """
    回测代码检查器
    Backtest code checker
    
    回测代码要求 ref: https://www.pandaai.online/community/article/117
    """

    def __init__(self, code: str):
        super().__init__(code)

    def complete_check(self) -> List[Dict[str, Any]]:
        """
        完整检查，包括语法检查、危险代码检查、导入检查、顶级模块检查等

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

        # 4. 检查必要的函数
        missing_function_error = self.check_mandatory_functions()
        if missing_function_error:
            result.append({"type": "missing_function", "msg": missing_function_error})

        # 5. 检查关键对象的使用
        key_object_violations = self.check_key_object_usage()
        if key_object_violations:
            # 扁平化列表但保留完整信息
            all_violations = []
            for violations in key_object_violations:
                all_violations.extend(violations)
            
            if all_violations:
                result.append({"type": "illegal_key_object_usage", "msg": all_violations})

        # 6. 检查回测引擎内置方法的使用
        key_method_violations = self.check_key_method_usage()
        if key_method_violations:
            # 扁平化列表但保留完整信息
            all_violations = []
            for violations in key_method_violations:
                all_violations.extend(violations)
            
            if all_violations:
                result.append({"type": "illegal_key_method_usage", "msg": all_violations})

        # 7. 检查日志方法
        log_error = self.check_log_method()
        if log_error:
            result.append({"type": "illegal_log_method", "msg": log_error})

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

    def check_mandatory_functions(self) -> str | None:
        """
        检查必要的函数
        Check mandatory functions
        """
        functions_info = self.extract_top_level_function_defs()
        function_names = [func_info["name"] for func_info in functions_info]
        
        missing_functions = [
            function_name
            for function_name in mandatory_functions
            if function_name not in function_names
        ]
        if missing_functions:
            error_msg = "Missing functions: " + ", ".join(missing_functions)
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
        检查回测引擎内置方法的使用
        Check backtest engine builtin methods usage
        
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

    def check_log_method(self) -> str | None:
        """
        检查日志方法
        Check log method
        """
        for call in self.extract_all_function_calls():
            if call["name"] == "print" and call["call_type"] == "function":
                error_msg = "Not allowed to use print method. Use SRLogger.debug, SRLogger.info, SRLogger.warn or SRLogger.error instead."
                return error_msg
                
        return None
