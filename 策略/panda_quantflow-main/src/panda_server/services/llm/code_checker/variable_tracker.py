import ast
from typing import List, Dict, Union, Any


class VirableTracker:
    """
    追踪变量在函数间的传递和使用
    Track the passing and usage of variables across functions
    """
    
    def __init__(self, tree: ast.Module, arg_name_or_mapping: Union[str, Dict[str, str]], allowed_attrs: List[str], allowed_methods: List[str], allow_custom_attributes: bool = True):
        self.tree = tree
        # 支持传入单个参数名（向后兼容）或函数名到参数名的映射
        if isinstance(arg_name_or_mapping, str):
            self.arg_name = arg_name_or_mapping
            self.function_arg_mapping = None
        else:
            self.arg_name = None
            self.function_arg_mapping = arg_name_or_mapping
            
        self.allowed_attributes = set(allowed_attrs)
        self.allow_custom_attributes = allow_custom_attributes
        self.allowed_methods = set(allowed_methods)
        self.violations = []
        self.function_map = {}  # 函数名到函数节点的映射
        self.variable_params = {}  # 记录哪些函数的哪些参数是被追踪的变量
        
        # 构建函数映射
        self._build_function_map()
    
    def _build_function_map(self):
        """构建函数名到函数节点的映射"""
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.function_map[node.name] = node
    
    def analyze_from_functions(self, start_functions: List[Union[ast.FunctionDef, ast.AsyncFunctionDef]]):
        """从指定的多个函数开始分析变量的传递"""
        # Import here to avoid circular import
        from .object_usage_checker import ObjectUsageChecker
        
        # 1. 从所有起始函数开始，标记变量
        for start_function in start_functions:
            if self.function_arg_mapping:
                # 使用函数名到参数名的映射
                param_name = self.function_arg_mapping.get(start_function.name)
                if param_name:
                    self.variable_params[start_function.name] = {param_name}
            else:
                # 使用单一参数名（向后兼容）
                self.variable_params[start_function.name] = {self.arg_name}
        
        # 2. 分析函数调用图，追踪变量传递
        for start_function in start_functions:
            self._trace_reference_flow(start_function)
        
        # 3. 在开始检查前，先收集所有函数中的自定义属性（全局共享）
        global_custom_attributes = set()
        if self.allow_custom_attributes:
            global_custom_attributes = self._collect_all_custom_attributes()
            # 将全局自定义属性添加到允许属性列表中
            self.allowed_attributes.update(global_custom_attributes)
        
        # 4. 检查所有可能使用变量的函数
        for func_name, param_names in self.variable_params.items():
            if func_name in self.function_map:
                function_node = self.function_map[func_name]
                for param_name in param_names:
                    checker = ObjectUsageChecker(
                        param_name, 
                        self.allowed_attributes, 
                        self.allowed_methods,
                        global_custom_attributes=global_custom_attributes if self.allow_custom_attributes else None
                    )
                    checker.check(function_node)
                    # 添加函数名信息
                    for violation in checker.violations:
                        violation['function'] = func_name
                    self.violations.extend(checker.violations)
        
        return self.violations
    
    def analyze_from_function(self, start_function):
        """从指定函数开始分析变量的传递（保持向后兼容）"""
        return self.analyze_from_functions([start_function])
    
    def _trace_reference_flow(self, function_node):
        """追踪变量在函数间的传递"""
        # 在当前函数中查找函数调用
        for node in ast.walk(function_node):
            if isinstance(node, ast.Call):
                # 检查是否是函数调用
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    # 方法调用暂不处理
                    continue
                
                if func_name and func_name in self.function_map:
                    # 分析传递给被调用函数的参数
                    self._analyze_function_call(function_node.name, func_name, node)
    
    def _analyze_function_call(self, caller_func, callee_func, call_node):
        """分析函数调用，确定变量是否被传递"""
        callee_function = self.function_map[callee_func]
        caller_variable_params = self.variable_params.get(caller_func, set())
        
        # 检查每个参数
        for i, arg in enumerate(call_node.args):
            if i < len(callee_function.args.args):
                param_name = callee_function.args.args[i].arg
                
                # 检查参数是否是变量
                if isinstance(arg, ast.Name) and arg.id in caller_variable_params:
                    # 这个参数是变量，记录到被调用函数
                    if callee_func not in self.variable_params:
                        self.variable_params[callee_func] = set()
                    self.variable_params[callee_func].add(param_name)
                    
                    # 递归分析被调用函数
                    if callee_func != caller_func:  # 避免无限递归
                        self._trace_reference_flow(callee_function)

    def _collect_all_custom_attributes(self):
        """收集所有可能使用到的函数中定义的自定义属性（全局共享）"""
        all_custom_attributes = set()
        
        # 遍历所有可能涉及的函数
        for func_name, param_names in self.variable_params.items():
            if func_name in self.function_map:
                function_node = self.function_map[func_name]
                for param_name in param_names:
                    # 收集这个函数中对指定参数的赋值操作
                    for node in ast.walk(function_node):
                        if isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == param_name:
                                    all_custom_attributes.add(target.attr)
        
        return all_custom_attributes

    def _collect_custom_attributes(self):
        """收集所有函数中定义的自定义属性（保持向后兼容，现在调用全局版本）"""
        global_custom_attributes = self._collect_all_custom_attributes()
        self.allowed_attributes.update(global_custom_attributes) 