import ast
from typing import List, Dict, Union, Any
from .variable_tracker import VirableTracker
from .object_usage_checker import ObjectUsageChecker


class BaseCodeChecker:
    """
    基础代码检查器，提供Python代码的语法检查、结构分析和安全审查功能
    Base code checker providing Python code syntax checking, structural analysis, and security auditing

    使用方式 / Usage:
        1. 首先调用 check_syntax() 确保代码语法正确
        2. 然后使用其他方法进行具体分析

        Example:
            checker = BaseCodeChecker(code)
            if checker.check_syntax() is None:  # No syntax errors
                imports = checker.extract_imports()
                functions = checker.extract_all_function_defs()

    主要功能 / Main Features:
        - check_syntax(): 检查代码语法错误并解析为AST树
        - extract_imports(): 提取代码中的所有导入语句信息（不包含别名）
        - extract_root_module_names(): 提取导入模块的顶级名称列表
        - extract_all_function_defs(): 提取所有函数定义信息（包括类方法和嵌套函数）
        - extract_top_level_function_defs(): 提取顶层函数定义信息（不包括类方法和嵌套函数）
        - extract_all_function_calls(): 提取所有函数调用的详细信息及其上下文
        - check_function_args(): 验证指定函数的参数使用是否符合签名要求
        - check_key_object_usage(): 检查关键对象的属性和方法使用是否合规
        - danger_code_check(): 检测潜在危险的代码模式和不安全的操作
    """

    def __init__(self, code: str):
        self.code = code
        self._tree: ast.Module | None = None

    def check_syntax(self) -> str | None:
        """
        解析代码为AST树，提供详细的错误信息
        Parse code into AST tree with detailed error information
        Returns:
            None if no error, otherwise returns error message
        """
        try:
            self._tree = ast.parse(self.code)
        except SyntaxError as e:
            # Build detailed error information
            error_msg = f"Code syntax error: {e.msg}"
            if e.lineno:
                error_msg += f" (line {e.lineno}"
                if e.offset:
                    error_msg += f", column {e.offset}"
                error_msg += ")"

            # If there's error text, show specific position
            if e.text:
                error_msg += f"\nError line: {e.text.strip()}"

            return error_msg

    def extract_imports(self) -> Dict[str, List[str]]:
        """
        提取代码中的导入语句，不包含别名信息
        Extract import statements from code without alias information

        Returns:
            Dictionary containing import info:
            {
                'imports': ['module1', 'module2'],  # import module1, module2 (without 'as' aliases)
                'from_imports': [
                    {'module': 'package', 'names': ['func1', 'func2']},  # from package import func1, func2
                    {'module': 'package.submodule', 'names': ['*']}      # from package.submodule import *
                ]
            }
        """

        imports_info = {"imports": [], "from_imports": []}

        for node in ast.walk(self._tree):
            if isinstance(node, ast.Import):
                # Handle 'import module' statements
                for n in node.names:
                    imports_info["imports"].append(n.name)

            elif isinstance(node, ast.ImportFrom):
                # Handle 'from module import ...' statements
                module = node.module or ""  # Handle relative imports
                names = []

                for n in node.names:
                    names.append(n.name)

                imports_info["from_imports"].append(
                    {
                        "module": module,
                        "names": names,
                    }
                )

        return imports_info

    def extract_root_module_names(self) -> List[str]:
        """
        提取代码中导入的模块，但只保留模块的第一个部分
        Extract imported modules from code, keeping only the first part of module names

        Examples:
            - `import A.B.C` -> `A`
            - `from A.B.C import something` -> `A`
            - `import A` -> `A`
            - `from A import something` -> `A`

        Returns:
            List of unique top-level module names
        """
        module_names = set()
        import_details = self.extract_imports()
        for import_info in import_details["imports"]:
            module_names.add(import_info.split(".")[0])
        for import_info in import_details["from_imports"]:
            module_names.add(import_info["module"].split(".")[0])
        return list(module_names)

    def _parse_function_args(self, node) -> List[str]:
        """
        解析函数参数为可读的字符串列表
        Parse function arguments into readable string list
        """
        args = []

        # Handle regular arguments
        for arg in node.args.args:
            args.append(arg.arg)

        # Handle arguments with defaults
        defaults_start = len(node.args.args) - len(node.args.defaults)
        for i, default in enumerate(node.args.defaults):
            arg_index = defaults_start + i
            if arg_index < len(args):
                default_val = (
                    ast.unparse(default) if hasattr(ast, "unparse") else "default"
                )
                args[arg_index] = f"{args[arg_index]}={default_val}"

        # Handle *args
        if node.args.vararg:
            args.append(f"*{node.args.vararg.arg}")

        # Handle **kwargs
        if node.args.kwarg:
            args.append(f"**{node.args.kwarg.arg}")

        # Handle keyword-only arguments
        for kw_arg in node.args.kwonlyargs:
            args.append(kw_arg.arg)

        return args

    def _get_function_def_info(
        self, node, class_name=None, parent_function=None
    ) -> Dict[str, Any]:
        """
        从AST节点获取函数信息
        Get function information from AST node
        """
        args = self._parse_function_args(node)

        # 确定函数类型
        function_type = (
            "method" if class_name and args and args[0] == "self" else "function"
        )

        return {
            "name": node.name,
            "args": args,
            "line": node.lineno,
            "type": function_type,
            "class": class_name,
            "parent_function": parent_function,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
        }

    def extract_all_function_defs(self) -> List[Dict[str, Any]]:
        """
        提取代码中的所有函数定义及其签名（包括类方法和嵌套函数）
        Extract all function definitions and their signatures from code (including class methods and nested functions)

        Returns:
            List of dictionaries containing function info:
            [
                {
                    'name': 'function_name',
                    'args': ['arg1', 'arg2', 'arg3=default'],
                    'line': line_number,
                    'type': 'function' or 'method',
                    'class': class_name or None,
                    'parent_function': parent_function_name or None,
                    'is_async': True or False
                }
            ]
        """

        # 使用NodeVisitor模式精确地跟踪上下文
        class FunctionVisitor(ast.NodeVisitor):
            def __init__(self, checker):
                self.checker = checker
                self.functions = []
                self.class_stack = []
                self.function_stack = []

            def visit_ClassDef(self, node):
                self.class_stack.append(node.name)
                # 继续处理类的内容
                self.generic_visit(node)
                self.class_stack.pop()

            def visit_FunctionDef(self, node):
                # 获取当前上下文
                current_class = self.class_stack[-1] if self.class_stack else None
                parent_function = (
                    self.function_stack[-1] if self.function_stack else None
                )

                # 添加函数信息
                self.functions.append(
                    self.checker._get_function_def_info(
                        node, current_class, parent_function
                    )
                )

                # 处理嵌套函数
                self.function_stack.append(node.name)
                self.generic_visit(node)
                self.function_stack.pop()

            def visit_AsyncFunctionDef(self, node):
                # 与普通函数相同的处理逻辑
                self.visit_FunctionDef(node)

        # 执行访问者模式
        visitor = FunctionVisitor(self)
        visitor.visit(self._tree)
        return visitor.functions

    def extract_top_level_function_defs(self) -> List[Dict[str, Any]]:
        """
        提取代码中的顶层函数定义及其签名（不包括类方法和嵌套函数）
        Extract top-level function definitions and their signatures from code (excluding class methods and nested functions)

        Returns:
            List of dictionaries containing function info (same structure as extract_all_function_defs but only top-level)
        """
        functions = []

        # 直接遍历模块级别的节点，顶层函数就在这里
        for node in self._tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(self._get_function_def_info(node))

        return functions

    def _get_function_call_info(
        self, node, class_name=None, parent_function=None
    ) -> Dict[str, Any]:
        """
        从AST节点获取函数调用信息
        Get function call information from AST node
        """
        # 解析函数名和调用类型
        call_name = ""
        call_type = "unknown"

        if isinstance(node.func, ast.Name):
            # 普通函数调用: func()
            call_name = node.func.id
            call_type = "function"
        elif isinstance(node.func, ast.Attribute):
            # 属性调用: obj.method() 或 module.func()
            call_name = node.func.attr
            call_type = "attribute"
            # 尝试获取完整的调用链
            if isinstance(node.func.value, ast.Name):
                call_name = f"{node.func.value.id}.{call_name}"
            elif isinstance(node.func.value, ast.Attribute):
                # 处理更深层的属性调用
                full_name = self._get_attribute_chain(node.func.value)
                call_name = f"{full_name}.{call_name}"
            elif isinstance(node.func.value, ast.Call):
                # 处理链式调用，如 obj.method().another_method()
                base_name = self._get_call_chain_base(node.func.value)
                call_name = f"{base_name}.{call_name}"
            else:
                # 其他复杂表达式
                call_name = f"complex_expression.{call_name}"
        elif isinstance(node.func, ast.Subscript):
            # 下标调用: obj[key]()
            call_name = "subscript_call"
            call_type = "subscript"

        # 统计参数信息
        args_count = len(node.args)
        keyword_args = [kw.arg for kw in node.keywords if kw.arg is not None]
        has_starargs = any(isinstance(arg, ast.Starred) for arg in node.args)
        has_kwargs = any(kw.arg is None for kw in node.keywords)

        return {
            "name": call_name,
            "line": node.lineno,
            "call_type": call_type,
            "args_count": args_count,
            "keyword_args": keyword_args,
            "has_starargs": has_starargs,
            "has_kwargs": has_kwargs,
            "class": class_name,
            "parent_function": parent_function,
        }

    def _get_call_chain_base(self, node) -> str:
        """
        获取调用链的基础名称，用于处理链式调用
        Get the base name of a call chain for handling chained calls
        """
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return self._get_attribute_chain(node.func)
        else:
            return "chained_call"

    def _get_attribute_chain(self, node) -> str:
        """
        递归获取属性调用链
        Recursively get attribute call chain
        """
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self._get_attribute_chain(node.value)
            return f"{base}.{node.attr}"
        elif isinstance(node, ast.Call):
            # 处理调用结果的属性访问
            base = self._get_call_chain_base(node)
            return f"{base}()"
        else:
            return "complex_expression"

    def extract_all_function_calls(self) -> List[Dict[str, Any]]:
        """
        提取代码中的所有函数调用及其详细信息和上下文
        Extract all function calls from code with detailed information and context

        Returns:
            List of dictionaries containing function call info:
            [
                {
                    'name': 'function_name' or 'obj.method_name' or 'complex_expression.method',
                    'line': line_number,
                    'call_type': 'function' | 'attribute' | 'subscript',
                    'args_count': number_of_positional_args,
                    'keyword_args': ['arg1', 'arg2'],  # keyword argument names
                    'has_starargs': True or False,     # has *args
                    'has_kwargs': True or False,       # has **kwargs
                    'class': class_name or None,       # containing class if any
                    'parent_function': parent_function_name or None  # containing function if any
                }
            ]
        """

        # 使用NodeVisitor模式精确地跟踪上下文
        class FunctionCallVisitor(ast.NodeVisitor):
            def __init__(self, checker):
                self.checker = checker
                self.function_calls = []
                self.class_stack = []
                self.function_stack = []

            def visit_ClassDef(self, node):
                self.class_stack.append(node.name)
                # 继续处理类的内容
                self.generic_visit(node)
                self.class_stack.pop()

            def visit_FunctionDef(self, node):
                # 进入函数上下文
                self.function_stack.append(node.name)
                self.generic_visit(node)
                self.function_stack.pop()

            def visit_AsyncFunctionDef(self, node):
                # 与普通函数相同的处理逻辑
                self.visit_FunctionDef(node)

            def visit_Call(self, node):
                # 获取当前上下文
                current_class = self.class_stack[-1] if self.class_stack else None
                parent_function = (
                    self.function_stack[-1] if self.function_stack else None
                )

                # 添加函数调用信息
                self.function_calls.append(
                    self.checker._get_function_call_info(
                        node, current_class, parent_function
                    )
                )

                # 继续处理嵌套的调用
                self.generic_visit(node)

        # 执行访问者模式
        visitor = FunctionCallVisitor(self)
        visitor.visit(self._tree)
        return visitor.function_calls

    def check_function_args(
        self,
        function_name: str,
        required_args: List[str],
        optional_args: List[str] = None,
        supports_varargs: bool = False,
        supports_kwargs: bool = False,
    ) -> List[Dict[str, Union[str, int]]]:
        """
        检查指定函数的所有调用是否符合预定义的参数签名要求
        Check whether all calls to the specified function comply with predefined parameter signature requirements

        Args:
            function_name: 要检查的函数名称
            required_args: 必需的位置参数列表
            optional_args: 可选的命名参数列表，默认为空列表
            supports_varargs: 是否支持 *args，默认False
            supports_kwargs: 是否支持 **kwargs，默认False

        Returns:
            List of violations found:
            [
                {
                    'type': 'invalid_args',
                    'function': 'function_name',
                    'line': line_number,
                    'description': 'detailed description of the issue'
                }
            ]
        """
        optional_args = optional_args or []

        violations = []

        # 遍历AST查找函数调用
        for node in ast.walk(self._tree):
            if isinstance(node, ast.Call):
                # 获取函数名
                checking_function_name = (
                    node.func.id if isinstance(node.func, ast.Name) else None
                )

                if checking_function_name == function_name:
                    # 检查这个函数调用
                    violation = self._check_single_function_call(
                        node,
                        function_name,
                        required_args,
                        optional_args,
                        supports_varargs,
                        supports_kwargs,
                    )
                    if violation:
                        violations.append(violation)

        return violations

    def _check_single_function_call(
        self,
        node: ast.Call,
        function_name: str,
        required_args: List[str],
        optional_args: List[str],
        supports_varargs: bool,
        supports_kwargs: bool,
    ) -> Dict[str, Union[str, int]] | None:
        """
        检查单个函数调用是否符合签名
        Check if a single function call matches the signature
        """
        # 获取调用的参数信息
        pos_args_count = len(node.args)
        keyword_args = {}
        has_starargs = False
        has_kwargs = False

        # 分析关键字参数
        for keyword in node.keywords:
            if keyword.arg is None:
                # **kwargs
                has_kwargs = True
            else:
                keyword_args[keyword.arg] = True

        # 检查是否有 *args
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                has_starargs = True
                break

        # 如果有 *args 或 **kwargs，且函数支持，则认为可能正确
        if (has_starargs and supports_varargs) or (has_kwargs and supports_kwargs):
            return None

        # 检查位置参数数量
        max_args = len(required_args) + len(optional_args)

        # 如果支持 *args，最大参数数量无限制
        if supports_varargs:
            max_args = float("inf")

        # 计算实际提供的参数数量（位置参数 + 关键字参数，但不重复计算）
        provided_required_args = 0
        provided_optional_args = 0

        # 位置参数填充必需参数
        provided_required_args = min(pos_args_count, len(required_args))

        # 剩余位置参数填充可选参数
        remaining_pos_args = max(0, pos_args_count - len(required_args))
        provided_optional_args += min(remaining_pos_args, len(optional_args))

        # 关键字参数
        for kw_name in keyword_args:
            if kw_name in required_args:
                # 检查必需参数是否已经通过位置参数提供
                kw_index = required_args.index(kw_name)
                if kw_index < pos_args_count:
                    return {
                        "type": "invalid_args",
                        "function": function_name,
                        "line": node.lineno,
                        "description": f"Argument '{kw_name}' provided both positionally and as keyword",
                    }
                provided_required_args += 1
            elif kw_name in optional_args:
                provided_optional_args += 1
            else:
                # 未知的关键字参数
                if not supports_kwargs:
                    return {
                        "type": "invalid_args",
                        "function": function_name,
                        "line": node.lineno,
                        "description": f"Unknown keyword argument '{kw_name}'",
                    }

        # 检查是否所有必需参数都被提供
        if provided_required_args < len(required_args):
            missing_args = required_args[provided_required_args:]
            return {
                "type": "invalid_args",
                "function": function_name,
                "line": node.lineno,
                "description": f"Missing required arguments: {', '.join(missing_args)}",
            }

        # 检查参数总数是否超出限制
        total_provided = pos_args_count + len(keyword_args)
        if not supports_varargs and not supports_kwargs and total_provided > max_args:
            return {
                "type": "invalid_args",
                "function": function_name,
                "line": node.lineno,
                "description": f"Too many arguments, expected at most {max_args}, got {total_provided}",
            }

        return None

    def check_key_object_usage(
        self,
        in_function: Union[str, List[str]],
        arg_name: str | None = None,
        allowed_attributes: List[str] | None = None,
        allowed_methods: List[str] | None = None,
        allow_custom_attributes: bool = True,
        track_across_functions: bool = False,
        arg_index: int | None = None,
    ) -> List[Dict[str, Union[str, int]]]:
        """
        检查代码中指定函数内关键对象的方法和属性使用是否合法
        Check whether the methods and attributes of key objects in specified function are legally used

        Args:
            in_function: 要检查的函数名(支持字符串或字符串列表) (e.g., "initialize" or ["initialize", "handle_data"])
            arg_name: 关键对象的参数名 (e.g., "context")，当为None时必须提供arg_index
            allowed_attributes: 允许使用的属性列表
            allowed_methods: 允许使用的方法列表
            allow_custom_attributes: 是否允许代码向对象添加自定义属性，默认True
            track_across_functions: 是否追踪跨函数传递，默认False
            arg_index: 关键对象在函数参数中的索引位置（从0开始），当arg_name为None时使用

        Returns:
            List of illegal usage found:
            [
                {
                    'type': 'illegal_attribute' or 'illegal_method',
                    'name': 'attribute_or_method_name',
                    'line': line_number,
                    'function': 'function_name',
                    'description': 'description of the issue'
                }
            ]
        """

        # 参数验证
        if arg_name is None and arg_index is None:
            raise ValueError("Either arg_name or arg_index must be provided")

        if arg_name is not None and arg_index is not None:
            raise ValueError(
                "Cannot provide both arg_name and arg_index at the same time"
            )

        if allowed_attributes is None:
            allowed_attributes = []

        if allowed_methods is None:
            allowed_methods = []

        # 标准化输入为列表
        if isinstance(in_function, str):
            function_names = [in_function]
        else:
            function_names = in_function

        # 查找指定的函数并解析参数名
        target_functions = []
        function_arg_mapping = {}  # 存储函数名到实际参数名的映射

        for node in self._tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name in function_names:
                    actual_arg_name = None

                    if arg_name is not None:
                        # 使用参数名模式
                        has_arg = any(arg.arg == arg_name for arg in node.args.args)
                        if has_arg:
                            actual_arg_name = arg_name
                    else:
                        # 使用参数索引模式
                        if arg_index < len(node.args.args):
                            actual_arg_name = node.args.args[arg_index].arg

                    if actual_arg_name:
                        target_functions.append(node)
                        function_arg_mapping[node.name] = actual_arg_name

        if not target_functions:
            return []

        if track_across_functions:
            # 使用跨函数追踪分析器
            tracker = VirableTracker(
                self._tree,
                function_arg_mapping,  # 传递函数名到参数名的映射
                allowed_attributes,
                allowed_methods,
                allow_custom_attributes=allow_custom_attributes,
            )
            return tracker.analyze_from_functions(target_functions)
        else:
            # 检查多个函数，但不追踪跨函数
            all_violations = []

            # 如果允许自定义属性，先全局收集所有自定义属性
            global_custom_attributes = set()
            if allow_custom_attributes:
                for func_node in target_functions:
                    current_arg_name = function_arg_mapping[func_node.name]
                    for node in ast.walk(func_node):
                        if isinstance(node, ast.Assign):
                            for target in node.targets:
                                if (
                                    isinstance(target, ast.Attribute)
                                    and isinstance(target.value, ast.Name)
                                    and target.value.id == current_arg_name
                                ):
                                    global_custom_attributes.add(target.attr)

            for func_node in target_functions:
                current_arg_name = function_arg_mapping[func_node.name]
                usage_checker = ObjectUsageChecker(
                    current_arg_name,
                    allowed_attributes,
                    allowed_methods,
                    global_custom_attributes=(
                        global_custom_attributes if allow_custom_attributes else None
                    ),
                )
                usage_checker.check(func_node)
                # 添加函数名信息
                for violation in usage_checker.violations:
                    violation["function"] = func_node.name
                all_violations.extend(usage_checker.violations)

            return all_violations

    def danger_code_check(self) -> List[Dict[str, Union[str, int]]]:
        """
        检查代码中是否存在危险的代码模式和不安全操作
        Check for dangerous code patterns and unsafe operations in the code

        Returns:
            List of dictionaries containing dangerous code info:
            [
                {
                    'type': 'dangerous_import' | 'dangerous_call' | 'file_write',
                    'name': 'os.system' | 'eval' | 'open(write)',
                    'line': line_number,
                    'description': 'Detailed description of the risk'
                }
            ]
        """

        dangerous_patterns = {
            # 危险的导入
            "os.system": "Can execute arbitrary system commands",
            "subprocess": "Can execute arbitrary system commands",
            "eval": "Can execute arbitrary Python code",
            "exec": "Can execute arbitrary Python code",
            "pickle.loads": "Can execute arbitrary Python code",
            "yaml.load": "Can execute arbitrary Python code unless using yaml.safe_load",
            "marshal.loads": "Can execute arbitrary Python code",
            "shutil.rmtree": "Can delete entire directory trees",
            "os.remove": "Can delete files",
            "os.unlink": "Can delete files",
            "__import__": "Dynamically imports modules, may lead to malicious code execution",
        }

        results = []

        # 遍历AST查找危险模式
        for node in ast.walk(self._tree):
            # 检查导入语句
            if isinstance(node, ast.Import):
                for n in node.names:
                    module_name = n.name
                    if module_name in dangerous_patterns or any(
                        module_name.startswith(d + ".") for d in dangerous_patterns
                    ):
                        results.append(
                            {
                                "type": "dangerous_import",
                                "name": module_name,
                                "line": node.lineno,
                                "description": dangerous_patterns.get(
                                    module_name,
                                    "Potentially dangerous module",
                                ),
                            }
                        )

            # 检查from导入语句
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for n in node.names:
                    import_name = n.name
                    full_name = f"{module}.{import_name}" if module else import_name

                    if (
                        full_name in dangerous_patterns
                        or import_name in dangerous_patterns
                    ):
                        results.append(
                            {
                                "type": "dangerous_import",
                                "name": full_name,
                                "line": node.lineno,
                                "description": dangerous_patterns.get(
                                    full_name,
                                    dangerous_patterns.get(
                                        import_name,
                                        "Potentially dangerous import",
                                    ),
                                ),
                            }
                        )

            # 检查函数调用
            elif isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    # 直接函数调用，如 eval()
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    # 属性调用，如 os.system()
                    if isinstance(node.func.value, ast.Name):
                        func_name = f"{node.func.value.id}.{node.func.attr}"

                if func_name in dangerous_patterns:
                    results.append(
                        {
                            "type": "dangerous_call",
                            "name": func_name,
                            "line": node.lineno,
                            "description": dangerous_patterns[func_name],
                        }
                    )

                # 检查open函数的写入模式
                if hasattr(node.func, "id") and node.func.id == "open":
                    if len(node.args) >= 2:  # open有至少两个参数
                        mode_arg = node.args[1]
                        if isinstance(mode_arg, ast.Constant) and (
                            "w" in mode_arg.value or "a" in mode_arg.value
                        ):
                            results.append(
                                {
                                    "type": "file_write",
                                    "name": "open(write)",
                                    "line": node.lineno,
                                    "description": "File write operation",
                                }
                            )

        return results
