import pytest
from panda_server.services.llm.code_checker.base_code_checker import BaseCodeChecker


class TestBaseCodeChecker:
    """测试 BaseCodeChecker 类的功能"""

    @classmethod
    def setup_class(cls):
        """在整个测试类开始前执行一次"""
        # 准备各种测试代码样本
        cls.syntax_error_code = """
def invalid_function(
    # 缺少闭合括号
    return "error"
"""
      
        cls.valid_code = """
import os
import pandas as pd
from typing import List, Dict
from datetime import datetime as dt

def calculate_sum(a: int, b: int = 10) -> int:
    '''计算两数之和'''
    return a + b

def process_data(data: List[Dict], *args, **kwargs):
    '''处理数据'''
    for item in data:
        print(item)
    
    def nested_function():
        '''嵌套函数'''
        pass

async def async_function(param1, param2="default"):
    '''异步函数'''
    await some_operation()

class MyClass:
    def method_in_class(self):
        '''类方法'''
        pass
    
    async def async_method(self, x):
        '''异步类方法'''
        return x * 2

def another_function():
    '''另一个函数'''
    pass
"""

        cls.function_calls_code = """
import os
import pandas as pd
from datetime import datetime

def simple_function():
    # 简单函数调用
    print("hello")
    len([1, 2, 3])
    max(1, 2, 3)
    
    # 带关键字参数的调用
    print("message", end="")
    range(1, 10, step=2)
    
    # 带*args和**kwargs的调用
    args = [1, 2, 3]
    kwargs = {"key": "value"}
    some_function(*args, **kwargs)

def attribute_calls():
    # 属性方法调用
    data = [1, 2, 3]
    data.append(4)
    data.extend([5, 6])
    
    # 模块方法调用
    os.path.join("a", "b")
    pd.DataFrame().head()
    
    # 链式调用
    result = pd.DataFrame().fillna(0).head(10)
    
    # 复杂属性调用
    datetime.now().strftime("%Y-%m-%d")

class TestClass:
    def method_with_calls(self):
        # 类方法中的函数调用
        self.helper_method()
        super().method()
        
        # 调用其他方法
        len(self.data)
        self.process_data(self.data)
    
    def helper_method(self):
        pass
    
    def process_data(self, data):
        pass

def complex_function():
    # 嵌套函数调用
    result = len(str(max([1, 2, 3])))
    
    # 下标调用
    func_dict = {"key": lambda x: x * 2}
    func_dict["key"](10)
    
    # 生成器表达式中的调用
    data = [str(x) for x in range(10)]
    
    def nested_function():
        # 嵌套函数中的调用
        print("nested")
        helper_function()
    
    nested_function()

def helper_function():
    pass
"""

        cls.key_object_tracking_code = """
def init(context):
    # 允许的属性赋值
    context.symbol = "000001"  
    
    # 自定义属性赋值 - 应该被允许
    context.my_data = []
    context.user_config = {"key": "value"}
    
    # 允许的方法调用
    context.log("开始初始化")
    
    # 读取未定义的属性 - 应该被检测为违规
    unknown1 = context.undefined_attr
    
    # 调用其他函数
    helper_function(context)

def handle_data(context):
    # 另一个入口函数 - 读取在init中定义的自定义属性应该被允许
    data = context.my_data
    config = context.user_config
    
    # 在此函数中定义新的自定义属性
    context.handle_data_flag = True
    context.processed_count = 0
    
    # 调用处理函数
    process_function(context)
    
    # 读取未定义的属性 - 应该被检测为违规
    unknown3 = context.another_undefined_attr
    
def helper_function(ctx):
    # 自定义属性赋值 - 应该被允许
    ctx.my_data_2 = 2
    ctx.calculation_result = 100
    
    # 读取允许的预定义属性 - 应该被允许
    current_time = ctx.now  
    portfolio = ctx.portfolio_dict
    
    # 读取自定义属性 - 应该被允许，因为已经在代码中定义了
    data = ctx.my_data
    config = ctx.user_config
    data_2 = ctx.my_data_2
    result = ctx.calculation_result
    
    # 非法方法调用 - 应该被检测为违规
    ctx.forbidden_method()
    
    # 读取未定义的属性 - 应该被检测为违规
    unknown2 = ctx.undefined_attribute

def process_function(ctx):
    # 读取在handle_data中定义的自定义属性 - 应该被允许（跨函数自定义属性共享）
    flag = ctx.handle_data_flag
    count = ctx.processed_count
    
    # 读取在init中定义的自定义属性 - 也应该被允许
    data = ctx.my_data
    
    # 定义新的自定义属性
    ctx.process_result = "success"
    
def independent_function(context):
    # 这个不会被追踪到，因为没有从init或handle_data调用
    context.some_method()
    
    # 在独立函数中定义的自定义属性
    context.independent_attr = "test"

def function_with_index_params(service_context, user_data, config_params):
    # 测试通过索引定位参数的函数
    service_context.service_name = "test_service"
    service_context.initialize()
    
    # 读取未定义的属性 - 应该被检测为违规
    unknown_attr = service_context.undefined_service_attr
"""

        cls.dangerous_code = """
import os
import subprocess
import pickle

def dangerous_function():
    # 危险函数调用
    os.system("ls")
    eval("print('hello')")
    exec("x = 1")
    subprocess.call(["ls"])
    pickle.loads(b"data")
    open("output.txt", "w").write("hello")  # 文件写入操作
    open("log.txt", "a").close()  # 文件追加操作
"""

    def test_check_syntax_valid_code(self):
        """测试检查有效代码的语法"""
        checker = BaseCodeChecker(self.valid_code)
        result = checker.check_syntax()
        assert result is None, "有效代码应该没有语法错误"

    def test_check_syntax_invalid_code(self):
        """测试检查无效代码的语法"""
        checker = BaseCodeChecker(self.syntax_error_code)
        result = checker.check_syntax()
        assert result is not None, "无效代码应该返回错误信息"
        assert "syntax error" in result.lower(), "错误信息应该包含语法错误"

    def test_extract_top_level_functions(self):
        """测试提取顶层函数"""
        checker = BaseCodeChecker(self.valid_code)

        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"

        functions = checker.extract_top_level_function_defs()

        # 验证返回结果
        assert isinstance(functions, list), "应该返回列表"
        assert len(functions) > 0, "应该找到函数"

        # 验证顶层函数
        function_names = [f["name"] for f in functions]
        assert "calculate_sum" in function_names, "应该包含 calculate_sum"
        assert "process_data" in function_names, "应该包含 process_data"
        assert "async_function" in function_names, "应该包含异步函数"
        assert "another_function" in function_names, "应该包含 another_function"

        # 不应该包含类方法和嵌套函数
        assert "method_in_class" not in function_names, "不应该包含类方法"
        assert "nested_function" not in function_names, "不应该包含嵌套函数"

        # 验证函数信息结构
        for func in functions:
            assert "name" in func, "应该包含函数名"
            assert "args" in func, "应该包含参数信息"
            assert "line" in func, "应该包含行号"
            assert "type" in func, "应该包含类型"
            assert "is_async" in func, "应该包含异步标识"

        # 验证特定函数的信息
        calculate_sum = next(f for f in functions if f["name"] == "calculate_sum")
        assert calculate_sum["args"] == ["a", "b=10"], "参数信息应该正确"
        assert calculate_sum["type"] == "function", "应该是函数类型"
        assert not calculate_sum["is_async"], "不应该是异步函数"

        async_func = next(f for f in functions if f["name"] == "async_function")
        assert async_func["is_async"], "应该是异步函数"

    def test_extract_all_functions(self):
        """测试提取所有函数（包括类方法和嵌套函数）"""
        checker = BaseCodeChecker(self.valid_code)

        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"

        all_functions = checker.extract_all_function_defs()

        # 验证返回结果
        assert isinstance(all_functions, list), "应该返回列表"
        assert len(all_functions) > 0, "应该找到函数"

        function_names = [f["name"] for f in all_functions]

        # 应该包含所有函数
        assert "calculate_sum" in function_names, "应该包含顶层函数"
        assert "method_in_class" in function_names, "应该包含类方法"
        assert "async_method" in function_names, "应该包含异步类方法"
        assert "nested_function" in function_names, "应该包含嵌套函数"

        # 验证类方法信息
        class_method = next(f for f in all_functions if f["name"] == "method_in_class")
        assert class_method["type"] == "method", "应该是方法类型"
        assert class_method["class"] == "MyClass", "应该属于正确的类"

        # 验证嵌套函数信息
        nested_func = next(f for f in all_functions if f["name"] == "nested_function")
        assert nested_func["parent_function"] == "process_data", "应该有正确的父函数"

    def test_extract_imports(self):
        """测试提取导入语句"""
        checker = BaseCodeChecker(self.valid_code)

        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"

        imports = checker.extract_imports()

        # 验证返回结果结构
        assert isinstance(imports, dict), "应该返回字典"
        assert "imports" in imports, "应该包含 imports 键"
        assert "from_imports" in imports, "应该包含 from_imports 键"

        # 验证 import 语句
        import_list = imports["imports"]
        assert isinstance(import_list, list), "imports 应该是列表"
        assert "os" in import_list, "应该包含 os"
        assert "pandas" in import_list, "应该包含 pandas（不带as别名）"

        # 验证 from import 语句
        from_imports = imports["from_imports"]
        assert isinstance(from_imports, list), "from_imports 应该是列表"

        # 查找特定的 from import
        typing_import = next(
            (imp for imp in from_imports if imp["module"] == "typing"), None
        )
        assert typing_import is not None, "应该找到 typing 导入"
        assert "List" in typing_import["names"], "应该包含 List"
        assert "Dict" in typing_import["names"], "应该包含 Dict"

        datetime_import = next(
            (imp for imp in from_imports if imp["module"] == "datetime"), None
        )
        assert datetime_import is not None, "应该找到 datetime 导入"
        assert "datetime" in datetime_import["names"], "应该包含 datetime（不带as别名）"

    def test_extract_root_module_names(self):
        """测试提取根模块名"""
        checker = BaseCodeChecker(self.valid_code)

        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"

        root_modules = checker.extract_root_module_names()

        # 验证返回结果
        assert isinstance(root_modules, list), "应该返回列表"
        assert len(root_modules) > 0, "应该找到根模块"

        # 验证特定模块
        assert "os" in root_modules, "应该包含 os"
        assert "pandas" in root_modules, "应该包含 pandas"
        assert "typing" in root_modules, "应该包含 typing"
        assert "datetime" in root_modules, "应该包含 datetime"

    def test_danger_code_check(self):
        """测试危险代码检查"""
        checker = BaseCodeChecker(self.dangerous_code)

        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"

        dangers = checker.danger_code_check()

        # 验证返回结果
        assert isinstance(dangers, list), "应该返回列表"
        assert len(dangers) > 0, "应该找到危险代码"

        # 验证危险代码信息结构
        for danger in dangers:
            assert "type" in danger, "应该包含类型"
            assert "name" in danger, "应该包含名称"
            assert "line" in danger, "应该包含行号"
            assert "description" in danger, "应该包含描述"

        # 验证特定危险代码
        danger_names = [d["name"] for d in dangers]
        assert "os.system" in danger_names, "应该检测到 os.system"
        assert "eval" in danger_names, "应该检测到 eval"
        assert "exec" in danger_names, "应该检测到 exec"
        assert (
            "subprocess" in danger_names or "subprocess.call" in danger_names
        ), "应该检测到 subprocess"

        # 验证文件写入检测
        file_write_dangers = [d for d in dangers if d["type"] == "file_write"]
        assert len(file_write_dangers) > 0, "应该检测到文件写入操作"

        # 验证检测到的文件写入操作
        file_write_names = [d["name"] for d in file_write_dangers]
        assert "open(write)" in file_write_names, "应该检测到文件写入操作"

    def test_check_key_object_usage_single_function(self):
        """测试检查关键对象使用（单函数模式）"""
        checker = BaseCodeChecker(self.key_object_tracking_code)

        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"

        # 定义允许的属性和方法
        allowed_attributes = ["symbol"]
        allowed_methods = ["log"]

        # 测试单函数检查（不追踪跨函数）
        violations = checker.check_key_object_usage(
            in_function="init",
            arg_name="context",
            allowed_attributes=allowed_attributes,
            allowed_methods=allowed_methods,
            track_across_functions=False,
        )

        # 验证返回结果
        assert isinstance(violations, list), "应该返回列表"
        # 在 init 函数中有一个违规：context.undefined_attr
        assert (
            len(violations) == 1
        ), f"单函数模式下应该有1个违规，实际有{len(violations)}个: {violations}"

        # 验证违规内容
        assert (
            violations[0]["type"] == "illegal_attribute"
        ), "违规类型应该是illegal_attribute"
        assert (
            violations[0]["name"] == "context.undefined_attr"
        ), "违规名称应该是context.undefined_attr"

    def test_check_key_object_usage_cross_function(self):
        """测试检查关键对象使用（跨函数追踪模式 + 自定义属性功能）"""
        checker = BaseCodeChecker(self.key_object_tracking_code)

        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"

        # 定义允许的属性和方法
        allowed_attributes = [
            "symbol",
            "now",
            "portfolio_dict",
            "stock_account_dict",
            "future_account_dict",
        ]
        allowed_methods = ["log"]

        # 测试跨函数追踪
        violations = checker.check_key_object_usage(
            in_function="init",
            arg_name="context",
            allowed_attributes=allowed_attributes,
            allowed_methods=allowed_methods,
            track_across_functions=True,
        )

        # 验证返回结果
        assert isinstance(violations, list), "应该返回列表"
        assert (
            len(violations) == 3
        ), f"应该有3个违规，实际有{len(violations)}个: {violations}"

        # 验证违规信息结构
        for violation in violations:
            assert "type" in violation, "应该包含类型"
            assert "name" in violation, "应该包含名称"
            assert "line" in violation, "应该包含行号"
            assert "function" in violation, "应该包含函数名"
            assert "description" in violation, "应该包含描述"

        # 验证特定违规
        violation_names = [v["name"] for v in violations]
        assert (
            "context.undefined_attr" in violation_names
        ), "应该检测到context.undefined_attr"
        assert (
            "ctx.undefined_attribute" in violation_names
        ), "应该检测到ctx.undefined_attribute"
        assert "ctx.forbidden_method()" in violation_names, "应该检测到非法方法"

        # 验证违规发生在正确的函数中
        violation_functions = [v["function"] for v in violations]
        assert "init" in violation_functions, "违规应该发生在 init 中"
        assert (
            "helper_function" in violation_functions
        ), "违规应该发生在 helper_function 中"

        # 验证违规类型
        violation_types = [v["type"] for v in violations]
        assert "illegal_attribute" in violation_types, "应该包含非法属性类型"
        assert "illegal_method" in violation_types, "应该包含非法方法类型"

    def test_check_key_object_usage_multiple_functions(self):
        """测试检查关键对象使用（多函数入口模式）"""
        checker = BaseCodeChecker(self.key_object_tracking_code)

        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"

        # 定义允许的属性和方法
        allowed_attributes = [
            "symbol",
            "now",
            "portfolio_dict",
            "stock_account_dict",
            "future_account_dict",
        ]
        allowed_methods = ["log"]

        # 测试多函数入口，不追踪跨函数
        violations = checker.check_key_object_usage(
            in_function=["init", "handle_data"],
            arg_name="context",
            allowed_attributes=allowed_attributes,
            allowed_methods=allowed_methods,
            track_across_functions=False,
        )

        # 验证返回结果
        assert isinstance(violations, list), "应该返回列表"
        # init 函数中有 1 个违规，handle_data 函数中有 1 个违规
        assert (
            len(violations) == 2
        ), f"多函数模式下应该有2个违规，实际有{len(violations)}个: {violations}"

        # 验证违规发生在正确的函数中
        violation_functions = [v["function"] for v in violations]
        assert "init" in violation_functions, "违规应该发生在 init 中"
        assert "handle_data" in violation_functions, "违规应该发生在 handle_data 中"

        # 验证特定违规
        violation_names = [v["name"] for v in violations]
        assert (
            "context.undefined_attr" in violation_names
        ), "应该检测到context.undefined_attr"
        assert (
            "context.another_undefined_attr" in violation_names
        ), "应该检测到context.another_undefined_attr"

    def test_check_key_object_usage_multiple_functions_cross_tracking(self):
        """测试检查关键对象使用（多函数入口 + 跨函数追踪 + 自定义属性跨函数共享）"""
        checker = BaseCodeChecker(self.key_object_tracking_code)

        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"

        # 定义允许的属性和方法
        allowed_attributes = [
            "symbol",
            "now",
            "portfolio_dict",
            "stock_account_dict",
            "future_account_dict",
        ]
        allowed_methods = ["log"]

        # 测试多函数入口 + 跨函数追踪
        violations = checker.check_key_object_usage(
            in_function=["init", "handle_data"],
            arg_name="context",
            allowed_attributes=allowed_attributes,
            allowed_methods=allowed_methods,
            track_across_functions=True,
        )

        # 验证返回结果
        assert isinstance(violations, list), "应该返回列表"
        # 应该有 4 个违规：
        # - init 中的 context.undefined_attr
        # - handle_data 中的 context.another_undefined_attr  
        # - helper_function 中的 ctx.forbidden_method() 和 ctx.undefined_attribute
        assert (
            len(violations) == 4
        ), f"多函数跨追踪模式下应该有4个违规，实际有{len(violations)}个: {violations}"

        # 验证违规发生在正确的函数中
        violation_functions = [v["function"] for v in violations]
        assert "init" in violation_functions, "违规应该发生在 init 中"
        assert "handle_data" in violation_functions, "违规应该发生在 handle_data 中"
        assert "helper_function" in violation_functions, "违规应该发生在 helper_function 中"
        # process_function 中没有违规，因为它只使用了已定义的自定义属性

        # 验证特定违规
        violation_names = [v["name"] for v in violations]
        assert (
            "context.undefined_attr" in violation_names
        ), "应该检测到context.undefined_attr"
        assert (
            "context.another_undefined_attr" in violation_names
        ), "应该检测到context.another_undefined_attr"
        assert "ctx.forbidden_method()" in violation_names, "应该检测到非法方法"
        assert (
            "ctx.undefined_attribute" in violation_names
        ), "应该检测到ctx.undefined_attribute"

    def test_check_key_object_usage_with_arg_index(self):
        """测试使用参数索引定位关键对象"""
        checker = BaseCodeChecker(self.key_object_tracking_code)

        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"

        # 定义允许的属性和方法
        allowed_attributes = ["service_name"]
        allowed_methods = ["initialize"]

        # 使用参数索引模式（第一个参数，索引为0）
        violations = checker.check_key_object_usage(
            in_function="function_with_index_params",
            arg_index=0,  # 第一个参数是 service_context
            allowed_attributes=allowed_attributes,
            allowed_methods=allowed_methods,
            track_across_functions=False,
        )

        # 验证返回结果
        assert isinstance(violations, list), "应该返回列表"
        assert (
            len(violations) == 1
        ), f"应该有1个违规，实际有{len(violations)}个: {violations}"

        # 验证违规内容
        assert (
            violations[0]["type"] == "illegal_attribute"
        ), "违规类型应该是illegal_attribute"
        assert (
            violations[0]["name"] == "service_context.undefined_service_attr"
        ), "违规名称应该是service_context.undefined_service_attr"
        assert (
            violations[0]["function"] == "function_with_index_params"
        ), "违规应该发生在function_with_index_params函数中"

    def test_check_key_object_usage_arg_validation(self):
        """测试参数验证"""
        checker = BaseCodeChecker(self.key_object_tracking_code)

        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"

        # 测试既不提供arg_name也不提供arg_index的情况
        with pytest.raises(ValueError, match="Either arg_name or arg_index must be provided"):
            checker.check_key_object_usage(
                in_function="init",
                allowed_attributes=["symbol"],
                allowed_methods=["log"]
            )

        # 测试同时提供arg_name和arg_index的情况
        with pytest.raises(ValueError, match="Cannot provide both arg_name and arg_index at the same time"):
            checker.check_key_object_usage(
                in_function="init",
                arg_name="context",
                arg_index=0,
                allowed_attributes=["symbol"],
                allowed_methods=["log"]
            )

    def test_check_key_object_usage_multiple_functions_with_arg_index(self):
        """测试多函数入口使用参数索引"""
        checker = BaseCodeChecker(self.key_object_tracking_code)

        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"

        # 定义允许的属性和方法
        allowed_attributes = ["symbol", "service_name"]
        allowed_methods = ["log", "initialize"]

        # 使用参数索引模式检查多个函数
        violations = checker.check_key_object_usage(
            in_function=["init", "function_with_index_params"],
            arg_index=0,  # 对所有函数都使用第一个参数
            allowed_attributes=allowed_attributes,
            allowed_methods=allowed_methods,
            track_across_functions=False,
        )

        # 验证返回结果
        assert isinstance(violations, list), "应该返回列表"
        assert (
            len(violations) == 2
        ), f"应该有2个违规，实际有{len(violations)}个: {violations}"

        # 验证违规发生在正确的函数中
        violation_functions = [v["function"] for v in violations]
        assert "init" in violation_functions, "违规应该发生在 init 中"
        assert "function_with_index_params" in violation_functions, "违规应该发生在 function_with_index_params 中"

        # 验证违规内容
        violation_names = [v["name"] for v in violations]
        assert (
            "context.undefined_attr" in violation_names
        ), "应该检测到context.undefined_attr"
        assert (
            "service_context.undefined_service_attr" in violation_names
        ), "应该检测到service_context.undefined_service_attr"

    def test_check_key_object_usage_function_not_found(self):
        """测试检查关键对象是否调用了不存在的方法"""
        checker = BaseCodeChecker(self.key_object_tracking_code)

        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"

        # 测试不存在的函数
        violations = checker.check_key_object_usage(
            in_function="nonexistent_function",
            arg_name="context",
            allowed_attributes=["symbol"],
            allowed_methods=["log"],
            track_across_functions=False,
        )

        # 应该返回空列表
        assert isinstance(violations, list), "应该返回列表"
        assert len(violations) == 0, "不存在的函数应该返回空列表"

    def test_check_key_object_usage_parameter_not_found(self):
        """测试检查函数中不存在的参数"""
        checker = BaseCodeChecker(self.key_object_tracking_code)

        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"

        # 测试不存在的参数
        violations = checker.check_key_object_usage(
            in_function="init",
            arg_name="nonexistent_param",
            allowed_attributes=["symbol"],
            allowed_methods=["log"],
            track_across_functions=False,
        )

        # 应该返回空列表
        assert isinstance(violations, list), "应该返回列表"
        assert len(violations) == 0, "不存在的参数应该返回空列表"

    def test_empty_code(self):
        """测试空代码"""
        checker = BaseCodeChecker("")

        # 语法检查应该通过
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "空代码应该语法正确"

        # 各种提取功能应该返回空结果
        functions = checker.extract_top_level_function_defs()
        assert len(functions) == 0, "空代码应该没有函数"

        all_functions = checker.extract_all_function_defs()
        assert len(all_functions) == 0, "空代码应该没有函数"

        function_calls = checker.extract_all_function_calls()
        assert len(function_calls) == 0, "空代码应该没有函数调用"

        imports = checker.extract_imports()
        assert len(imports["imports"]) == 0, "空代码应该没有导入"
        assert len(imports["from_imports"]) == 0, "空代码应该没有from导入"

        root_modules = checker.extract_root_module_names()
        assert len(root_modules) == 0, "空代码应该没有根模块"

        dangers = checker.danger_code_check()
        assert len(dangers) == 0, "空代码应该没有危险代码"

    def test_check_function_args_correct_usage(self):
        """测试正确的函数参数使用"""
        test_code = """
def init(context):
    # 正确的调用
    order_shares("account1", "000001", 100)
    order_shares("account1", "000001", 100, "market")
    order_shares("account1", "000001", 100, style="limit")
    order_shares(account="account1", symbol="000001", amount=100)
    order_shares(account="account1", symbol="000001", amount=100, style="market")
"""
        
        checker = BaseCodeChecker(test_code)
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"
        
        violations = checker.check_function_args(
            function_name="order_shares",
            required_args=["account", "symbol", "amount"],
            optional_args=["style"]
        )
        
        assert len(violations) == 0, f"正确的调用不应该有违规，但发现: {violations}"

    def test_check_function_args_missing_required(self):
        """测试缺少必需参数"""
        test_code = """
def init(context):
    # 缺少必需参数
    order_shares("account1")
    order_shares("account1", "000001")
"""
        
        checker = BaseCodeChecker(test_code)
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"
        
        violations = checker.check_function_args(
            function_name="order_shares",
            required_args=["account", "symbol", "amount"],
            optional_args=["style"]
        )
        
        assert len(violations) == 2, f"应该有2个违规，实际有{len(violations)}个"
        
        # 检查违规类型
        for violation in violations:
            assert violation["type"] == "invalid_args"
            assert "缺少必需参数" in violation["description"] or "Missing required arguments" in violation["description"]

    def test_check_function_args_too_many_args(self):
        """测试参数过多"""
        test_code = """
def init(context):
    # 参数过多
    order_shares("account1", "000001", 100, "market", "extra1", "extra2")
"""
        
        checker = BaseCodeChecker(test_code)
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"
        
        violations = checker.check_function_args(
            function_name="order_shares",
            required_args=["account", "symbol", "amount"],
            optional_args=["style"]
        )
        
        assert len(violations) == 1, f"应该有1个违规，实际有{len(violations)}个"
        assert violations[0]["type"] == "invalid_args"
        assert "参数过多" in violations[0]["description"] or "Too many arguments" in violations[0]["description"]

    def test_check_function_args_unknown_keyword(self):
        """测试未知关键字参数"""
        test_code = """
def init(context):
    # 未知关键字参数
    order_shares("account1", "000001", 100, unknown_param="value")
    order_shares(account="account1", symbol="000001", amount=100, invalid_style="market")
"""
        
        checker = BaseCodeChecker(test_code)
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"
        
        violations = checker.check_function_args(
            function_name="order_shares",
            required_args=["account", "symbol", "amount"],
            optional_args=["style"]
        )
        
        assert len(violations) == 2, f"应该有2个违规，实际有{len(violations)}个"
        
        # 检查违规类型
        for violation in violations:
            assert violation["type"] == "invalid_args"
            assert "未知的关键字参数" in violation["description"] or "Unknown keyword argument" in violation["description"]

    def test_check_function_args_duplicate_args(self):
        """测试重复提供参数"""
        test_code = """
def init(context):
    # 重复提供参数
    order_shares("account1", "000001", 100, account="account2")
"""
        
        checker = BaseCodeChecker(test_code)
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"
        
        violations = checker.check_function_args(
            function_name="order_shares",
            required_args=["account", "symbol", "amount"],
            optional_args=["style"]
        )
        
        assert len(violations) == 1, f"应该有1个违规，实际有{len(violations)}个"
        assert violations[0]["type"] == "invalid_args"
        assert "重复提供" in violations[0]["description"] or "provided both" in violations[0]["description"]

    def test_check_function_args_with_varargs(self):
        """测试支持 *args 的函数"""
        test_code = """
def init(context):
    # 支持 *args 的函数，这些调用应该都是合法的
    log_info("message")
    log_info("message", "arg1", "arg2", "arg3")
    log_info("message", extra_param="value")
    
    args = ["arg1", "arg2"]
    log_info("message", *args)
"""
        
        checker = BaseCodeChecker(test_code)
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"
        
        violations = checker.check_function_args(
            function_name="log_info",
            required_args=["message"],
            optional_args=[],
            supports_varargs=True
        )
        
        assert len(violations) == 1, f"应该有1个违规（未知关键字参数），实际有{len(violations)}个: {violations}"
        # 应该只有 extra_param 这个未知关键字参数被检测为违规
        assert "unknown keyword argument" in violations[0]["description"].lower()

    def test_check_function_args_with_kwargs(self):
        """测试支持 **kwargs 的函数"""
        test_code = """
def init(context):
    # 支持 **kwargs 的函数，这些调用应该都是合法的
    config_func("param1", unknown_param="value", another_param="value2")
    
    kwargs = {"param1": "value1", "param2": "value2"}
    config_func("param1", **kwargs)
"""
        
        checker = BaseCodeChecker(test_code)
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"
        
        violations = checker.check_function_args(
            function_name="config_func",
            required_args=["param1"],
            optional_args=[],
            supports_kwargs=True
        )
        
        assert len(violations) == 0, f"支持**kwargs的函数不应该有违规，但发现: {violations}"

    def test_extract_simple_function_calls(self):
        """测试提取简单函数调用"""
        checker = BaseCodeChecker(self.function_calls_code)

        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"

        function_calls = checker.extract_all_function_calls()

        # 验证返回结果
        assert isinstance(function_calls, list), "应该返回列表"
        assert len(function_calls) > 0, "应该找到函数调用"

        # 验证函数调用信息结构
        for call in function_calls:
            assert "name" in call, "应该包含函数名"
            assert "line" in call, "应该包含行号"
            assert "call_type" in call, "应该包含调用类型"
            assert "args_count" in call, "应该包含参数数量"
            assert "keyword_args" in call, "应该包含关键字参数"
            assert "has_starargs" in call, "应该包含*args标识"
            assert "has_kwargs" in call, "应该包含**kwargs标识"
            assert "class" in call, "应该包含类名"
            assert "parent_function" in call, "应该包含父函数名"

        # 查找特定的简单函数调用
        call_names = [call["name"] for call in function_calls]
        assert "print" in call_names, "应该找到print函数调用"
        assert "len" in call_names, "应该找到len函数调用"
        assert "max" in call_names, "应该找到max函数调用"

        # 验证简单函数调用的详细信息
        print_calls = [call for call in function_calls if call["name"] == "print"]
        assert len(print_calls) >= 2, "应该找到多个print调用"

        # 验证带关键字参数的调用
        print_with_end = [call for call in print_calls 
                         if "end" in call["keyword_args"]]
        assert len(print_with_end) >= 1, "应该找到带end参数的print调用"

        # 验证*args和**kwargs调用
        some_function_calls = [call for call in function_calls 
                              if call["name"] == "some_function"]
        if some_function_calls:
            call = some_function_calls[0]
            assert call["has_starargs"], "应该检测到*args"
            assert call["has_kwargs"], "应该检测到**kwargs"

    def test_extract_attribute_function_calls(self):
        """测试提取属性方法调用"""
        checker = BaseCodeChecker(self.function_calls_code)

        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"

        function_calls = checker.extract_all_function_calls()
        call_names = [call["name"] for call in function_calls]

        # 验证属性方法调用
        assert "data.append" in call_names, "应该找到data.append调用"
        assert "data.extend" in call_names, "应该找到data.extend调用"

        # 验证模块方法调用
        assert "os.path.join" in call_names, "应该找到os.path.join调用"
        assert "pd.DataFrame" in call_names, "应该找到pd.DataFrame调用"

        # 验证链式调用 - 更新断言以匹配实际输出
        fillna_calls = [call for call in function_calls if "fillna" in call["name"]]
        head_calls = [call for call in function_calls if "head" in call["name"]]
        assert len(fillna_calls) >= 1, "应该找到fillna调用"
        assert len(head_calls) >= 2, "应该找到多个head调用"

        # 验证属性调用类型
        attribute_calls = [call for call in function_calls 
                          if call["call_type"] == "attribute"]
        assert len(attribute_calls) > 0, "应该有属性调用"

        # 验证复杂属性调用
        datetime_calls = [call for call in function_calls 
                         if "datetime" in call["name"] and "strftime" in call["name"]]
        assert len(datetime_calls) >= 1, "应该找到datetime相关的链式调用"

    def test_extract_function_calls_with_context(self):
        """测试提取带上下文信息的函数调用"""
        checker = BaseCodeChecker(self.function_calls_code)

        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"

        function_calls = checker.extract_all_function_calls()

        # 验证类方法中的调用上下文
        class_method_calls = [call for call in function_calls 
                             if call["class"] == "TestClass"]
        assert len(class_method_calls) > 0, "应该找到类中的函数调用"

        # 验证特定类方法中的调用
        method_calls_in_class = [call for call in function_calls 
                               if call["parent_function"] == "method_with_calls"]
        assert len(method_calls_in_class) > 0, "应该找到method_with_calls中的调用"

        # 验证self方法调用
        self_calls = [call for call in function_calls if call["name"].startswith("self.")]
        assert len(self_calls) >= 2, "应该找到self方法调用"

        # 验证super()调用
        super_calls = [call for call in function_calls if "super" in call["name"]]
        assert len(super_calls) >= 1, "应该找到super()调用"

        # 验证嵌套函数中的调用
        nested_calls = [call for call in function_calls 
                       if call["parent_function"] == "nested_function"]
        assert len(nested_calls) >= 2, "应该找到嵌套函数中的调用"

        # 验证顶层函数中的调用
        top_level_calls = [call for call in function_calls 
                          if call["parent_function"] in ["simple_function", 
                                                        "attribute_calls", 
                                                        "complex_function"]]
        assert len(top_level_calls) > 0, "应该找到顶层函数中的调用"

    def test_extract_complex_function_calls(self):
        """测试提取复杂函数调用"""
        checker = BaseCodeChecker(self.function_calls_code)

        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"

        function_calls = checker.extract_all_function_calls()

        # 验证嵌套函数调用
        # 例如: len(str(max([1, 2, 3])))
        nested_calls = [call for call in function_calls 
                       if call["parent_function"] == "complex_function"]
        call_names_in_complex = [call["name"] for call in nested_calls]
        
        assert "len" in call_names_in_complex, "应该找到len调用"
        assert "str" in call_names_in_complex, "应该找到str调用" 
        assert "max" in call_names_in_complex, "应该找到max调用"

        # 验证下标调用
        subscript_calls = [call for call in function_calls 
                          if call["call_type"] == "subscript"]
        assert len(subscript_calls) >= 1, "应该找到下标调用"

        # 验证列表推导式中的调用
        range_calls = [call for call in function_calls if call["name"] == "range"]
        str_calls = [call for call in function_calls if call["name"] == "str"]
        assert len(range_calls) >= 2, "应该找到range调用（包括列表推导式中的）"
        assert len(str_calls) >= 2, "应该找到str调用（包括列表推导式和嵌套调用中的）"

        # 验证lambda函数不被当作函数调用提取
        lambda_calls = [call for call in function_calls if "lambda" in call["name"]]
        assert len(lambda_calls) == 0, "lambda定义不应该被当作函数调用"

if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
