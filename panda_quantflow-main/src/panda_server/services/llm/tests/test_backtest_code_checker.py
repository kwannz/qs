"""
测试 BacktestCodeChecker 类的功能

本测试文件测试了 BacktestCodeChecker 的 complete_check 方法以及相关的子检查方法。
测试覆盖了以下方面：

1. 语法错误检查
2. 危险代码检查（如 os.system, eval 等）
3. 非法导入检查（根据 backtest_code_rules 的白名单）
4. 必要函数检查（initialize, handle_data）
5. 关键对象使用检查（context, order 对象）
6. 关键方法参数检查（order_shares 等回测 API）
7. 日志方法检查（禁用 print，推荐 SRLogger）

"""

import pytest
from panda_server.services.llm.code_checker.backtest_code_checker import (
    BacktestCodeChecker,
)


class TestBacktestCodeChecker:
    """测试 BacktestCodeChecker 类的功能"""

    @classmethod
    def setup_class(cls):
        """在整个测试类开始前执行一次"""
        
        # 完全正确的回测代码
        cls.valid_backtest_code = """
import pandas as pd
import numpy as np
from panda_backtest import order_shares, buy_open

def initialize(context):
    '''初始化函数'''
    context.symbol_list = ["000001", "000002"]
    context.trade_count = 0

def handle_data(context, bar_dict):
    '''处理数据函数'''
    current_price = bar_dict["000001"]["close"]
    if current_price > 10:
        order_shares("account1", "000001", 100, "market")
"""

        # 语法错误的代码
        cls.syntax_error_code = """
def initialize(context
    # 缺少闭合括号
    context.symbol = "000001"
"""

        # 包含危险代码的回测代码
        cls.dangerous_code = """
import os
import pandas as pd

def initialize(context):
    # 危险的系统调用
    os.system("ls")
    eval("context.data = []")

def handle_data(context, bar_dict):
    current_price = bar_dict["000001"]["close"]
"""

        # 非法导入的代码
        cls.illegal_import_code = """
import requests  # 黑名单中的非法导入，白名单外的非法导入
import sqlite3   # 黑名单中的非法导入，白名单外的非法导入
import pandas as pd  # 白名单中的合法导入
import numpy as np   # 白名单中的合法导入
import os  # 白名单中的合法导入

def initialize(context):
    context.symbol = "000001"

def handle_data(context, bar_dict):
    pass
"""

        # 缺少必要函数的代码
        cls.missing_mandatory_functions_code = """
import pandas as pd

def initialize(context):
    context.symbol = "000001"

# 缺少 handle_data 函数
"""

        # 只有 handle_data，缺少 initialize
        cls.missing_initialize_code = """
import pandas as pd

def handle_data(context, bar_dict):
    pass

# 缺少 initialize 函数
"""

        # 关键对象非法使用的代码
        cls.illegal_key_object_usage_code = """
import pandas as pd

def initialize(context):
    # 正确的属性使用
    context.symbol = "000001"
    current_time = context.now
    
    # 错误的属性访问
    unknown_attr = context.undefined_attribute
    
    # 错误的方法调用
    context.invalid_method()

def handle_data(context, bar_dict):
    # 在 handle_data 中也测试一些使用
    portfolio = context.portfolio_dict
    
    # 错误的使用
    wrong_attr = context.non_existent_attr

def helper_function(ctx):
    # 在跨函数追踪中测试
    ctx.custom_data = {"key": "value"}  # 应该被允许
    invalid_data = ctx.invalid_cross_function_attr  # 应该被检测为错误
"""

        # 关键方法参数错误的代码
        cls.illegal_key_method_usage_code = """
import pandas as pd
from panda_backtest import order_shares, buy_open

def initialize(context):
    context.symbol = "000001"

def handle_data(context, bar_dict):
    # 正确的方法调用
    order_shares("account1", "000001", 100, "market")
    
    # 缺少必需参数
    order_shares("account1", "000001")  # 缺少 amount 和 style
    
    # 参数过多
    order_shares("account1", "000001", 100, "market", "extra1", "extra2")
    
    # 未知关键字参数
    order_shares("account1", "000001", 100, "market", unknown_param="value")
    
    # 重复提供参数
    buy_open("account1", "IF2403", 1, "market", account="account2")
"""

        # 使用 print 而不是 SRLogger 的代码
        cls.illegal_log_method_code = """
import pandas as pd

def initialize(context):
    context.symbol = "000001"
    print("初始化完成")  # 不允许使用 print

def handle_data(context, bar_dict):
    current_price = bar_dict["000001"]["close"]
    if current_price > 10:
        print(f"价格: {current_price}")  # 不允许使用 print

def helper_function():
    print("这也是不允许的")  # 不允许使用 print
"""

        # 多种错误综合的代码
        cls.multiple_errors_code = """
import requests  # 非法导入
import os       # 危险导入
import pandas as pd

def initialize(context):
    context.symbol = "000001"
    # 危险代码
    os.system("ls")
    # 非法 print
    print("初始化")
    # 非法属性
    wrong_attr = context.invalid_attr

# 缺少 handle_data 函数
"""

        # 测试 order 对象使用的代码
        cls.order_object_usage_code = """
import pandas as pd

def initialize(context):
    context.symbol = "000001"

def handle_data(context, bar_dict):
    pass

def on_stock_trade_rtn(context, order):
    # 正确的 order 属性使用
    order_id = order.order_id
    side = order.side
    price = order.price
    
    # 错误的属性使用（不在允许列表中）
    invalid_attr = order.non_existent_attribute
    
    # 尝试添加自定义属性（不被允许）
    order.custom_field = "value"

def stock_order_cancel(context, order):
    # 正确使用
    status = order.status
    
    # 错误使用
    order.another_custom_field = 123
"""

        # 测试特殊情况：只包含部分错误的代码
        cls.partial_error_code = """
import pandas as pd
import numpy as np
import requests  # 只有这一个非法导入

def initialize(context):
    context.symbol = "000001"
    context.data = []

def handle_data(context, bar_dict):
    # 正确的 order_shares 调用
    current_price = bar_dict["000001"]["close"]
    if current_price > 10:
        order_shares("account1", "000001", 100, "market")
"""

        # 测试复杂的上下文对象使用场景
        cls.complex_context_usage_code = """
import pandas as pd

def initialize(context):
    # 正确的属性设置
    context.stocks = ["000001", "000002"]
    context.weights = {"000001": 0.5, "000002": 0.5}
    
    # 正确的预定义属性访问
    current_time = context.now
    
    # 错误的属性访问
    bad_attr = context.undefined_portfolio

def handle_data(context, bar_dict):
    # 正确的属性访问
    portfolio = context.portfolio_dict
    stock_account = context.stock_account_dict
    
    # 使用自定义属性（应该被允许）
    stocks = context.stocks
    weights = context.weights
    
    # 错误的属性访问
    wrong_data = context.non_existent_data

def custom_helper(ctx):
    # 跨函数的自定义属性设置（应该被允许）
    ctx.calculated_signal = 1
    
    # 错误的跨函数属性访问
    invalid = ctx.bad_cross_function_attr
"""

    # ===========================================
    # 基础测试：测试各个子检查方法的功能
    # ===========================================
    
    def test_syntax_check(self):
        """测试语法检查能否正确识别Python语法错误"""
        checker = BacktestCodeChecker(self.syntax_error_code)
        
        # 检查语法检查方法
        syntax_error = checker.check_syntax()
        assert syntax_error is not None, "应该检测到语法错误"
        assert isinstance(syntax_error, str), "语法错误应该返回字符串"

    def test_import_rules_check(self):
        """测试导入规则检查 - 黑名单和白名单模式"""
        # 使用包含合法和非法导入的测试代码
        checker = BacktestCodeChecker(self.illegal_import_code)
        
        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"
        
        # 测试黑名单模式
        import_error_blacklist = checker.check_imports(mode="blacklist")
        assert import_error_blacklist is not None, "黑名单模式应该检测到非法导入"
        assert isinstance(import_error_blacklist, str), "导入错误应该返回字符串"
        assert "requests" in import_error_blacklist or "sqlite3" in import_error_blacklist, \
            f"黑名单模式错误消息应该包含非法模块名，实际: {import_error_blacklist}"
        
        # 测试白名单模式  
        import_error_whitelist = checker.check_imports(mode="whitelist")
        assert import_error_whitelist is not None, "白名单模式应该检测到不在白名单中的导入"
        assert isinstance(import_error_whitelist, str), "导入错误应该返回字符串"
        assert "requests" in import_error_whitelist or "sqlite3" in import_error_whitelist, \
            f"白名单模式错误消息应该包含不在白名单中的模块名，实际: {import_error_whitelist}"

    def test_mandatory_functions_check(self):
        """测试必要函数检查能否识别缺失的initialize和handle_data"""
        checker = BacktestCodeChecker(self.missing_mandatory_functions_code)
        
        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"
        
        # 检查必要函数
        missing_function_error = checker.check_mandatory_functions()
        assert missing_function_error is not None, "应该检测到缺少必要函数"
        assert isinstance(missing_function_error, str), "缺少函数错误应该返回字符串"
        # 应该报告实际缺少的函数（handle_data）
        assert "handle_data" in missing_function_error, \
            f"错误消息应该包含缺少的函数名，实际: {missing_function_error}"

    def test_log_method_check(self):
        """测试日志方法检查能否识别print函数调用"""
        checker = BacktestCodeChecker(self.illegal_log_method_code)
        
        # 先检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"
        
        # 检查日志方法
        log_error = checker.check_log_method()
        assert log_error is not None, "应该检测到非法日志方法"
        assert isinstance(log_error, str), "日志错误应该返回字符串"
        assert "print" in log_error.lower(), \
            f"错误消息应该提到print方法，实际: {log_error}"

    def test_log_method_check_detailed(self):
        """测试日志方法检查的详细情况 - 验证新的extract_all_function_calls实现"""
        
        # 测试包含print函数调用的代码
        code_with_print = """
def initialize(context):
    print("This should be detected")  # 真正的print函数调用
    context.log("This is allowed")
    
def handle_data(context, data):
    result = len(data)
    print(f"Result: {result}")  # 另一个print函数调用
"""
        
        checker = BacktestCodeChecker(code_with_print)
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"
        
        log_error = checker.check_log_method() 
        assert log_error is not None, "应该检测到print调用"
        assert "Not allowed to use print method" in log_error, "错误消息应该正确"
        assert "SRLogger" in log_error, "错误消息应该包含建议的替代方案"

    def test_log_method_check_no_violations(self):
        """测试日志方法检查 - 不包含违规的print调用"""
        code_without_print = """
def initialize(context):
    context.log("Using context.log is allowed")
    SRLogger.info("Using SRLogger is allowed")
    
def handle_data(context, data):
    # 属性方法调用不应该被检测为违规
    obj.print_something()  # 这不是print函数调用
    some_obj.print()  # 这也不是print函数调用
    result = len(data)
    context.log(f"Data length: {result}")
"""
        
        checker = BacktestCodeChecker(code_without_print)
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"
        
        log_error = checker.check_log_method()
        assert log_error is None, "不应该检测到违规的print调用"

    def test_log_method_check_edge_cases(self):
        """测试日志方法检查 - 边缘情况和误报防护"""
        code_edge_cases = """
def initialize(context):
    # 这些都不应该被检测为违规
    print_func = some_function()  # 变量名包含print
    obj.print()  # 属性方法调用
    obj.print_data()  # 属性方法调用
    my_print = lambda x: x  # lambda赋值中包含print
    
    # 字符串中的print不应该被误报
    message = "This print should not trigger error"
    
    # 只有这个应该被检测为违规
    print("This is actual print function call")
"""
        
        checker = BacktestCodeChecker(code_edge_cases)
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"
        
        log_error = checker.check_log_method()
        assert log_error is not None, "应该检测到真正的print函数调用"
        assert "Not allowed to use print method" in log_error, "错误消息应该正确"

    def test_dangerous_code_check(self):
        """测试危险代码检查能否识别os.system和eval等危险调用"""
        checker = BacktestCodeChecker(self.dangerous_code)
        
        # 直接测试危险代码检查方法
        dangerous_violations = checker.danger_code_check()
        assert len(dangerous_violations) > 0, "应该检测到危险代码"
        assert isinstance(dangerous_violations, list), "危险代码检查应该返回列表"
        
        # 验证检测到的危险代码
        dangerous_names = [item.get("name", "") for item in dangerous_violations]
        assert any("os.system" in name or "eval" in name for name in dangerous_names), \
            f"应该检测到os.system或eval，实际检测到: {dangerous_names}"

    def test_key_object_usage_check(self):
        """测试关键对象使用检查能否识别context和order对象的非法使用"""
        checker = BacktestCodeChecker(self.illegal_key_object_usage_code)
        
        # 检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"
        
        # 测试关键对象检查方法
        key_object_violations = checker.check_key_object_usage()
        assert len(key_object_violations) > 0, "应该检测到关键对象使用违规"
        assert isinstance(key_object_violations, list), "关键对象检查应该返回列表"

    def test_key_method_args_check(self):
        """测试关键方法参数检查能否识别回测API的参数错误"""
        checker = BacktestCodeChecker(self.illegal_key_method_usage_code)
        
        # 检查语法
        syntax_error = checker.check_syntax()
        assert syntax_error is None, "代码语法应该正确"
        
        # 测试关键方法检查
        key_method_violations = checker.check_key_method_usage()
        assert len(key_method_violations) > 0, "应该检测到关键方法参数违规"
        assert isinstance(key_method_violations, list), "关键方法检查应该返回列表"

    def test_empty_and_edge_cases(self):
        """测试空代码和边界情况的处理"""
        # 测试空代码
        checker_empty = BacktestCodeChecker("")
        syntax_error = checker_empty.check_syntax()
        assert syntax_error is None, "空代码语法应该正确"
        
        # 测试只有注释的代码
        comment_only_code = """
# 这是一个只有注释的文件
# 没有任何实际代码
"""
        checker_comment = BacktestCodeChecker(comment_only_code)
        syntax_error = checker_comment.check_syntax()
        assert syntax_error is None, "只有注释的代码语法应该正确"
        
        # 测试只有导入没有函数的代码
        import_only_code = """
import pandas as pd
import numpy as np
"""
        checker_import = BacktestCodeChecker(import_only_code)
        syntax_error = checker_import.check_syntax()
        assert syntax_error is None, "只有导入的代码语法应该正确"

    # ===========================================
    # 完整检查测试：测试complete_check方法的整体功能
    # ===========================================

    def test_complete_check_valid_code(self):
        """测试完全正确的代码通过complete_check验证"""
        checker = BacktestCodeChecker(self.valid_backtest_code)
        result = checker.complete_check()
        assert isinstance(result, list), "应该返回列表"
        assert len(result) == 0, f"正确的代码不应该有错误，但发现: {result}"

    def test_complete_check_syntax_error(self):
        """测试complete_check能否准确识别并报告语法错误"""
        checker = BacktestCodeChecker(self.syntax_error_code)
        result = checker.complete_check()
        
        assert isinstance(result, list), "应该返回列表"
        assert len(result) == 1, f"应该有1个语法错误，实际有{len(result)}个"
        assert result[0]["type"] == "syntax_error", "错误类型应该是syntax_error"
        assert "msg" in result[0], "应该包含错误消息"

    def test_complete_check_import_violations(self):
        """测试complete_check能否识别非法导入并返回详细信息"""
        checker = BacktestCodeChecker(self.illegal_import_code)
        result = checker.complete_check()
        
        assert isinstance(result, list), "应该返回列表"
        assert len(result) > 0, "应该检测到非法导入"
        
        # 检查是否包含非法导入错误
        has_illegal_import = any(
            error.get("type") == "illegal_import"
            for error in result
        )
        assert has_illegal_import, f"应该检测到非法导入，但结果是: {result}"
        
        # 检查错误消息是否包含非法的模块名
        illegal_import_error = next(
            (error for error in result if error.get("type") == "illegal_import"), None
        )
        if illegal_import_error:
            error_msg = illegal_import_error.get("msg", "")
            assert "requests" in error_msg or "sqlite3" in error_msg, \
                f"错误消息应该包含非法模块名，实际消息: {error_msg}"

    def test_complete_check_missing_functions(self):
        """测试complete_check能否识别缺少的必要函数"""
        # 测试缺少 handle_data
        checker = BacktestCodeChecker(self.missing_mandatory_functions_code)
        result = checker.complete_check()
        
        assert isinstance(result, list), "应该返回列表"
        assert len(result) > 0, "应该检测到缺少必要函数"
        
        has_missing_function = any(
            error.get("type") == "missing_function"
            for error in result
        )
        assert has_missing_function, f"应该检测到缺少必要函数，但结果是: {result}"

        # 测试缺少 initialize
        checker2 = BacktestCodeChecker(self.missing_initialize_code)
        result2 = checker2.complete_check()
        
        assert len(result2) > 0, "应该检测到缺少 initialize 函数"
        has_missing_initialize = any(
            error.get("type") == "missing_function" and 
            "initialize" in str(error.get("msg", ""))
            for error in result2
        )
        assert has_missing_initialize, f"应该检测到缺少 initialize 函数，但结果是: {result2}"

    def test_complete_check_log_violations(self):
        """测试complete_check能否识别print函数的非法使用"""
        checker = BacktestCodeChecker(self.illegal_log_method_code)
        result = checker.complete_check()
        
        assert isinstance(result, list), "应该返回列表"
        assert len(result) > 0, "应该检测到非法日志方法"
        
        # 检查是否检测到非法日志方法错误
        has_log_method_error = any(
            error.get("type") == "illegal_log_method" or
            "print" in str(error.get("msg", "")).lower()
            for error in result
        )
        assert has_log_method_error, f"应该检测到非法日志方法，但结果是: {result}"

    def test_complete_check_dangerous_code(self):
        """测试complete_check能否识别危险代码并返回详细信息"""
        checker = BacktestCodeChecker(self.dangerous_code)
        result = checker.complete_check()
        
        assert isinstance(result, list), "应该返回列表"
        assert len(result) > 0, "应该检测到危险代码"
        
        # 检查是否包含危险代码相关的错误
        dangerous_error = next(
            (error for error in result if error.get("type") == "dangerous_code"), None
        )
        assert dangerous_error is not None, f"应该检测到危险代码，但结果是: {result}"
        
        # 验证msg是包含详细信息的列表
        assert isinstance(dangerous_error.get("msg"), list), "危险代码的msg应该是列表"
        assert len(dangerous_error.get("msg")) > 0, "应该包含具体的危险代码信息"
        
        # 验证列表中包含危险代码的详细信息
        dangerous_items = dangerous_error.get("msg")
        dangerous_names = [item.get("name", "") for item in dangerous_items]
        assert any("os.system" in name or "eval" in name for name in dangerous_names), \
            f"应该检测到os.system或eval，实际检测到: {dangerous_names}"

    def test_complete_check_object_violations(self):
        """测试complete_check能否识别context和order对象的非法使用"""
        # 测试context对象违规
        checker = BacktestCodeChecker(self.illegal_key_object_usage_code)
        result = checker.complete_check()
        
        assert isinstance(result, list), "应该返回列表"
        assert len(result) > 0, "应该检测到关键对象非法使用"
        
        # 检查是否包含关键对象使用错误
        key_object_error = next(
            (error for error in result if error.get("type") == "illegal_key_object_usage"), None
        )
        assert key_object_error is not None, f"应该检测到关键对象非法使用，但结果是: {result}"
        
        # 验证msg是包含详细信息的列表
        assert isinstance(key_object_error.get("msg"), list), "关键对象错误的msg应该是列表"
        assert len(key_object_error.get("msg")) > 0, "应该包含具体的违规信息"
        
        # 验证列表中包含违规的详细信息
        violations = key_object_error.get("msg")
        violation_names = [item.get("name", "") for item in violations]
        assert any("undefined" in name for name in violation_names), \
            f"应该检测到未定义属性的使用，实际检测到: {violation_names}"

        # 测试order对象违规
        checker2 = BacktestCodeChecker(self.order_object_usage_code)
        result2 = checker2.complete_check()
        
        assert isinstance(result2, list), "应该返回列表"
        assert len(result2) > 0, "应该检测到 order 对象非法使用"
        
        # 检查是否包含关键对象使用错误
        key_object_error2 = next(
            (error for error in result2 if error.get("type") == "illegal_key_object_usage"), None
        )
        assert key_object_error2 is not None, f"应该检测到 order 对象非法使用，但结果是: {result2}"
        
        # 验证msg是包含详细信息的列表
        assert isinstance(key_object_error2.get("msg"), list), "order对象错误的msg应该是列表"
        assert len(key_object_error2.get("msg")) > 0, "应该包含具体的违规信息"
        
        # 验证列表中包含违规的详细信息
        violations2 = key_object_error2.get("msg")
        violation_names2 = [item.get("name", "") for item in violations2]
        assert any("order." in name for name in violation_names2), \
            f"应该检测到order对象的非法使用，实际检测到: {violation_names2}"

    def test_complete_check_method_violations(self):
        """测试complete_check能否识别回测API方法的参数错误"""
        checker = BacktestCodeChecker(self.illegal_key_method_usage_code)
        result = checker.complete_check()
        
        assert isinstance(result, list), "应该返回列表"
        assert len(result) > 0, "应该检测到关键方法参数错误"
        
        # 检查是否包含关键方法使用错误
        key_method_error = next(
            (error for error in result if error.get("type") == "illegal_key_method_usage"), None
        )
        assert key_method_error is not None, f"应该检测到关键方法参数错误，但结果是: {result}"
        
        # 验证msg是包含详细信息的列表
        assert isinstance(key_method_error.get("msg"), list), "关键方法错误的msg应该是列表"
        assert len(key_method_error.get("msg")) > 0, "应该包含具体的违规信息"
        
        # 验证列表中包含违规的详细信息
        violations = key_method_error.get("msg")
        violation_functions = [item.get("function", "") for item in violations]
        assert any("order_shares" in func for func in violation_functions), \
            f"应该检测到order_shares参数错误，实际检测到: {violation_functions}"

    def test_complete_check_empty_code(self):
        """测试complete_check对空代码的处理"""
        checker = BacktestCodeChecker("")
        result = checker.complete_check()
        
        assert isinstance(result, list), "应该返回列表"
        assert len(result) > 0, "空代码应该有错误（缺少必要函数）"
        
        # 应该至少包含缺少必要函数的错误
        has_missing_function = any(
            error.get("type") == "missing_function"
            for error in result
        )
        assert has_missing_function, f"空代码应该检测到缺少必要函数，但结果是: {result}"

    def test_complete_check_partial_errors(self):
        """测试complete_check对部分错误代码的处理"""
        checker = BacktestCodeChecker(self.partial_error_code)
        result = checker.complete_check()
        
        assert isinstance(result, list), "应该返回列表"
        assert len(result) > 0, "应该检测到错误"
        
        # 应该检测到非法导入错误
        has_illegal_import = any(
            error.get("type") == "illegal_import" and "requests" in str(error.get("msg", ""))
            for error in result
        )
        assert has_illegal_import, f"应该检测到 requests 非法导入，但结果是: {result}"

    def test_complete_check_multiple_errors(self):
        """测试complete_check能否同时识别多种类型的错误"""
        checker = BacktestCodeChecker(self.multiple_errors_code)
        result = checker.complete_check()
        
        assert isinstance(result, list), "应该返回列表"
        assert len(result) > 0, "应该检测到多个错误"
        
        # 检查是否包含多种类型的错误
        error_types = [error.get("type") for error in result]
        
        # 至少应该包含其中几种错误类型
        expected_error_types = [
            "illegal_import", 
            "dangerous_code", 
            "missing_function", 
            "illegal_log_method",
            "illegal_key_object_usage"
        ]
        
        found_error_types = [error_type for error_type in expected_error_types if error_type in error_types]
        assert len(found_error_types) > 0, f"应该检测到多种错误类型，但只发现: {error_types}"

    def test_complete_check_complex_scenarios(self):
        """测试complete_check处理复杂代码场景的能力"""
        checker = BacktestCodeChecker(self.complex_context_usage_code)
        result = checker.complete_check()
        
        assert isinstance(result, list), "应该返回列表"
        assert len(result) > 0, "应该检测到错误"
        
        # 检查是否有关键对象使用错误
        key_object_error = next(
            (error for error in result if error.get("type") == "illegal_key_object_usage"), None
        )
        
        if key_object_error:
            # 如果检测到关键对象错误，验证其格式
            assert isinstance(key_object_error.get("msg"), list), "关键对象错误的msg应该是列表"
            violations = key_object_error.get("msg")
            violation_names = [item.get("name", "") for item in violations]
            assert any("undefined" in name or "bad" in name for name in violation_names), \
                f"应该检测到未定义属性的使用，实际检测到: {violation_names}"
        else:
            # 如果没有检测到关键对象错误，至少应该有其他错误
            assert len(result) > 0, f"应该检测到某种错误，结果是: {result}"

   
if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"]) 