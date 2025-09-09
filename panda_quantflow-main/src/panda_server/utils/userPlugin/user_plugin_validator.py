"""
用户插件代码验证器
User Plugin Code Validator

"""

import ast
import logging
import re
from typing import List, Dict, Any
from fastapi import HTTPException, status
from .user_plugin_rules import PluginValidationRules

logger = logging.getLogger(__name__)


class PluginValidator:
    """插件代码验证器"""
    
    # 接收用户提交的源代码字符串，初始化 AST 容器 _tree 和违规记录列表 violations。
    def __init__(self, code: str):
        self.code = code
        self._tree: ast.Module = None
        self.violations: List[Dict[str, Any]] = []
    
    # 构造并保存一条违规记录
    def _add_violation(self, violation_type: str, message: str, line: int = None, **kwargs):
        """添加违规记录"""
        violation = {
            'type': violation_type,
            'message': message,
            'line': line,
            **kwargs
        }
        self.violations.append(violation)
        
    # 检查代码语法错误并解析为AST树
    def check_syntax(self) -> bool:
        """
        检查代码语法错误并解析为AST树

        Returns:
            bool: True if syntax is valid, False otherwise
        """
        try:
            self._tree = ast.parse(self.code)
            return True
        except SyntaxError as e:
            error_msg = f"代码语法错误: {e.msg}"
            if e.lineno:
                error_msg += f" (第 {e.lineno} 行"
                if e.offset:
                    error_msg += f", 第 {e.offset} 列"
                error_msg += ")"
            
            if e.text:
                error_msg += f"\n错误行: {e.text.strip()}"
            
            self._add_violation('syntax_error', error_msg, e.lineno)
            return False

    # 找到所有 Import 和 ImportFrom 节点。验证import的黑名单
    def validate_imports(self) -> bool:
        """
        验证import的黑名单
        
        Returns:
            bool: True if imports are valid, False otherwise
        """
        is_valid = True
        
        for node in ast.walk(self._tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name in PluginValidationRules.IMPORT_BLACKLIST:
                        self._add_violation(
                            'forbidden_import',
                            f"禁止导入 {module_name} 模块",
                            node.lineno
                        )
                        is_valid = False
                        
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name in PluginValidationRules.IMPORT_BLACKLIST:
                        self._add_violation(
                            'forbidden_import',
                            f"禁止导入 {module_name} 模块",
                            node.lineno
                        )
                        is_valid = False
        
        return is_valid

    # 找到所有 Call 节点。验证危险函数调用
    def validate_dangerous_code(self) -> bool:
        """
        验证是否有危险代码
        
        Returns:
            bool: True if no dangerous code found, False otherwise
        """
        is_valid = True
        
        # 使用AST检查危险函数调用和导入
        for node in ast.walk(self._tree):
            if isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        func_name = f"{node.func.value.id}.{node.func.attr}"
                
                if func_name in PluginValidationRules.DANGEROUS_CALLS:
                    self._add_violation(
                        'dangerous_call',
                        f"禁止调用 {func_name} 危险函数",
                        node.lineno
                    )
                    is_valid = False
        
        return is_valid

    # 找到所有 ClassDef 节点。验证是否存在类，继承自BaseWorkNode，并且这个类有且仅有1个
    def validate_base_work_node_class(self) -> str:
        """
        验证是否存在类，继承自BaseWorkNode，并且这个类有且仅有1个
        
        Returns:
            str: 工作节点类名
            
        Raises:
            HTTPException: 类定义不符合要求时抛出
        """
        work_node_classes = []
        
        for node in ast.walk(self._tree):
            if isinstance(node, ast.ClassDef):
                # 检查是否继承自BaseWorkNode
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == 'BaseWorkNode':
                        work_node_classes.append((node.name, node.lineno))
                        break
        
        if len(work_node_classes) == 0:
            self._add_violation(
                'missing_base_class',
                "代码必须包含一个继承自BaseWorkNode的类",
                None
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="代码必须包含一个继承自BaseWorkNode的类"
            )
        elif len(work_node_classes) > 1:
            class_names = [cls[0] for cls in work_node_classes]
            self._add_violation(
                'multiple_base_classes',
                f"代码只能包含一个继承自BaseWorkNode的类，发现了{len(work_node_classes)}个: {', '.join(class_names)}",
                None
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"代码只能包含一个继承自BaseWorkNode的类，发现了{len(work_node_classes)}个: {', '.join(class_names)}"
            )
        
        return work_node_classes[0][0]
    
    # 找到所有 ClassDef 节点。验证这个类是否添加 @work_node 来注册节点，并且属性是否是定义的那几个
    def validate_work_node_decorator(self, class_name: str) -> Dict[str, Any]:
        """
        验证这个类是否添加 @work_node 来注册节点，并且属性是否是定义的那几个
        
        Args:
            class_name: 工作节点类名
            
        Returns:
            Dict[str, Any]: 装饰器参数
            
        Raises:
            HTTPException: 装饰器不符合要求时抛出
        """
        for node in ast.walk(self._tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # 检查装饰器
                work_node_decorator = None
                for decorator in node.decorator_list:
                    if isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name) and decorator.func.id == 'work_node':
                            work_node_decorator = decorator
                            break
                    elif isinstance(decorator, ast.Name) and decorator.id == 'work_node':
                        work_node_decorator = decorator
                        break
                
                if not work_node_decorator:
                    self._add_violation(
                        'missing_decorator',
                        f"类 {class_name} 必须使用 @work_node 装饰器",
                        node.lineno
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"类 {class_name} 必须使用 @work_node 装饰器"
                    )
                
                # 解析装饰器参数（如果有）
                decorator_params = {}
                if isinstance(work_node_decorator, ast.Call):
                    # 验证参数名称是否在允许的范围内
                    for keyword in work_node_decorator.keywords:
                        if keyword.arg not in PluginValidationRules.ALLOWED_WORK_NODE_PARAMS:
                            self._add_violation(
                                'invalid_decorator_param',
                                f"@work_node 装饰器不支持参数: {keyword.arg}",
                                node.lineno
                            )
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail=f"@work_node 装饰器不支持参数: {keyword.arg}"
                            )
                        else:
                            if isinstance(keyword.value, ast.Constant):
                                decorator_params[keyword.arg] = keyword.value.value
                            elif isinstance(keyword.value, ast.Str):  # Python < 3.8 兼容
                                decorator_params[keyword.arg] = keyword.value.s
                            elif isinstance(keyword.value, ast.Num):  # Python < 3.8 兼容
                                decorator_params[keyword.arg] = keyword.value.n
                    
                    # 解析位置参数（第一个参数是name）
                    if work_node_decorator.args:
                        first_arg = work_node_decorator.args[0]
                        if isinstance(first_arg, ast.Constant):
                            decorator_params['name'] = first_arg.value
                        elif isinstance(first_arg, ast.Str):  # Python < 3.8 兼容
                            decorator_params['name'] = first_arg.s
                
                # 验证必需的参数
                if 'name' not in decorator_params:
                    self._add_violation(
                        'missing_required_param',
                        "@work_node 装饰器必须指定 name 参数",
                        node.lineno
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="@work_node 装饰器必须指定 name 参数"
                    )
                
                return decorator_params
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"未找到类 {class_name}"
        )

    # 找到所有 ClassDef 节点。验证这个类是否存在至少三个function，分别是（input_model，output_model，run）
    def validate_required_methods(self, class_name: str) -> bool:
        """
        验证这个类是否存在至少三个function，分别是（input_model，output_model，run）
        
        Args:
            class_name: 工作节点类名
            
        Returns:
            bool: True if all required methods exist, False otherwise
        """
        found_methods = set()
        
        for node in ast.walk(self._tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name in PluginValidationRules.MANDATORY_METHODS:
                            found_methods.add(item.name)
        
        missing_methods = set(PluginValidationRules.MANDATORY_METHODS) - found_methods
        if missing_methods:
            self._add_violation(
                'missing_methods',
                f"类 {class_name} 缺少必需的方法: {', '.join(missing_methods)}",
                None
            )
            return False
        
        return True
    
    # 验证插件代码的完整性
    def _validate_plugin_code(self) -> Dict[str, Any]:
        """
        验证插件代码的完整性
        
        Returns:
            Dict[str, Any]: 验证结果，包含success标记、错误信息、类名和装饰器参数
        """
        # 清空之前的违规记录
        self.violations = []
        
        # 1. 检查代码语法错误并解析为AST树
        if not self.check_syntax():
            syntax_errors = [v for v in self.violations if v['type'] == 'syntax_error']
            if syntax_errors:
                return {
                    'success': False,
                    'error_message': syntax_errors[0]['message'],
                    'error_type': 'syntax_error'
                }
        
        # 2. 验证import的黑名单
        if not self.validate_imports():
            import_errors = [v for v in self.violations if v['type'] == 'forbidden_import']
            if import_errors:
                return {
                    'success': False,
                    'error_message': import_errors[0]['message'],
                    'error_type': 'forbidden_import'
                }
        
        # 3. 验证是否有危险代码
        if not self.validate_dangerous_code():
            dangerous_errors = [v for v in self.violations if v['type'] in ['dangerous_code', 'dangerous_call']]
            if dangerous_errors:
                return {
                    'success': False,
                    'error_message': dangerous_errors[0]['message'],
                    'error_type': 'dangerous_code'
                }
        
        # 4. 验证是否存在类，继承自BaseWorkNode，并且这个类有且仅有1个
        try:
            class_name = self.validate_base_work_node_class()
        except HTTPException as e:
            return {
                'success': False,
                'error_message': e.detail,
                'error_type': 'base_class_error'
            }
        
        # 5. 验证这个类是否添加 @work_node 来注册节点，并且属性是否是定义的那几个
        try:
            decorator_params = self.validate_work_node_decorator(class_name)
        except HTTPException as e:
            return {
                'success': False,
                'error_message': e.detail,
                'error_type': 'decorator_error'
            }
        
        # 6. 验证这个类是否存在至少三个function，分别是（input_model，output_model，run）
        if not self.validate_required_methods(class_name):
            method_errors = [v for v in self.violations if v['type'] == 'missing_methods']
            if method_errors:
                return {
                    'success': False,
                    'error_message': method_errors[0]['message'],
                    'error_type': 'missing_methods'
                }
        
        return {
            'success': True,
            'class_name': class_name,
            'decorator_params': decorator_params,
            'violations': self.violations
        }

    @classmethod
    def validate_plugin_code(cls, code: str) -> Dict[str, Any]:
        """
        静态方法：验证插件代码的完整性（保持向后兼容）
        
        Args:
            code: 用户提交的代码
            
        Returns:
            Dict[str, Any]: 验证结果，包含类名和装饰器参数
            
        Raises:
            HTTPException: 验证失败时抛出
        """
        validator = cls(code)
        return validator._validate_plugin_code() 