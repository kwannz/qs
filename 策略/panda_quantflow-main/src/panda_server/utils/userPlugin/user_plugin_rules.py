"""
用户插件验证规则配置
User Plugin Validation Rules Configuration
"""

class PluginValidationRules:
    """插件验证规则配置"""
    
    # 黑名单 - 禁止导入的模块
    IMPORT_BLACKLIST = {
        'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests', 'http', 'ftplib',
        'smtplib', 'poplib', 'imaplib', 'telnetlib', 'ssl', 'email', 'mimetypes',
        'webbrowser', 'multiprocessing', 'threading', 'asyncio', 'concurrent',
        'pickle', 'shelve', 'dbm', 'sqlite3', 'mysql', 'pymongo', 'redis',
        'exec', 'eval', 'compile', 'globals', 'locals', 'vars', 'dir',
        'importlib', '__import__', 'reload', 'input', 'raw_input'
    }

    # 危险函数调用 - 禁止调用的函数名称集合
    DANGEROUS_CALLS = {
        'exec', 'eval', 'compile', '__import__', 'globals', 'locals',
        'vars', 'dir', 'open', 'file', 'input', 'raw_input', 'reload'
    }


    # 必须实现的方法
    MANDATORY_METHODS = ['input_model', 'output_model', 'run']

    # 允许的work_node装饰器参数
    ALLOWED_WORK_NODE_PARAMS = {
        'name', 'group', 'order', 'type', 'box_color'
    } 