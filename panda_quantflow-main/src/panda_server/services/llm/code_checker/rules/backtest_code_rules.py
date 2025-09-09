"""
回测代码检查规则
Backtest code check rules

参考文档: https://www.pandaai.online/community/article/117
"""

# 允许导入的库: 仅在白名单检查模式下启用, 这个模式下其它导入都视为非法
allowed_imports = [
    "panda_backtest",
    "panda_factor",
    "pandas",
    "numpy",
    "scipy",
    "sklearn",
    "datetime",
]

# 不允许导入的库: 仅在黑名单检查模式下启用, 这个模式下其它导入都视为合法
not_allowed_imports = [
    "os",
    "sys",
    "logging",
    "subprocess",
    "shutil",
    "eval",
    "exec",
    "pickle",
    "yaml",
    "marshal",
    "requests",
    "sqlite3",
    "pymysql",
    "pymongo",
    "redis",
    "elasticsearch",
    "kafka",
    "rabbitmq",
]

# 必须实现的函数
mandatory_functions = [
    "initialize",
    "handle_data",
]

# 关键内置对象规则
key_object_rules = [
    # context 对象
    {
        "in_function": [
            "initialize",
            "handle_data",
            "before_trading",
            "after_trading",
            "on_stock_trade_rtn",
            "stock_order_cancel",
            "on_future_trade_rtn",
            "future_order_cancel",
        ],
        "arg_name": "context",
        "allowed_attributes": [
            "now",
            "portfolio_dict",
            "stock_account_dict",
            "future_account_dict",
            "df_factor",
        ],
        "allowed_methods": [],
        "allow_custom_attributes": True,
        "track_across_functions": True,
    },
    #  order 对象
    {
        "in_function": [
            "on_stock_trade_rtn",
            "stock_order_cancel",
            "on_future_trade_rtn",
            "future_order_cancel",
        ],
        "arg_name": "order",
        "allowed_attributes": [
            "order_id",
            "order_book_id",
            "side",
            "effect",
            "price",
            "quantity",
            "filled_quantity",
            "unfilled_quantity",
            "status",
            "message",
        ],
        "allowed_methods": [],
        "allow_custom_attributes": False,
        "track_across_functions": True,
    },
]

# 关键内置方法规则
key_method_rules = [
    # 框架方法
    {
        "function_name": "initialize",
        "required_args": ["context"],
        "optional_args": [],
        "supports_varargs": False,
        "supports_kwargs": False,
    },
    {
        "function_name": "handle_data",
        "required_args": ["context", "bar_dict"],
        "optional_args": [],
        "supports_varargs": False,
        "supports_kwargs": False,
    },
    {
        "function_name": "before_trading",
        "required_args": ["context"],
        "optional_args": [],
        "supports_varargs": False,
        "supports_kwargs": False,
    },
    {
        "function_name": "after_trading",
        "required_args": ["context"],
        "optional_args": [],
        "supports_varargs": False,
        "supports_kwargs": False,
    },
    {
        "function_name": "on_stock_trade_rtn",
        "required_args": ["context", "order"],
        "optional_args": [],
        "supports_varargs": False,
        "supports_kwargs": False,
    },
    {
        "function_name": "stock_order_cancel",
        "required_args": ["context", "order"],
        "optional_args": [],
        "supports_varargs": False,
        "supports_kwargs": False,
    },
    {
        "function_name": "on_future_trade_rtn",
        "required_args": ["context", "order"],
        "optional_args": [],
        "supports_varargs": False,
        "supports_kwargs": False,
    },
    {
        "function_name": "future_order_cancel",
        "required_args": ["context", "order"],
        "optional_args": [],
        "supports_varargs": False,
        "supports_kwargs": False,
    },
    # 交易API函数
    {
        "function_name": "order_shares",
        "required_args": ["account", "symbol", "amount", "style"],
        "optional_args": [],
        "supports_varargs": False,
        "supports_kwargs": False,
    },
    {
        "function_name": "buy_open",
        "required_args": ["account", "symbol", "amount", "style"],
        "optional_args": [],
        "supports_varargs": False,
        "supports_kwargs": False,
    },
    {
        "function_name": "sell_open",
        "required_args": ["account", "symbol", "amount", "style"],
        "optional_args": [],
        "supports_varargs": False,
        "supports_kwargs": False,
    },
    {
        "function_name": "buy_close",
        "required_args": ["account", "symbol", "amount", "style"],
        "optional_args": [],
        "supports_varargs": False,
        "supports_kwargs": False,
    },
    {
        "function_name": "sell_close",
        "required_args": ["account", "symbol", "amount", "style"],
        "optional_args": [],
        "supports_varargs": False,
        "supports_kwargs": False,
    },
    {
        "function_name": "target_future_group_order",
        "required_args": ["account", "orders", "positions"],
        "optional_args": [],
        "supports_varargs": False,
        "supports_kwargs": False,
    },
]
