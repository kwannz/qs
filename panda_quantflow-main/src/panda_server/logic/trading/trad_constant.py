"""
trading_constants.py

Redis Key 常量定义
"""
class TradingConstant:
    """
    Redis 键值常量定义
    """

    # 实盘策略期货信号路由
    FUTURE_TRADE_ROUTE = "future"

    # 实盘策略账号运行情况
    REDEFINE_REAL_ACCOUNT_ASSETS = "redefine_real_account_assets"

    # 实盘策略持仓运行情况
    REDEFINE_REAL_ACCOUNT_POSITIONS = "redefine_real_account_positions"

    # 实盘策略运行进程 (带冒号表示 key 前缀)
    REAL_TRADE_PROGRESS = "real_trade_progress:"

    # 实盘启动订阅路由
    REAL_TRADE_SERVER = "real_trade_server:"

    # 账号监控服务器路由
    ACCOUNT_MONITOR_SERVER = "account_monitor_server:"

    # 监控账号资金缓存
    MONITOR_ACCOUNT_ASSETS = "monitor_account_assets"

    # 监控账号持仓缓存
    MONITOR_ACCOUNT_POSITIONS = "monitor_account_positions"

    # 账号监控进程状态
    ACCOUNT_MONITOR_PROGRESS = "account_monitor_progress:"