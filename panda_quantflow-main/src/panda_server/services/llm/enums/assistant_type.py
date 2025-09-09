from enum import Enum


class AssistantType(str, Enum):
    """
    助手类型枚举
    Assistant type enumeration for limiting valid assistant types
    """

    GENERAL = "general-assistant"  # 通用助手(解决非代码类问题)
    CODE = "code-assistant"  # 通用代码助手
    BACKTEST = "backtest-assistant"  # 回测代码助手
    FACTOR = "factor-assistant"  # 因子分析代码助手
    SIGNAL = "signal-assistant"  # 信号分析代码助手
    TRADE = "trade-assistant"  # 实盘交易代码助手
