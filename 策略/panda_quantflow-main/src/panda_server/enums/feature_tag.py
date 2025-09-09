from enum import Enum

class FeatureTag(str, Enum):
    """
    Feature Tag
    """
    BACKTEST = "backtest"
    SIGNAL = "signal"
    FACTOR = "factor"
    TRADE = "trade"

# Maintain a mapping of feature tag to specific node names
FeatureTagNodeNames = {
    FeatureTag.BACKTEST: ["FutureBacktestControl","StockBacktestControl"],
    FeatureTag.SIGNAL: [],
    FeatureTag.FACTOR: ["FactorAnalysisControl"],
    FeatureTag.TRADE: [],
}