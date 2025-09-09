from enum import Enum

from collections import defaultdict
import traceback

class ConstantEvent(Enum):
    # ###########################策略事件################################
    # 策略初始化
    STRATEGY_INIT = 'STRATEGY_INIT'
    # 策略定时bar执行
    STRATEGY_HANDLE_BAR = 'STRATEGY_HANDLE_BAR'
    # 策略before_trading执行
    STRATEGY_TRADING_BEFORE = 'STRATEGY_TRADING_BEFORE'
    # 策略after_trading执行
    STRATEGY_TRADING_AFTER = 'STRATEGY_TRADING_AFTER'
    # 策略day_before执行
    STRATEGY_DAY_BEFORE = 'STRATEGY_DAY_BEFORE'
    # 策略股票cancel_order执行
    STOCK_ORDER_CANCEL = 'STOCK_ORDER_CANCEL'
    # 策略期货future_cancel_order执行
    FUTURE_ORDER_CANCEL = 'FUTURE_ORDER_CANCEL'
    # 策略基金cancel_order执行
    FUND_ORDER_CANCEL = 'FUND_ORDER_CANCEL'
    # 策略day_before执行
    STRATEGY_HANDLE_TICK = 'STRATEGY_HANDLE_TICK'
    # 策略期货成交on_future_trade_rtn执行
    ON_FUTURE_TRADE_RTN = 'ON_FUTURE_TRADE_RTN'
    # 策略股票成交on_stock_trade_rtn执行
    ON_STOCK_TRADE_RTN = 'ON_STOCK_TRADE_RTN'

    # ############################风控事件################################
    # 风控重载
    RISK_CONTROL_RELOAD = 'RISK_CONTROL_RELOAD'
    # 风控初始化
    RISK_CONTROL_INIT = 'RISK_CONTROL_INIT'
    # 风控before_trading执行
    RISK_CONTROL_TRADING_BEFORE = 'RISK_CONTROL_TRADING_BEFORE'
    # 风控after_trading执行
    RISK_CONTROL_TRADING_AFTER = 'RISK_CONTROL_TRADING_AFTER'
    # 风控day_before校验
    RISK_CONTROL_DAY_BEFORE = 'RISK_CONTROL_DAY_BEFORE'
    # 策略定时bar执行
    RISK_CONTROL_HANDLE_BAR = 'RISK_CONTROL_HANDLE_BAR'
    # 风控订单校验
    RISK_CONTROL_ORDER_VERIFY = 'RISK_CONTROL_ORDER_VERIFY'

    # ###########################交易事件################################
    # 系统订单撮合事件
    SYSTEM_STOCK_ORDER_CROSS = 'SYSTEM_STOCK_ORDER_CROSS'
    # 系统股票订单回报事件
    SYSTEM_STOCK_RTN_ORDER = 'SYSTEM_STOCK_RTN_ORDER'
    # 系统订单成交事件
    SYSTEM_STOCK_RTN_TRADE = 'SYSTEM_STOCK_RTN_TRADE'
    # 系统股票资金调动
    SYSTEM_STOCK_RTN_TRANSFER = 'SYSTEM_STOCK_RTN_TRANSFER'
    # 系统订单撤单
    SYSTEM_STOCK_ORDER_CANCEL = 'SYSTEM_STOCK_ORDER_CANCEL'
    # 股票分红
    SYSTEM_STOCK_DIVIDEND = 'SYSTEM_STOCK_DIVIDEND'
    # etf份额拆分
    SYSTEM_ETF_SPLIT = 'SYSTEM_ETF_SPLIT'
    # 股票行情更新
    SYSTEM_STOCK_QUOTATION_CHANGE = 'SYSTEM_STOCK_QUOTATION_CHANGE'
    # 股票账号资金更新
    SYSTEM_STOCK_ASSET_REFRESH = 'SYSTEM_STOCK_ASSET_REFRESH'
    # 实盘所有持仓刷新
    SYSTEM_STOCK_ALL_POSITION_REFRESH = 'SYSTEM_STOCK_ALL_POSITION_REFRESH'
    # 成交对应持仓刷新
    SYSTEM_STOCK_TRADE_POSITION_REFRESH = 'SYSTEM_STOCK_TRADE_POSITION_REFRESH'
    # 股票开始订阅
    SYSTEM_STOCK_QUOTATION_START_SUB = 'SYSTEM_STOCK_QUOTATION_START_SUB'
    # 股票取消订阅
    SYSTEM_STOCK_QUOTATION_START_UN_SUB = 'SYSTEM_STOCK_QUOTATION_START_UN_SUB'
    # 系统订单撮合事件
    SYSTEM_FUTURE_ORDER_CROSS = 'SYSTEM_FUTURE_ORDER_CROSS'
    # 系统订单回报事件
    SYSTEM_FUTURE_RTN_ORDER = 'SYSTEM_FUTURE_RTN_ORDER'
    # 系统订单成交事件
    SYSTEM_FUTURE_RTN_TRADE = 'SYSTEM_FUTURE_RTN_TRADE'
    # 系统订单撤单
    SYSTEM_FUTURE_ORDER_CANCEL = 'SYSTEM_FUTURE_ORDER_CANCEL'
    # 系统股票资金调动
    SYSTEM_FUTURE_RTN_TRANSFER = 'SYSTEM_FUTURE_RTN_TRANSFER'
    # 期货爆仓
    SYSTEM_FUTURE_BURNED = 'SYSTEM_FUTURE_BURNED'
    # 期货每日结算
    SYSTEM_FUTURE_SETTLE = 'SYSTEM_FUTURE_SETTLE'
    # 期货合约交割
    SYSTEM_FUTURE_DELIVERY = 'SYSTEM_FUTURE_DELIVERY'
    # 期货行情更新
    SYSTEM_FUTURE_QUOTATION_CHANGE = 'SYSTEM_FUTURE_QUOTATION_CHANGE'
    # 实盘所有持仓刷新
    SYSTEM_FUTURE_ALL_POSITION_REFRESH = 'SYSTEM_FUTURE_ALL_POSITION_REFRESH'
    # 成交对应持仓刷新
    SYSTEM_FUTURE_TRADE_POSITION_REFRESH = 'SYSTEM_FUTURE_TRADE_POSITION_REFRESH'
    # 期货账号资金刷新
    SYSTEM_FUTURE_ASSET_REFRESH = 'SYSTEM_FUTURE_ASSET_REFRESH'
    # 期货开始订阅
    SYSTEM_FUTURE_QUOTATION_START_SUB = 'SYSTEM_FUTURE_QUOTATION_START_SUB'
    # 期货开始取消订阅
    SYSTEM_FUTURE_QUOTATION_START_UN_SUB = 'SYSTEM_FUTURE_QUOTATION_START_UN_SUB'
    # 基金撮合事件
    SYSTEM_FUND_ORDER_CROSS = 'SYSTEM_FUND_ORDER_CROSS'
    # 基金行情更新
    SYSTEM_FUND_QUOTATION_CHANGE = 'SYSTEM_FUND_QUOTATION_CHANGE'
    # 系统基金订单回报
    SYSTEM_FUND_RTN_ORDER = 'SYSTEM_FUND_RTN_ORDER'
    # 系统基金订单成交事件
    SYSTEM_FUND_RTN_TRADE = 'SYSTEM_FUND_RTN_TRADE'
    # 系统订单撤单
    SYSTEM_FUND_ORDER_CANCEL = 'SYSTEM_FUND_ORDER_CANCEL'
    # 基金分红
    SYSTEM_FUND_DIVIDEND = 'SYSTEM_FUND_DIVIDEND'
    # 基金份额拆分
    SYSTEM_FUND_SPLIT = 'SYSTEM_FUND_SPLIT'
    # 基金开始订阅
    SYSTEM_FUND_QUOTATION_START_SUB = 'SYSTEM_FUND_QUOTATION_START_SUB'
    # 基金取消订阅
    SYSTEM_FUND_QUOTATION_START_UN_SUB = 'SYSTEM_FUND_QUOTATION_START_UN_SUB'
    # 基金清算
    SYSTEM_FUND_SETTLEMENT = 'SYSTEM_FUND_SETTLEMENT'
    # ###########################时间事件################################
    # 系统新交易日开始
    SYSTEM_NEW_DATE = 'SYSTEM_NEW_DATE'
    # 系统交易日结束
    SYSTEM_END_DATE = 'SYSTEM_END_DATE'
    # 系统每日开始
    SYSTEM_DAY_START = 'SYSTEM_DAY_START'
    # 夜盘结束
    SYSTEM_NIGHT_END = 'SYSTEM_NIGHT_END'
    # bar触发
    SYSTEM_HANDLE_BAR = 'SYSTEM_HANDLE_BAR'
    # tick触发
    SYSTEM_HANDLE_TICK = 'SYSTEM_HANDLE_TICK'
    # 实盘每日数据保存
    SYSTEM_DAILY_DATA_SAVE = 'SYSTEM_DAILY_DATA_SAVE'
    # 统计数据
    SYSTEM_CALCULATE_RESULT = 'SYSTEM_CALCULATE_RESULT'

    STRATEGY_TRADE_ERROR = 'STRATEGY_TRADE_ERROR'

    # 保存运行数据
    SYSTEM_RESTORE_STRATEGY = 'SYSTEM_RESTORE_STRATEGY'

    # 微信消息通知
    SYSTEM_WX_NOTIFY = 'SYSTEM_WX_NOTIFY'

class Event(object):
    def __init__(self, event_name, **kwargs):
        self.kwargs = kwargs
        self.event_name = event_name

    def __repr__(self):
        return ' '.join('{}:{}'.format(k, v) for k, v in self.__dict__.items())

class EventBus(object):
    def __init__(self):
        self._handles = defaultdict(list)

    def register_handle(self, event, handle):
        self._handles[event].append(handle)

    def add_handle(self, event, handle):
        self._handles[event].insert(0, handle)

    def publish_event(self, event):
        for l in self._handles[event.event_name]:
            if event.event_name == ConstantEvent.STRATEGY_HANDLE_BAR or \
                    event.event_name == ConstantEvent.STRATEGY_INIT or \
                    event.event_name == ConstantEvent.STRATEGY_TRADING_BEFORE or \
                    event.event_name == ConstantEvent.STRATEGY_TRADING_AFTER or \
                    event.event_name == ConstantEvent.STRATEGY_DAY_BEFORE or \
                    event.event_name == ConstantEvent.STOCK_ORDER_CANCEL or \
                    event.event_name == ConstantEvent.FUND_ORDER_CANCEL or \
                    event.event_name == ConstantEvent.STRATEGY_HANDLE_TICK or \
                    event.event_name == ConstantEvent.ON_FUTURE_TRADE_RTN or \
                    event.event_name == ConstantEvent.ON_STOCK_TRADE_RTN or \
                    event.event_name == ConstantEvent.RISK_CONTROL_INIT or \
                    event.event_name == ConstantEvent.RISK_CONTROL_TRADING_BEFORE or \
                    event.event_name == ConstantEvent.RISK_CONTROL_TRADING_AFTER or \
                    event.event_name == ConstantEvent.RISK_CONTROL_DAY_BEFORE or \
                    event.event_name == ConstantEvent.RISK_CONTROL_ORDER_VERIFY:
                l(*event.kwargs.values())
            else:
                l(*event.kwargs.values())