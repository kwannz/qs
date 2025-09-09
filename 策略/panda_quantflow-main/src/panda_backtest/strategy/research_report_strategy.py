"""
基于研报的多因子选股策略
策略核心思路：
1. 多因子模型选股：结合价值、成长、质量、动量等多个维度
2. 定期调仓：根据因子评分重新选择股票组合
3. 风险控制：设置止损、仓位管理等风险控制措施
4. 动态调整：根据市场环境调整选股数量和仓位
"""

from panda_backtest.api.api import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def initialize(context):
    """
    策略初始化
    """
    print("=== 多因子选股策略初始化 ===")
    
    # 策略参数设置
    context.stock_account = '15032863'  # 股票账户
    context.max_stocks = 10             # 最大持仓股票数量
    context.rebalance_period = 5        # 调仓周期（天）
    context.position_size = 0.08        # 单只股票仓位上限（8%）
    context.stop_loss = -0.15           # 止损线（-15%）
    context.take_profit = 0.25          # 止盈线（25%）
    
    # 股票池设置（沪深300成分股示例）
    context.stock_pool = [
        '000001.SZ', '000002.SZ', '000858.SZ', '000895.SZ', '000938.SZ',
        '000977.SZ', '002415.SZ', '002594.SZ', '002601.SZ', '002714.SZ',
        '600000.SH', '600036.SH', '600519.SH', '600887.SH', '601318.SH',
        '601398.SH', '601857.SH', '601988.SH', '600276.SH', '600309.SH'
    ]
    
    # 因子权重配置
    context.factor_weights = {
        'value_factor': 0.3,      # 价值因子权重
        'growth_factor': 0.25,    # 成长因子权重
        'quality_factor': 0.25,   # 质量因子权重
        'momentum_factor': 0.2    # 动量因子权重
    }
    
    # 初始化持仓记录
    context.positions_cost = {}  # 记录持仓成本价
    context.last_rebalance_date = None
    
    print(f"策略参数设置完成：")
    print(f"  - 股票池数量：{len(context.stock_pool)}")
    print(f"  - 最大持仓数：{context.max_stocks}")
    print(f"  - 调仓周期：{context.rebalance_period}天")
    print(f"  - 单股仓位上限：{context.position_size*100}%")

def before_trading(context):
    """
    开盘前处理
    """
    account = context.stock_account_dict[context.stock_account]
    print(f"[{context.now}] 开盘前 - 账户总价值：{account.total_value:.2f}, 可用资金：{account.cash:.2f}")

def handle_data(context, bar_dict):
    """
    主策略逻辑
    """
    current_date = context.now
    
    # 检查是否到达调仓日
    if not should_rebalance(context, current_date):
        # 非调仓日进行风险控制
        risk_management(context, bar_dict)
        return
    
    print(f"\n=== [{current_date}] 开始调仓 ===")
    
    # 获取当前可交易的股票（有行情数据的股票）
    available_stocks = [stock for stock in context.stock_pool if stock in bar_dict]
    print(f"可交易股票数量：{len(available_stocks)}")
    
    if len(available_stocks) == 0:
        print("无可交易股票，跳过本次调仓")
        return
    
    # 计算多因子评分
    stock_scores = calculate_multi_factor_scores(context, bar_dict, available_stocks)
    
    # 选择目标股票
    target_stocks = select_target_stocks(context, stock_scores)
    print(f"选中目标股票：{target_stocks}")
    
    # 执行调仓操作
    rebalance_portfolio(context, bar_dict, target_stocks)
    
    # 更新调仓日期
    context.last_rebalance_date = current_date
    
    print(f"=== [{current_date}] 调仓完成 ===\n")

def after_trading(context):
    """
    收盘后处理
    """
    account = context.stock_account_dict[context.stock_account]
    positions = account.positions
    
    # 统计持仓信息
    total_positions = len([pos for pos in positions.values() if pos.quantity > 0])
    print(f"[{context.now}] 收盘后统计：")
    print(f"  - 持仓股票数：{total_positions}")
    print(f"  - 账户总价值：{account.total_value:.2f}")
    print(f"  - 持仓市值：{account.market_value:.2f}")
    print(f"  - 可用资金：{account.cash:.2f}")

def should_rebalance(context, current_date):
    """
    判断是否应该调仓
    """
    if context.last_rebalance_date is None:
        return True
    
    # 计算距离上次调仓的天数
    try:
        last_date = datetime.strptime(str(context.last_rebalance_date), '%Y%m%d')
        curr_date = datetime.strptime(str(current_date), '%Y%m%d')
        days_diff = (curr_date - last_date).days
        return days_diff >= context.rebalance_period
    except:
        return True

def calculate_multi_factor_scores(context, bar_dict, stocks):
    """
    计算多因子评分
    """
    scores = {}
    
    for stock in stocks:
        if stock not in bar_dict:
            continue
            
        bar = bar_dict[stock]
        
        # 简化的因子计算（实际应用中需要更复杂的因子计算）
        try:
            # 价值因子：以价格倒数作为简化价值指标
            value_score = 1.0 / max(bar.close, 1.0) * 10000
            
            # 成长因子：以成交量增长作为简化成长指标  
            growth_score = min(bar.volume / 1000000, 10.0)
            
            # 质量因子：以价格稳定性作为质量指标
            quality_score = 100.0 / max(abs(bar.high - bar.low), 0.01)
            
            # 动量因子：以当日涨跌幅作为动量指标
            momentum_score = (bar.close - bar.open) / max(bar.open, 0.01) * 100
            
            # 加权计算总分
            total_score = (
                value_score * context.factor_weights['value_factor'] +
                growth_score * context.factor_weights['growth_factor'] +
                quality_score * context.factor_weights['quality_factor'] +
                momentum_score * context.factor_weights['momentum_factor']
            )
            
            scores[stock] = total_score
            
        except Exception as e:
            print(f"计算{stock}因子评分时出错：{e}")
            scores[stock] = 0
    
    return scores

def select_target_stocks(context, stock_scores):
    """
    根据因子评分选择目标股票
    """
    # 按评分排序
    sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
    
    # 选择前N只股票
    target_stocks = [stock for stock, score in sorted_stocks[:context.max_stocks]]
    
    print("股票评分排名（前10）：")
    for i, (stock, score) in enumerate(sorted_stocks[:10]):
        print(f"  {i+1}. {stock}: {score:.2f}")
    
    return target_stocks

def rebalance_portfolio(context, bar_dict, target_stocks):
    """
    执行组合调仓
    """
    account = context.stock_account_dict[context.stock_account]
    current_positions = account.positions
    
    # 计算目标仓位
    target_value_per_stock = account.total_value * context.position_size
    
    print(f"目标仓位分配：每只股票 {target_value_per_stock:.2f} 元")
    
    # 卖出不在目标列表中的股票
    for stock, position in current_positions.items():
        if position.quantity > 0 and stock not in target_stocks:
            print(f"卖出 {stock}: {position.quantity} 股")
            order_shares(context.stock_account, stock, -position.quantity)
            # 移除成本价记录
            if stock in context.positions_cost:
                del context.positions_cost[stock]
    
    # 买入或调整目标股票仓位
    for stock in target_stocks:
        if stock not in bar_dict:
            continue
            
        current_price = bar_dict[stock].close
        target_quantity = int(target_value_per_stock / current_price / 100) * 100  # 整手
        
        current_quantity = current_positions.get(stock, type('obj', (object,), {'quantity': 0})).quantity
        quantity_diff = target_quantity - current_quantity
        
        if abs(quantity_diff) >= 100:  # 最小交易单位
            print(f"调整 {stock}: 当前{current_quantity}股 -> 目标{target_quantity}股 (变动{quantity_diff}股)")
            order_shares(context.stock_account, stock, quantity_diff)
            
            # 更新成本价记录
            if quantity_diff > 0:
                context.positions_cost[stock] = current_price

def risk_management(context, bar_dict):
    """
    风险管理：止损止盈
    """
    account = context.stock_account_dict[context.stock_account]
    positions = account.positions
    
    for stock, position in positions.items():
        if position.quantity <= 0 or stock not in bar_dict:
            continue
            
        current_price = bar_dict[stock].close
        cost_price = context.positions_cost.get(stock, position.avg_price)
        
        if cost_price <= 0:
            continue
            
        # 计算收益率
        return_rate = (current_price - cost_price) / cost_price
        
        # 止损检查
        if return_rate <= context.stop_loss:
            print(f"止损卖出 {stock}: 成本价{cost_price:.2f}, 当前价{current_price:.2f}, 亏损{return_rate*100:.1f}%")
            order_shares(context.stock_account, stock, -position.quantity)
            if stock in context.positions_cost:
                del context.positions_cost[stock]
        
        # 止盈检查
        elif return_rate >= context.take_profit:
            print(f"止盈卖出 {stock}: 成本价{cost_price:.2f}, 当前价{current_price:.2f}, 盈利{return_rate*100:.1f}%")
            order_shares(context.stock_account, stock, -position.quantity)
            if stock in context.positions_cost:
                del context.positions_cost[stock]

def on_stock_trade_rtn(context, order):
    """
    股票交易回报
    """
    print(f"交易回报 - {order.order_book_id}: {order.side}{'买入' if order.side == 1 else '卖出'} {order.filled_quantity}股")

def stock_order_cancel(context, order):
    """
    股票订单撤销回报
    """
    print(f"订单撤销 - {order.order_book_id}: {order.quantity}股订单被撤销")