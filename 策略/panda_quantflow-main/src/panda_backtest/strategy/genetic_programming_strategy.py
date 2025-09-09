"""
基于遗传规划的期货基本面量化因子策略
根据东证期货研究报告复现

策略核心：
1. 使用降维方法处理期货基本面数据
2. 遗传规划算法优化因子组合
3. 以夏普值作为适应度函数
4. 动态调整持仓和交易信号

适用品种：期货(铝、原油、镍、锡、锌等)
"""

from panda_backtest.api.api import *
import numpy as np
import random
from datetime import datetime

def initialize(context):
    """
    策略初始化
    """
    print("=== 遗传规划期货策略初始化 ===")
    
    # 账户设置
    context.future_account = '5588'  # 期货账户
    
    # 策略参数
    context.rebalance_period = 5        # 调仓周期（天）
    context.position_size = 0.3         # 单品种仓位上限（30%）
    context.stop_loss = -0.08           # 止损线（-8%）
    context.take_profit = 0.15          # 止盈线（15%）
    
    # 期货品种池（主要工业品和能源）
    context.futures_pool = [
        'AL2501.SHF',    # 铝
        'CU2501.SHF',    # 铜  
        'ZN2501.SHF',    # 锌
        'NI2501.SHF',    # 镍
        'SN2501.SHF',    # 锡
        'RB2501.SHF',    # 螺纹钢
        'SC2501.INE',    # 原油
        'FU2501.SHF',    # 燃油
        'AU2512.SHF',    # 黄金
        'AG2512.SHF'     # 白银
    ]
    
    # 遗传规划参数
    context.population_size = 20        # 种群大小
    context.generations = 10            # 进化代数
    context.mutation_rate = 0.1         # 变异率
    context.crossover_rate = 0.8        # 交叉率
    context.factor_window = 20          # 因子计算窗口
    
    # 函数库（简化版本的时序处理函数）
    context.function_library = {
        'ts_rank': ts_rank,
        'ts_mean': ts_mean, 
        'ts_std': ts_std,
        'ts_max': ts_max,
        'ts_min': ts_min,
        'ts_delta': ts_delta
    }
    
    # 初始化数据存储
    context.price_history = {}          # 价格历史数据
    context.factor_population = []      # 因子种群
    context.fitness_scores = []         # 适应度分数
    context.best_factors = {}           # 最优因子
    context.last_rebalance_date = None
    context.position_costs = {}         # 持仓成本
    
    print(f"策略参数：")
    print(f"  - 品种数量：{len(context.futures_pool)}")
    print(f"  - 调仓周期：{context.rebalance_period}天")
    print(f"  - 种群大小：{context.population_size}")
    print(f"  - 进化代数：{context.generations}")

def before_trading(context):
    """
    开盘前处理
    """
    account = context.future_account_dict[context.future_account]
    print(f"[{context.now}] 期货账户总权益：{account.total_value:.2f}")

def handle_data(context, bar_dict):
    """
    主策略逻辑
    """
    current_date = context.now
    
    # 更新价格历史
    update_price_history(context, bar_dict)
    
    # 检查是否到达调仓日
    if not should_rebalance(context, current_date):
        # 非调仓日进行风险控制
        risk_management(context, bar_dict)
        return
    
    print(f"\n=== [{current_date}] 开始遗传规划调仓 ===")
    
    # 获取可交易的期货合约
    available_futures = [symbol for symbol in context.futures_pool if symbol in bar_dict]
    print(f"可交易期货数量：{len(available_futures)}")
    
    if len(available_futures) == 0:
        print("无可交易期货，跳过调仓")
        return
    
    # 执行遗传规划优化
    best_factors = genetic_programming_optimization(context, available_futures)
    
    # 基于最优因子生成交易信号
    trading_signals = generate_trading_signals(context, bar_dict, available_futures, best_factors)
    
    # 执行交易
    execute_trades(context, bar_dict, trading_signals)
    
    # 更新状态
    context.last_rebalance_date = current_date
    print(f"=== [{current_date}] 调仓完成 ===\n")

def should_rebalance(context, current_date):
    """判断是否应该调仓"""
    if context.last_rebalance_date is None:
        return True
    
    try:
        last_date = datetime.strptime(str(context.last_rebalance_date), '%Y%m%d')
        curr_date = datetime.strptime(str(current_date), '%Y%m%d')
        days_diff = (curr_date - last_date).days
        return days_diff >= context.rebalance_period
    except:
        return True

def update_price_history(context, bar_dict):
    """更新价格历史数据"""
    for symbol in context.futures_pool:
        if symbol in bar_dict:
            if symbol not in context.price_history:
                context.price_history[symbol] = []
            
            context.price_history[symbol].append({
                'date': context.now,
                'open': bar_dict[symbol].open,
                'high': bar_dict[symbol].high,
                'low': bar_dict[symbol].low,
                'close': bar_dict[symbol].close,
                'volume': bar_dict[symbol].volume
            })
            
            # 保持历史数据长度
            if len(context.price_history[symbol]) > context.factor_window * 2:
                context.price_history[symbol] = context.price_history[symbol][-context.factor_window * 2:]

def genetic_programming_optimization(context, available_futures):
    """
    遗传规划优化因子
    """
    print("执行遗传规划优化...")
    
    # 初始化种群（如果没有）
    if not context.factor_population:
        context.factor_population = initialize_population(context, available_futures)
    
    # 进化过程
    for generation in range(context.generations):
        # 计算适应度（夏普值）
        fitness_scores = evaluate_fitness(context, available_futures)
        context.fitness_scores = fitness_scores
        
        # 选择、交叉、变异
        new_population = evolve_population(context, fitness_scores)
        context.factor_population = new_population
        
        if generation % 3 == 0:  # 每3代打印一次
            best_fitness = max(fitness_scores) if fitness_scores else 0
            print(f"  第{generation+1}代，最佳适应度：{best_fitness:.4f}")
    
    # 返回最优因子
    if context.fitness_scores:
        best_index = context.fitness_scores.index(max(context.fitness_scores))
        best_factors = context.factor_population[best_index]
        print(f"遗传规划完成，最佳适应度：{max(context.fitness_scores):.4f}")
        return best_factors
    
    return {}

def initialize_population(context, symbols):
    """初始化因子种群"""
    population = []
    
    for _ in range(context.population_size):
        individual = {}
        for symbol in symbols:
            # 为每个合约生成随机因子组合
            individual[symbol] = {
                'primary_factor': random.choice(list(context.function_library.keys())),
                'secondary_factor': random.choice(list(context.function_library.keys())),
                'window1': random.randint(5, 20),
                'window2': random.randint(5, 20),
                'weight1': random.uniform(0.3, 0.7),
                'weight2': random.uniform(0.3, 0.7)
            }
        population.append(individual)
    
    return population

def evaluate_fitness(context, symbols):
    """计算种群适应度（夏普值）"""
    fitness_scores = []
    
    for individual in context.factor_population:
        returns = []
        
        for symbol in symbols:
            if symbol in context.price_history and len(context.price_history[symbol]) >= 10:
                # 计算因子值
                factor_score = calculate_factor_score(context, symbol, individual[symbol])
                
                # 简化的收益计算
                if factor_score is not None:
                    price_data = context.price_history[symbol][-10:]
                    price_returns = [(price_data[i]['close'] / price_data[i-1]['close'] - 1) 
                                   for i in range(1, len(price_data))]
                    
                    # 因子信号与收益的相关性作为收益代理
                    if price_returns:
                        signal_return = factor_score * np.mean(price_returns)
                        returns.append(signal_return)
        
        # 计算夏普值作为适应度
        if returns:
            returns = np.array(returns)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = mean_return / (std_return + 1e-6)  # 避免除零
            fitness_scores.append(sharpe)
        else:
            fitness_scores.append(0)
    
    return fitness_scores

def calculate_factor_score(context, symbol, factor_config):
    """计算单个因子得分"""
    if symbol not in context.price_history:
        return None
    
    price_data = context.price_history[symbol]
    if len(price_data) < max(factor_config['window1'], factor_config['window2']):
        return None
    
    try:
        # 提取价格序列
        closes = [p['close'] for p in price_data]
        volumes = [p['volume'] for p in price_data]
        
        # 计算主因子
        primary_func = context.function_library[factor_config['primary_factor']]
        primary_score = primary_func(closes, factor_config['window1'])
        
        # 计算次因子
        secondary_func = context.function_library[factor_config['secondary_factor']]
        secondary_score = secondary_func(volumes, factor_config['window2'])
        
        # 加权合成
        if primary_score is not None and secondary_score is not None:
            total_weight = factor_config['weight1'] + factor_config['weight2']
            combined_score = (primary_score * factor_config['weight1'] + 
                            secondary_score * factor_config['weight2']) / total_weight
            return combined_score
    
    except Exception as e:
        print(f"计算因子得分出错 {symbol}: {e}")
    
    return None

def evolve_population(context, fitness_scores):
    """进化种群：选择、交叉、变异"""
    population = context.factor_population
    new_population = []
    
    # 保留最优个体
    if fitness_scores:
        best_index = fitness_scores.index(max(fitness_scores))
        new_population.append(population[best_index].copy())
    
    # 轮盘赌选择和交叉
    while len(new_population) < context.population_size:
        # 选择父母
        parent1 = tournament_selection(population, fitness_scores)
        parent2 = tournament_selection(population, fitness_scores)
        
        # 交叉
        if random.random() < context.crossover_rate:
            child1, child2 = crossover(parent1, parent2)
        else:
            child1, child2 = parent1.copy(), parent2.copy()
        
        # 变异
        if random.random() < context.mutation_rate:
            child1 = mutate(context, child1)
        if random.random() < context.mutation_rate:
            child2 = mutate(context, child2)
        
        new_population.extend([child1, child2])
    
    return new_population[:context.population_size]

def tournament_selection(population, fitness_scores, tournament_size=3):
    """锦标赛选择"""
    tournament_indices = random.sample(range(len(population)), 
                                     min(tournament_size, len(population)))
    tournament_fitness = [fitness_scores[i] for i in tournament_indices]
    winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
    return population[winner_index].copy()

def crossover(parent1, parent2):
    """交叉操作"""
    child1, child2 = parent1.copy(), parent2.copy()
    
    # 随机选择交叉的合约
    symbols = list(parent1.keys())
    crossover_symbols = random.sample(symbols, len(symbols) // 2)
    
    for symbol in crossover_symbols:
        child1[symbol], child2[symbol] = child2[symbol].copy(), child1[symbol].copy()
    
    return child1, child2

def mutate(context, individual):
    """变异操作"""
    mutated = individual.copy()
    symbols = list(individual.keys())
    
    # 随机选择一个合约进行变异
    if symbols:
        symbol = random.choice(symbols)
        
        # 随机变异一个参数
        param = random.choice(['primary_factor', 'secondary_factor', 'window1', 'window2', 'weight1', 'weight2'])
        
        if param in ['primary_factor', 'secondary_factor']:
            mutated[symbol][param] = random.choice(list(context.function_library.keys()))
        elif param in ['window1', 'window2']:
            mutated[symbol][param] = random.randint(5, 20)
        else:  # weight1, weight2
            mutated[symbol][param] = random.uniform(0.3, 0.7)
    
    return mutated

def generate_trading_signals(context, bar_dict, symbols, best_factors):
    """基于最优因子生成交易信号"""
    signals = {}
    
    for symbol in symbols:
        if symbol in best_factors:
            factor_score = calculate_factor_score(context, symbol, best_factors[symbol])
            
            if factor_score is not None:
                # 标准化信号强度
                if factor_score > 0.02:      # 强买入
                    signals[symbol] = 1.0
                elif factor_score > 0.01:    # 买入
                    signals[symbol] = 0.5
                elif factor_score < -0.02:   # 强卖出
                    signals[symbol] = -1.0
                elif factor_score < -0.01:   # 卖出
                    signals[symbol] = -0.5
                else:                        # 中性
                    signals[symbol] = 0.0
            else:
                signals[symbol] = 0.0
    
    print(f"交易信号生成：{len([s for s in signals.values() if s != 0])}个非零信号")
    return signals

def execute_trades(context, bar_dict, signals):
    """执行交易"""
    account = context.future_account_dict[context.future_account]
    total_value = account.total_value
    
    for symbol, signal in signals.items():
        if symbol not in bar_dict:
            continue
        
        current_price = bar_dict[symbol].close
        current_position = account.positions.get(symbol)
        current_quantity = current_position.buy_quantity - current_position.sell_quantity if current_position else 0
        
        # 计算目标仓位
        target_value = total_value * context.position_size * abs(signal)
        
        # 获取合约乘数（简化处理）
        contract_multiplier = get_contract_multiplier(symbol)
        target_quantity = int(target_value / (current_price * contract_multiplier))
        
        if signal > 0:  # 做多信号
            quantity_diff = target_quantity - max(0, current_quantity)
            if quantity_diff > 0:
                print(f"买开 {symbol}: {quantity_diff}手")
                buy_open(context.future_account, symbol, quantity_diff)
                context.position_costs[symbol] = current_price
            elif current_quantity < 0:  # 平空仓
                close_quantity = min(abs(current_quantity), target_quantity)
                print(f"买平 {symbol}: {close_quantity}手")
                buy_close(context.future_account, symbol, close_quantity)
        
        elif signal < 0:  # 做空信号
            quantity_diff = target_quantity - max(0, -current_quantity)
            if quantity_diff > 0:
                print(f"卖开 {symbol}: {quantity_diff}手")
                sell_open(context.future_account, symbol, quantity_diff)
                context.position_costs[symbol] = current_price
            elif current_quantity > 0:  # 平多仓
                close_quantity = min(current_quantity, target_quantity)
                print(f"卖平 {symbol}: {close_quantity}手")
                sell_close(context.future_account, symbol, close_quantity)
        
        else:  # 平仓信号
            if current_quantity > 0:
                print(f"平多 {symbol}: {current_quantity}手")
                sell_close(context.future_account, symbol, current_quantity)
                if symbol in context.position_costs:
                    del context.position_costs[symbol]
            elif current_quantity < 0:
                print(f"平空 {symbol}: {abs(current_quantity)}手")
                buy_close(context.future_account, symbol, abs(current_quantity))
                if symbol in context.position_costs:
                    del context.position_costs[symbol]

def get_contract_multiplier(symbol):
    """获取合约乘数（简化版本）"""
    multipliers = {
        'AL': 5, 'CU': 5, 'ZN': 5, 'NI': 1, 'SN': 1,
        'RB': 10, 'SC': 1000, 'FU': 10, 'AU': 1000, 'AG': 15
    }
    
    # 提取品种代码
    for code in multipliers:
        if symbol.startswith(code):
            return multipliers[code]
    
    return 10  # 默认乘数

def risk_management(context, bar_dict):
    """风险管理"""
    account = context.future_account_dict[context.future_account]
    positions = account.positions
    
    for symbol, position in positions.items():
        net_quantity = position.buy_quantity - position.sell_quantity
        if net_quantity == 0 or symbol not in bar_dict:
            continue
        
        current_price = bar_dict[symbol].close
        cost_price = context.position_costs.get(symbol)
        
        if cost_price is None:
            continue
        
        # 计算收益率
        if net_quantity > 0:  # 多头
            return_rate = (current_price - cost_price) / cost_price
        else:  # 空头
            return_rate = (cost_price - current_price) / cost_price
        
        # 止损
        if return_rate <= context.stop_loss:
            print(f"止损 {symbol}: 收益率{return_rate*100:.1f}%")
            if net_quantity > 0:
                sell_close(context.future_account, symbol, net_quantity)
            else:
                buy_close(context.future_account, symbol, abs(net_quantity))
            
            if symbol in context.position_costs:
                del context.position_costs[symbol]
        
        # 止盈
        elif return_rate >= context.take_profit:
            print(f"止盈 {symbol}: 收益率{return_rate*100:.1f}%")
            if net_quantity > 0:
                sell_close(context.future_account, symbol, net_quantity)
            else:
                buy_close(context.future_account, symbol, abs(net_quantity))
            
            if symbol in context.position_costs:
                del context.position_costs[symbol]

# 函数库实现（简化版时序处理函数）
def ts_rank(data, window):
    """时序排名函数"""
    if len(data) < window:
        return None
    recent_data = data[-window:]
    current_value = recent_data[-1]
    rank = sum(1 for x in recent_data if x <= current_value)
    return rank / window

def ts_mean(data, window):
    """时序平均值"""
    if len(data) < window:
        return None
    return np.mean(data[-window:])

def ts_std(data, window):
    """时序标准差"""
    if len(data) < window:
        return None
    return np.std(data[-window:])

def ts_max(data, window):
    """时序最大值"""
    if len(data) < window:
        return None
    return np.max(data[-window:])

def ts_min(data, window):
    """时序最小值"""
    if len(data) < window:
        return None
    return np.min(data[-window:])

def ts_delta(data, window):
    """时序变化率"""
    if len(data) < window:
        return None
    return (data[-1] - data[-window]) / data[-window] if data[-window] != 0 else 0

def after_trading(context):
    """收盘后处理"""
    account = context.future_account_dict[context.future_account]
    positions = account.positions
    
    # 统计持仓
    total_positions = sum(1 for pos in positions.values() 
                         if pos.buy_quantity > 0 or pos.sell_quantity > 0)
    
    print(f"[{context.now}] 收盘统计：")
    print(f"  - 持仓合约数：{total_positions}")
    print(f"  - 账户总权益：{account.total_value:.2f}")
    print(f"  - 持仓盈亏：{account.holding_pnl:.2f}")
    print(f"  - 可用资金：{account.cash:.2f}")

# 交易回报函数
def on_future_trade_rtn(_context, order):
    """期货交易回报"""
    side_text = "买入" if order.side == 1 else "卖出"
    effect_text = "开仓" if order.effect == 0 else "平仓"
    print(f"交易回报 - {order.order_book_id}: {side_text}{effect_text} {order.filled_quantity}手")

def future_order_cancel(_context, order):
    """期货撤单回报"""
    print(f"订单撤销 - {order.order_book_id}: {order.quantity}手")