"""
遗传规划因子计算模块
基于研报中的时序处理函数库和降维方法

展示5个核心因子的计算逻辑和最终合成结果
"""

import numpy as np
from typing import Dict, List

class FactorCalculator:
    """
    因子计算器
    实现研报中提到的时序处理函数库和因子计算逻辑
    """
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.factor_weights = {
            'momentum_factor': 0.25,      # 动量因子
            'volatility_factor': 0.20,    # 波动率因子  
            'volume_factor': 0.20,        # 成交量因子
            'price_rank_factor': 0.20,    # 价格排名因子
            'trend_factor': 0.15          # 趋势因子
        }
    
    def calculate_all_factors(self, price_data: List[Dict]) -> Dict[str, float]:
        """
        计算所有因子并返回最终合成结果
        
        Args:
            price_data: 价格数据列表，每个元素包含 'open', 'high', 'low', 'close', 'volume'
            
        Returns:
            包含各个因子值和最终合成因子的字典
        """
        if len(price_data) < self.window_size:
            return {'error': 'insufficient_data', 'final_factor': 0.0}
        
        # 提取基础数据序列
        highs = [d['high'] for d in price_data]
        lows = [d['low'] for d in price_data]
        closes = [d['close'] for d in price_data]
        volumes = [d['volume'] for d in price_data]
        
        # 计算5个核心因子
        factors = {}
        
        # 1. 动量因子 - 基于价格变化率和ts_rank
        factors['momentum_factor'] = self._calculate_momentum_factor(closes, volumes)
        
        # 2. 波动率因子 - 基于价格波动性
        factors['volatility_factor'] = self._calculate_volatility_factor(highs, lows, closes)
        
        # 3. 成交量因子 - 基于成交量变化和趋势
        factors['volume_factor'] = self._calculate_volume_factor(volumes, closes)
        
        # 4. 价格排名因子 - 基于ts_rank函数
        factors['price_rank_factor'] = self._calculate_price_rank_factor(closes)
        
        # 5. 趋势因子 - 基于多周期趋势强度
        factors['trend_factor'] = self._calculate_trend_factor(closes)
        
        # 计算最终合成因子
        final_factor = self._synthesize_factors(factors)
        factors['final_factor'] = final_factor
        
        return factors
    
    def _calculate_momentum_factor(self, closes: List[float], volumes: List[float]) -> float:
        """
        动量因子：结合价格动量和成交量确认
        公式：ts_rank(close/close[5], 10) * ts_rank(volume, 5)
        """
        try:
            # 价格动量：5日收益率
            if len(closes) >= 6:
                returns_5d = [closes[i] / closes[i-5] - 1 for i in range(5, len(closes))]
                price_momentum = self.ts_rank(returns_5d, min(10, len(returns_5d)))
            else:
                price_momentum = 0.5
            
            # 成交量动量
            volume_momentum = self.ts_rank(volumes, min(5, len(volumes)))
            
            # 合成动量因子
            momentum_factor = price_momentum * volume_momentum
            
            # 标准化到[-1, 1]
            return np.tanh(momentum_factor)
            
        except Exception as e:
            print(f"计算动量因子出错: {e}")
            return 0.0
    
    def _calculate_volatility_factor(self, highs: List[float], lows: List[float], closes: List[float]) -> float:
        """
        波动率因子：基于价格波动性的反向指标
        高波动通常对应负因子值
        """
        try:
            # 真实波动范围（True Range）
            tr_list = []
            for i in range(1, len(closes)):
                tr = max(
                    highs[i] - lows[i],  # 当日高低价差
                    abs(highs[i] - closes[i-1]),  # 当日高价与昨收差
                    abs(lows[i] - closes[i-1])    # 当日低价与昨收差
                )
                tr_list.append(tr / closes[i-1])  # 标准化
            
            # 平均真实波动率
            if tr_list:
                atr = np.mean(tr_list[-10:])  # 取最近10期
                # 波动率因子为负值（低波动为正信号）
                volatility_factor = -atr * 10  # 放大并取负
            else:
                volatility_factor = 0.0
            
            # 限制在合理范围
            return np.clip(volatility_factor, -1.0, 1.0)
            
        except Exception as e:
            print(f"计算波动率因子出错: {e}")
            return 0.0
    
    def _calculate_volume_factor(self, volumes: List[float], closes: List[float]) -> float:
        """
        成交量因子：基于量价配合度
        公式：correlation(ts_delta(volume, 5), ts_delta(close, 5))
        """
        try:
            if len(volumes) < 6 or len(closes) < 6:
                return 0.0
            
            # 成交量变化
            volume_changes = [volumes[i] / volumes[i-5] - 1 for i in range(5, len(volumes))]
            
            # 价格变化
            price_changes = [closes[i] / closes[i-5] - 1 for i in range(5, len(closes))]
            
            # 计算相关性
            if len(volume_changes) >= 5 and len(price_changes) >= 5:
                min_len = min(len(volume_changes), len(price_changes))
                vol_changes = volume_changes[-min_len:]
                price_changes = price_changes[-min_len:]
                
                correlation = np.corrcoef(vol_changes, price_changes)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                
                return correlation
            
            return 0.0
            
        except Exception as e:
            print(f"计算成交量因子出错: {e}")
            return 0.0
    
    def _calculate_price_rank_factor(self, closes: List[float]) -> float:
        """
        价格排名因子：基于ts_rank函数
        公式：ts_rank(close, 20) - 0.5  (中心化到[-0.5, 0.5])
        """
        try:
            rank_value = self.ts_rank(closes, min(self.window_size, len(closes)))
            # 中心化：从[0,1]转为[-0.5, 0.5]
            return rank_value - 0.5
            
        except Exception as e:
            print(f"计算价格排名因子出错: {e}")
            return 0.0
    
    def _calculate_trend_factor(self, closes: List[float]) -> float:
        """
        趋势因子：基于多周期趋势强度
        结合短期(5日)、中期(10日)、长期(20日)趋势
        """
        try:
            if len(closes) < 20:
                return 0.0
            
            # 短期趋势 (5日)
            short_trend = (closes[-1] / closes[-6] - 1) if len(closes) >= 6 else 0
            
            # 中期趋势 (10日)
            mid_trend = (closes[-1] / closes[-11] - 1) if len(closes) >= 11 else 0
            
            # 长期趋势 (20日)
            long_trend = (closes[-1] / closes[-21] - 1) if len(closes) >= 21 else 0
            
            # 趋势一致性权重
            trend_factor = (short_trend * 0.5 + mid_trend * 0.3 + long_trend * 0.2)
            
            # 标准化
            return np.tanh(trend_factor * 10)
            
        except Exception as e:
            print(f"计算趋势因子出错: {e}")
            return 0.0
    
    def _synthesize_factors(self, factors: Dict[str, float]) -> float:
        """
        合成最终因子
        使用加权平均，权重可以通过遗传规划优化
        """
        try:
            final_score = 0.0
            total_weight = 0.0
            
            for factor_name, weight in self.factor_weights.items():
                if factor_name in factors:
                    final_score += factors[factor_name] * weight
                    total_weight += weight
            
            if total_weight > 0:
                final_score /= total_weight
            
            # 最终标准化
            return np.clip(final_score, -1.0, 1.0)
            
        except Exception as e:
            print(f"合成因子出错: {e}")
            return 0.0
    
    # 时序处理函数库（研报中提到的函数）
    def ts_rank(self, data: List[float], window: int) -> float:
        """
        时序排名函数 - 研报核心函数
        计算当前值在过去window期内的排名占比
        """
        if len(data) < window or window <= 0:
            return 0.5  # 默认中位数
        
        recent_data = data[-window:]
        current_value = recent_data[-1]
        
        # 计算排名
        rank = sum(1 for x in recent_data if x <= current_value)
        return rank / window
    
    def ts_mean(self, data: List[float], window: int) -> float:
        """时序平均值"""
        if len(data) < window:
            return np.mean(data) if data else 0.0
        return np.mean(data[-window:])
    
    def ts_std(self, data: List[float], window: int) -> float:
        """时序标准差"""
        if len(data) < window:
            return np.std(data) if len(data) > 1 else 0.0
        return np.std(data[-window:])
    
    def ts_delta(self, data: List[float], window: int) -> float:
        """时序变化率"""
        if len(data) < window:
            return 0.0
        if data[-window] == 0:
            return 0.0
        return (data[-1] - data[-window]) / data[-window]


def demo_factor_calculation():
    """
    演示因子计算过程
    """
    print("=== 遗传规划因子计算演示 ===\n")
    
    # 创建因子计算器
    calculator = FactorCalculator(window_size=20)
    
    # 模拟期货价格数据（铝主力合约为例）
    np.random.seed(42)  # 保证结果可重现
    
    # 生成30天的模拟价格数据
    base_price = 18500  # 铝期货基准价格
    price_data = []
    
    for i in range(30):
        # 模拟价格波动
        change = np.random.normal(0, 0.02)  # 2%的日波动
        if i == 0:
            price = base_price
        else:
            price = price_data[-1]['close'] * (1 + change)
        
        # 模拟开高低收
        open_price = price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, price) * (1 - abs(np.random.normal(0, 0.01)))
        close_price = price
        volume = np.random.lognormal(10, 0.5)  # 对数正态分布的成交量
        
        price_data.append({
            'date': f"2024-01-{i+1:02d}",
            'open': round(open_price, 1),
            'high': round(high_price, 1),
            'low': round(low_price, 1),
            'close': round(close_price, 1),
            'volume': int(volume)
        })
    
    # 计算因子
    factors = calculator.calculate_all_factors(price_data)
    
    # 显示最近5天的价格数据
    print("📊 最近5天价格数据：")
    print("日期\t\t开盘\t最高\t最低\t收盘\t成交量")
    print("-" * 60)
    for data in price_data[-5:]:
        print(f"{data['date']}\t{data['open']}\t{data['high']}\t{data['low']}\t{data['close']}\t{data['volume']}")
    
    print(f"\n🔍 因子计算结果：")
    print("=" * 50)
    
    factor_descriptions = {
        'momentum_factor': '动量因子（价格动量×成交量动量）',
        'volatility_factor': '波动率因子（低波动为正信号）',
        'volume_factor': '成交量因子（量价配合度）',
        'price_rank_factor': '价格排名因子（ts_rank标准化）',
        'trend_factor': '趋势因子（多周期趋势强度）'
    }
    
    for factor_name, description in factor_descriptions.items():
        if factor_name in factors:
            value = factors[factor_name]
            signal = "🔴强空" if value < -0.3 else "🟠弱空" if value < -0.1 else "🟡中性" if abs(value) <= 0.1 else "🟢弱多" if value < 0.3 else "🔵强多"
            print(f"{description:25s}: {value:8.4f} {signal}")
    
    print("=" * 50)
    final_score = factors.get('final_factor', 0.0)
    final_signal = "🔴强空头" if final_score < -0.3 else "🟠弱空头" if final_score < -0.1 else "🟡中性" if abs(final_score) <= 0.1 else "🟢弱多头" if final_score < 0.3 else "🔵强多头"
    print(f"{'🏆 最终合成因子':25s}: {final_score:8.4f} {final_signal}")
    
    print(f"\n📋 因子权重配置：")
    for factor_name, weight in calculator.factor_weights.items():
        if factor_name.replace('_factor', '') != 'final':
            print(f"  {factor_descriptions.get(factor_name, factor_name):25s}: {weight:.1%}")
    
    print(f"\n💡 交易建议：")
    if final_score > 0.2:
        print("  建议：开多头仓位")
        print("  信号强度：强")
    elif final_score > 0.05:
        print("  建议：轻仓多头")
        print("  信号强度：中等")
    elif final_score < -0.2:
        print("  建议：开空头仓位") 
        print("  信号强度：强")
    elif final_score < -0.05:
        print("  建议：轻仓空头")
        print("  信号强度：中等")
    else:
        print("  建议：观望或平仓")
        print("  信号强度：弱")
    
    return factors


if __name__ == "__main__":
    # 运行演示
    demo_factor_calculation()