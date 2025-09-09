"""
遗传规划因子计算核心逻辑演示
展示5个因子的具体计算方法和最终合成
"""

import numpy as np
from typing import List, Dict

class SimpleFactorDemo:
    """简化的因子计算演示类"""
    
    def __init__(self):
        # 研报中提到的时序处理函数
        self.functions = {
            'ts_rank': self.ts_rank,
            'ts_mean': self.ts_mean,
            'ts_delta': self.ts_delta,
            'ts_std': self.ts_std
        }
    
    def ts_rank(self, data: List[float], window: int) -> float:
        """
        研报核心函数：ts_rank
        示例：ts_rank([3,5,6,1,10], 5) 
        对过去5个数据进行排序，当前值10在5个数中排名第5，返回 5/5 = 1.0
        """
        if len(data) < window:
            return 0.5
        recent = data[-window:]
        current = recent[-1]
        rank = sum(1 for x in recent if x <= current)
        return rank / window
    
    def ts_mean(self, data: List[float], window: int) -> float:
        """时序均值"""
        if len(data) < window:
            return np.mean(data) if data else 0
        return np.mean(data[-window:])
    
    def ts_delta(self, data: List[float], window: int) -> float:
        """时序变化率：(current - past) / past"""
        if len(data) < window or data[-window] == 0:
            return 0
        return (data[-1] - data[-window]) / data[-window]
    
    def ts_std(self, data: List[float], window: int) -> float:
        """时序标准差"""
        if len(data) < window:
            return 0
        return np.std(data[-window:])

    def calculate_five_factors(self, price_data: Dict[str, List[float]]) -> Dict[str, float]:
        """
        计算5个核心因子并展示计算过程
        
        Args:
            price_data: 包含 'close', 'high', 'low', 'volume' 的价格数据
        """
        closes = price_data['close']
        highs = price_data['high'] 
        lows = price_data['low']
        volumes = price_data['volume']
        
        print("🔍 因子计算过程展示：\n")
        
        # ============ 因子1：动量因子 ============
        print("📈 因子1：动量因子")
        print("公式：ts_rank(close/close[5], 10) * sqrt(ts_rank(volume, 5))")
        
        # 5日收益率
        returns_5d = [closes[i]/closes[i-5] - 1 for i in range(5, len(closes))]
        price_momentum = self.ts_rank(returns_5d, 10)
        volume_momentum = self.ts_rank(volumes, 5)
        
        factor1 = price_momentum * np.sqrt(volume_momentum)
        
        print(f"  - 5日收益率序列(最后5个): {[f'{x:.3f}' for x in returns_5d[-5:]]}")
        print(f"  - 价格动量排名: {price_momentum:.4f}")
        print(f"  - 成交量动量排名: {volume_momentum:.4f}")
        print(f"  - 动量因子值: {factor1:.4f}\n")
        
        # ============ 因子2：波动率因子 ============
        print("📊 因子2：波动率因子")
        print("公式：-ts_std(close, 10) / ts_mean(close, 10)")
        
        price_std = self.ts_std(closes, 10)
        price_mean = self.ts_mean(closes, 10)
        factor2 = -price_std / price_mean if price_mean != 0 else 0
        
        print(f"  - 10日价格标准差: {price_std:.2f}")
        print(f"  - 10日价格均值: {price_mean:.2f}")
        print(f"  - 波动率因子值: {factor2:.4f} (负号表示低波动为好信号)\n")
        
        # ============ 因子3：量价配合因子 ============
        print("🔄 因子3：量价配合因子")  
        print("公式：correlation(ts_delta(volume,5), ts_delta(close,5))")
        
        vol_changes = [self.ts_delta(volumes[:i+1], 5) for i in range(4, len(volumes))]
        price_changes = [self.ts_delta(closes[:i+1], 5) for i in range(4, len(closes))]
        
        if len(vol_changes) >= 5 and len(price_changes) >= 5:
            correlation = np.corrcoef(vol_changes[-10:], price_changes[-10:])[0,1]
            factor3 = correlation if not np.isnan(correlation) else 0
        else:
            factor3 = 0
            
        print(f"  - 成交量5日变化率(最后3个): {[f'{x:.3f}' for x in vol_changes[-3:]]}")
        print(f"  - 价格5日变化率(最后3个): {[f'{x:.3f}' for x in price_changes[-3:]]}")
        print(f"  - 量价相关性: {factor3:.4f}\n")
        
        # ============ 因子4：价格位置因子 ============ 
        print("📍 因子4：价格位置因子")
        print("公式：ts_rank(close, 20) - 0.5  (中心化)")
        
        price_rank = self.ts_rank(closes, 20)
        factor4 = price_rank - 0.5  # 中心化到[-0.5, 0.5]
        
        print(f"  - 当前价格: {closes[-1]:.1f}")
        print(f"  - 20日内排名占比: {price_rank:.4f}")
        print(f"  - 中心化后因子值: {factor4:.4f}\n")
        
        # ============ 因子5：趋势强度因子 ============
        print("📈 因子5：趋势强度因子")
        print("公式：tanh(0.5*ts_delta(close,5) + 0.3*ts_delta(close,10) + 0.2*ts_delta(close,20))")
        
        trend_5d = self.ts_delta(closes, 5)
        trend_10d = self.ts_delta(closes, 10) 
        trend_20d = self.ts_delta(closes, 20)
        
        trend_strength = 0.5 * trend_5d + 0.3 * trend_10d + 0.2 * trend_20d
        factor5 = np.tanh(trend_strength)
        
        print(f"  - 5日趋势: {trend_5d:.4f}")
        print(f"  - 10日趋势: {trend_10d:.4f}")
        print(f"  - 20日趋势: {trend_20d:.4f}")
        print(f"  - 加权趋势强度: {trend_strength:.4f}")
        print(f"  - tanh标准化后: {factor5:.4f}\n")
        
        # ============ 最终因子合成 ============
        print("🏆 最终因子合成")
        print("权重：动量25% + 波动20% + 量价20% + 位置20% + 趋势15%")
        
        weights = [0.25, 0.20, 0.20, 0.20, 0.15]
        factors = [factor1, factor2, factor3, factor4, factor5]
        
        final_factor = sum(w * f for w, f in zip(weights, factors))
        
        print(f"  - 因子1(动量) × 0.25 = {factor1:.4f} × 0.25 = {factor1*0.25:.4f}")
        print(f"  - 因子2(波动) × 0.20 = {factor2:.4f} × 0.20 = {factor2*0.20:.4f}")
        print(f"  - 因子3(量价) × 0.20 = {factor3:.4f} × 0.20 = {factor3*0.20:.4f}")
        print(f"  - 因子4(位置) × 0.20 = {factor4:.4f} × 0.20 = {factor4*0.20:.4f}")
        print(f"  - 因子5(趋势) × 0.15 = {factor5:.4f} × 0.15 = {factor5*0.15:.4f}")
        print(f"  ────────────────────────────────────────")
        print(f"  🎯 最终合成因子: {final_factor:.4f}")
        
        # 交易信号判断
        if final_factor > 0.15:
            signal = "🔵 强多头信号"
        elif final_factor > 0.05:
            signal = "🟢 弱多头信号"
        elif final_factor < -0.15:
            signal = "🔴 强空头信号"
        elif final_factor < -0.05:
            signal = "🟠 弱空头信号"
        else:
            signal = "🟡 中性信号"
            
        print(f"  📊 交易信号: {signal}")
        
        return {
            'momentum': factor1,
            'volatility': factor2, 
            'volume_price': factor3,
            'price_position': factor4,
            'trend_strength': factor5,
            'final_factor': final_factor,
            'signal': signal
        }


def run_factor_demo():
    """运行因子计算演示"""
    
    print("=" * 60)
    print("🧬 遗传规划因子计算核心逻辑演示")
    print("=" * 60)
    
    # 模拟期货价格数据 - 螺纹钢主力合约
    np.random.seed(123)  # 固定随机种子便于复现
    
    # 生成25天的价格数据
    base_price = 3800.0
    data_length = 25
    
    closes, highs, lows, volumes = [], [], [], []
    
    for i in range(data_length):
        # 模拟价格趋势（前期下跌，后期上涨）
        if i < 10:
            trend = -0.008  # 前10天下跌趋势
        else:
            trend = 0.012   # 后15天上涨趋势
            
        # 随机波动
        noise = np.random.normal(0, 0.015)
        change = trend + noise
        
        if i == 0:
            close = base_price
        else:
            close = closes[-1] * (1 + change)
            
        # 生成开高低价
        open_price = close * (1 + np.random.normal(0, 0.005))
        high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.008)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.008)))
        
        # 生成成交量（与价格变化正相关）
        vol_base = 50000
        vol_multiplier = 1 + abs(change) * 3  # 大幅波动时成交量增加
        volume = vol_base * vol_multiplier * (1 + np.random.normal(0, 0.3))
        
        closes.append(round(close, 1))
        highs.append(round(high, 1))  
        lows.append(round(low, 1))
        volumes.append(int(max(volume, 1000)))
    
    # 展示最近几天的数据
    print(f"\n📊 螺纹钢主力合约最近5天数据：")
    print("日期\t\t收盘价\t最高价\t最低价\t成交量")
    print("-" * 50)
    for i in range(-5, 0):
        day = data_length + i + 1
        print(f"第{day:2d}天\t\t{closes[i]:6.1f}\t{highs[i]:6.1f}\t{lows[i]:6.1f}\t{volumes[i]:8d}")
    
    # 准备数据
    price_data = {
        'close': closes,
        'high': highs,
        'low': lows,
        'volume': volumes
    }
    
    # 计算因子
    demo = SimpleFactorDemo()
    result = demo.calculate_five_factors(price_data)
    
    print("\n" + "=" * 60)
    print("✅ 因子计算完成！")
    print("这就是遗传规划中每个个体的因子计算过程")
    print("在实际策略中，会对多个期货品种并行计算")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    run_factor_demo()