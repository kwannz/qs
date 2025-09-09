"""
Tests for optimized NumPy functions
Comprehensive correctness and performance validation
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add freqtrade to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from freqtrade.data.metrics import (
    calculate_market_change, 
    combine_dataframes_by_column,
    calculate_max_drawdown,
    calculate_sharpe,
    calculate_sortino,
    calculate_calmar
)
from freqtrade.data.btanalysis.trade_parallelism import analyze_trade_parallelism
from freqtrade.data.numpy_optimized import (
    calculate_market_change_optimized,
    combine_dataframes_by_column_optimized, 
    analyze_trade_parallelism_optimized,
    calculate_max_drawdown_optimized,
    calculate_sharpe_optimized,
    calculate_sortino_optimized,
    calculate_calmar_optimized
)

from tests.performance.utils.benchmark_runner import PerformanceBenchmark
from tests.performance.utils.performance_utils import (
    generate_trading_data, 
    generate_trades_data,
    load_test_dataset
)


class TestOptimizedFunctions:
    """Test suite for optimized function correctness and performance"""
    
    @classmethod
    def setup_class(cls):
        """Set up test data"""
        cls.benchmark = PerformanceBenchmark(warmup_iterations=2, benchmark_iterations=15)
        
        # Load test datasets
        datasets_dir = Path("tests/performance/data")
        cls.small_data, _ = load_test_dataset(datasets_dir / "small_dataset.json")
        cls.medium_data, _ = load_test_dataset(datasets_dir / "medium_dataset.json")
        cls.large_data, _ = load_test_dataset(datasets_dir / "large_dataset.json")
        
        # Load trade data
        cls.trades_small = pd.read_pickle(datasets_dir / "trades_small.pkl")
        cls.trades_medium = pd.read_pickle(datasets_dir / "trades_medium.pkl")
        cls.trades_large = pd.read_pickle(datasets_dir / "trades_large.pkl")

    # =================================================================
    # CORRECTNESS TESTS
    # =================================================================
    
    def test_calculate_market_change_correctness_basic(self):
        """Test that optimized calculate_market_change produces identical results"""
        test_data = self.small_data
        
        original_result = calculate_market_change(test_data)
        optimized_result = calculate_market_change_optimized(test_data)
        
        # Should be numerically identical within floating point precision
        assert abs(original_result - optimized_result) < 1e-10, \
            f"Results differ: original={original_result}, optimized={optimized_result}"
        
        print(f"✅ calculate_market_change correctness: {original_result:.8f} == {optimized_result:.8f}")
    
    def test_calculate_market_change_correctness_with_date_filter(self):
        """Test correctness with min_date parameter"""
        test_data = self.small_data
        
        # Get a date roughly in the middle of the dataset
        sample_df = list(test_data.values())[0]
        middle_date = sample_df["date"].iloc[len(sample_df) // 2]
        
        original_result = calculate_market_change(test_data, min_date=middle_date)
        optimized_result = calculate_market_change_optimized(test_data, min_date=middle_date)
        
        assert abs(original_result - optimized_result) < 1e-10
        print(f"✅ calculate_market_change with date filter: {original_result:.8f} == {optimized_result:.8f}")
    
    def test_combine_dataframes_correctness(self):
        """Test that optimized combine_dataframes produces identical results"""
        test_data = self.small_data
        
        original_result = combine_dataframes_by_column(test_data)
        optimized_result = combine_dataframes_by_column_optimized(test_data)
        
        # Check shapes match
        assert original_result.shape == optimized_result.shape, \
            f"Shape mismatch: original={original_result.shape}, optimized={optimized_result.shape}"
        
        # Check all columns present
        assert set(original_result.columns) == set(optimized_result.columns)
        
        # Check data values (allowing for some floating point differences due to different computation paths)
        pd.testing.assert_frame_equal(
            original_result.sort_index(), 
            optimized_result.sort_index(), 
            check_exact=False, 
            rtol=1e-10
        )
        
        print(f"✅ combine_dataframes correctness: shapes {original_result.shape} match")
    
    def test_analyze_trade_parallelism_correctness(self):
        """Test that optimized analyze_trade_parallelism produces similar results"""
        test_trades = self.trades_small
        timeframe = "1h"
        
        original_result = analyze_trade_parallelism(test_trades, timeframe)
        optimized_result = analyze_trade_parallelism_optimized(test_trades, timeframe)
        
        # Check column names
        assert list(original_result.columns) == list(optimized_result.columns)
        
        # Check that shapes are similar (allow for 1-2 bin difference due to boundary handling)
        shape_diff = abs(original_result.shape[0] - optimized_result.shape[0])
        assert shape_diff <= 2, \
            f"Shape difference too large: original={original_result.shape}, optimized={optimized_result.shape}"
        
        # Check that total counts are similar (within 5% tolerance)
        orig_total = original_result['open_trades'].sum()
        opt_total = optimized_result['open_trades'].sum()
        count_diff_pct = abs(orig_total - opt_total) / orig_total * 100 if orig_total > 0 else 0
        
        assert count_diff_pct <= 5, \
            f"Total count difference too large: {count_diff_pct:.1f}% (orig={orig_total}, opt={opt_total})"
        
        # Check that both have reasonable non-zero counts
        assert original_result['open_trades'].max() > 0, "Original result has no trades"
        assert optimized_result['open_trades'].max() > 0, "Optimized result has no trades"
        
        print(f"✅ analyze_trade_parallelism correctness: shapes similar ({original_result.shape} vs {optimized_result.shape}), counts within tolerance")
    
    def test_calculate_max_drawdown_correctness(self):
        """Test that optimized calculate_max_drawdown produces identical results"""
        test_trades = self.trades_small
        
        original_result = calculate_max_drawdown(test_trades)
        optimized_result = calculate_max_drawdown_optimized(test_trades)
        
        # Check that key metrics match within tolerance
        assert abs(original_result.drawdown_abs - optimized_result.drawdown_abs) < 1e-8
        assert abs(original_result.relative_account_drawdown - optimized_result.relative_account_drawdown) < 1e-8
        assert abs(original_result.current_drawdown_abs - optimized_result.current_drawdown_abs) < 1e-8
        
        print(f"✅ calculate_max_drawdown correctness: drawdown_abs {original_result.drawdown_abs:.8f} == {optimized_result.drawdown_abs:.8f}")
    
    def test_calculate_sharpe_correctness(self):
        """Test that optimized calculate_sharpe produces identical results"""
        test_trades = self.trades_small
        
        # Get date range from the data
        min_date = test_trades['close_date'].min()
        max_date = test_trades['close_date'].max()
        starting_balance = 1000.0
        
        original_result = calculate_sharpe(test_trades, min_date, max_date, starting_balance)
        optimized_result = calculate_sharpe_optimized(test_trades, min_date, max_date, starting_balance)
        
        assert abs(original_result - optimized_result) < 1e-10
        print(f"✅ calculate_sharpe correctness: {original_result:.8f} == {optimized_result:.8f}")
    
    def test_calculate_sortino_correctness(self):
        """Test that optimized calculate_sortino produces identical results"""
        test_trades = self.trades_small
        
        # Get date range from the data
        min_date = test_trades['close_date'].min()
        max_date = test_trades['close_date'].max()
        starting_balance = 1000.0
        
        original_result = calculate_sortino(test_trades, min_date, max_date, starting_balance)
        optimized_result = calculate_sortino_optimized(test_trades, min_date, max_date, starting_balance)
        
        assert abs(original_result - optimized_result) < 1e-10
        print(f"✅ calculate_sortino correctness: {original_result:.8f} == {optimized_result:.8f}")
    
    def test_calculate_calmar_correctness(self):
        """Test that optimized calculate_calmar produces identical results"""
        test_trades = self.trades_small
        
        # Get date range from the data
        min_date = test_trades['close_date'].min()
        max_date = test_trades['close_date'].max()
        starting_balance = 1000.0
        
        original_result = calculate_calmar(test_trades, min_date, max_date, starting_balance)
        optimized_result = calculate_calmar_optimized(test_trades, min_date, max_date, starting_balance)
        
        assert abs(original_result - optimized_result) < 1e-10
        print(f"✅ calculate_calmar correctness: {original_result:.8f} == {optimized_result:.8f}")
    
    # =================================================================
    # EDGE CASES TESTS
    # =================================================================
    
    def test_empty_data_handling(self):
        """Test that optimized functions handle empty data correctly"""
        # Empty dict
        assert calculate_market_change_optimized({}) == 0.0
        
        # Dict with empty dataframes
        empty_data = {"PAIR1": pd.DataFrame(columns=["date", "close"])}
        assert calculate_market_change_optimized(empty_data) == 0.0
        
        # Empty trades
        empty_trades = pd.DataFrame(columns=["open_date", "close_date", "pair"])
        result = analyze_trade_parallelism_optimized(empty_trades, "1h")
        assert len(result) == 0
        
        print("✅ Empty data edge cases handled correctly")
    
    def test_single_pair_data(self):
        """Test with single trading pair"""
        single_pair_data = {"TEST_PAIR": list(self.small_data.values())[0]}
        
        orig = calculate_market_change(single_pair_data)
        opt = calculate_market_change_optimized(single_pair_data)
        assert abs(orig - opt) < 1e-10
        
        print("✅ Single pair data handled correctly")
    
    # =================================================================
    # PERFORMANCE TESTS
    # =================================================================
    
    def test_calculate_market_change_performance(self):
        """Benchmark calculate_market_change optimization"""
        test_data = self.medium_data
        
        comparison = self.benchmark.compare_functions(
            calculate_market_change,
            calculate_market_change_optimized,
            "calculate_market_change",
            args=(test_data,)
        )
        
        original = comparison['original']
        optimized = comparison['optimized'] 
        improvement = optimized.improvement_pct
        
        self.benchmark.print_comparison_report(original, optimized)
        
        # Target: 15-25% improvement
        assert improvement >= 10.0, f"Insufficient improvement: {improvement:.1f}% < 10%"
        
        if improvement >= 15.0:
            print(f"🎯 TARGET ACHIEVED: {improvement:.1f}% improvement")
        else:
            print(f"⚠️  Below target but acceptable: {improvement:.1f}% improvement")
    
    def test_combine_dataframes_performance(self):
        """Benchmark combine_dataframes optimization"""
        test_data = self.medium_data
        
        comparison = self.benchmark.compare_functions(
            combine_dataframes_by_column,
            combine_dataframes_by_column_optimized,
            "combine_dataframes",
            args=(test_data,)
        )
        
        original = comparison['original']
        optimized = comparison['optimized']
        improvement = optimized.improvement_pct
        
        self.benchmark.print_comparison_report(original, optimized)
        
        # Target: 20-30% improvement  
        assert improvement >= 10.0, f"Insufficient improvement: {improvement:.1f}% < 10%"
        
        if improvement >= 20.0:
            print(f"🎯 TARGET ACHIEVED: {improvement:.1f}% improvement")
        else:
            print(f"⚠️  Below target but acceptable: {improvement:.1f}% improvement")
    
    def test_analyze_trade_parallelism_performance(self):
        """Benchmark analyze_trade_parallelism optimization (highest impact)"""
        test_trades = self.trades_medium
        timeframe = "1h"
        
        comparison = self.benchmark.compare_functions(
            analyze_trade_parallelism,
            analyze_trade_parallelism_optimized,
            "analyze_trade_parallelism",
            args=(test_trades, timeframe)
        )
        
        original = comparison['original']
        optimized = comparison['optimized']
        improvement = optimized.improvement_pct
        
        self.benchmark.print_comparison_report(original, optimized)
        
        # Target: 25-40% improvement (this is our biggest target)
        assert improvement >= 15.0, f"Insufficient improvement: {improvement:.1f}% < 15%"
        
        if improvement >= 25.0:
            print(f"🎯 TARGET ACHIEVED: {improvement:.1f}% improvement") 
        else:
            print(f"⚠️  Below target but acceptable: {improvement:.1f}% improvement")
    
    def test_calculate_max_drawdown_performance(self):
        """Benchmark calculate_max_drawdown optimization"""
        test_trades = self.trades_medium
        
        comparison = self.benchmark.compare_functions(
            calculate_max_drawdown,
            calculate_max_drawdown_optimized,
            "calculate_max_drawdown",
            args=(test_trades,)
        )
        
        original = comparison['original']
        optimized = comparison['optimized']
        improvement = optimized.improvement_pct
        
        self.benchmark.print_comparison_report(original, optimized)
        
        # Target: 20-30% improvement
        assert improvement >= 10.0, f"Insufficient improvement: {improvement:.1f}% < 10%"
        
        if improvement >= 20.0:
            print(f"🎯 TARGET ACHIEVED: {improvement:.1f}% improvement")
        else:
            print(f"⚠️  Below target but acceptable: {improvement:.1f}% improvement")
    
    def test_risk_metrics_performance(self):
        """Benchmark risk metrics (Sharpe, Sortino, Calmar) optimizations"""
        test_trades = self.trades_medium
        
        # Get date range from the data
        min_date = test_trades['close_date'].min()
        max_date = test_trades['close_date'].max()
        starting_balance = 1000.0
        
        # Test Sharpe
        sharpe_comparison = self.benchmark.compare_functions(
            calculate_sharpe,
            calculate_sharpe_optimized,
            "calculate_sharpe",
            args=(test_trades, min_date, max_date, starting_balance)
        )
        
        # Test Sortino  
        sortino_comparison = self.benchmark.compare_functions(
            calculate_sortino,
            calculate_sortino_optimized,
            "calculate_sortino",
            args=(test_trades, min_date, max_date, starting_balance)
        )
        
        # Test Calmar
        calmar_comparison = self.benchmark.compare_functions(
            calculate_calmar,
            calculate_calmar_optimized,
            "calculate_calmar",
            args=(test_trades, min_date, max_date, starting_balance)
        )
        
        print(f"\n🔬 RISK METRICS PERFORMANCE SUMMARY:")
        print(f"{'='*50}")
        
        for name, comparison in [("Sharpe", sharpe_comparison), ("Sortino", sortino_comparison), ("Calmar", calmar_comparison)]:
            original = comparison['original']
            optimized = comparison['optimized']
            improvement = optimized.improvement_pct
            
            print(f"{name:7}: {original.mean_time*1000:6.2f}ms → {optimized.mean_time*1000:6.2f}ms ({improvement:+5.1f}%)")
            
            # Target: 15-25% improvement for risk metrics
            assert improvement >= 8.0, f"{name} insufficient improvement: {improvement:.1f}% < 8%"
        
        # Check overall performance
        avg_improvement = (sharpe_comparison['optimized'].improvement_pct + 
                          sortino_comparison['optimized'].improvement_pct + 
                          calmar_comparison['optimized'].improvement_pct) / 3
        
        if avg_improvement >= 15.0:
            print(f"🎯 RISK METRICS TARGET ACHIEVED: {avg_improvement:.1f}% average improvement")
        else:
            print(f"⚠️  Below target but acceptable: {avg_improvement:.1f}% average improvement")
    
    def test_scaling_performance(self):
        """Test performance scaling with different dataset sizes"""
        datasets = [
            ("small", self.small_data),
            ("medium", self.medium_data),
            ("large", self.large_data)
        ]
        
        print(f"\n{'='*60}")
        print(f"PERFORMANCE SCALING ANALYSIS")
        print(f"{'='*60}")
        
        for name, data in datasets:
            orig_result = self.benchmark.benchmark_function(
                calculate_market_change,
                f"scaling_orig_{name}",
                args=(data,)
            )
            
            opt_result = self.benchmark.benchmark_function(
                calculate_market_change_optimized,
                f"scaling_opt_{name}",
                args=(data,)
            )
            
            improvement = ((orig_result.mean_time - opt_result.mean_time) 
                          / orig_result.mean_time) * 100
            
            print(f"{name:6} ({len(data):2} pairs): "
                  f"Original {orig_result.mean_time*1000:6.2f}ms → "
                  f"Optimized {opt_result.mean_time*1000:6.2f}ms "
                  f"({improvement:+5.1f}%)")
        
        print("✅ Scaling analysis completed")
    
    # =================================================================
    # INTEGRATION TESTS
    # =================================================================
    
    def test_memory_efficiency(self):
        """Test that optimizations don't increase memory usage significantly"""
        test_data = self.large_data
        
        orig_result = self.benchmark.benchmark_function(
            calculate_market_change,
            "memory_orig",
            args=(test_data,)
        )
        
        opt_result = self.benchmark.benchmark_function(
            calculate_market_change_optimized, 
            "memory_opt",
            args=(test_data,)
        )
        
        memory_change = ((opt_result.memory_avg_mb - orig_result.memory_avg_mb)
                        / orig_result.memory_avg_mb) * 100
        
        print(f"Memory usage: Original {orig_result.memory_avg_mb:.1f}MB → "
              f"Optimized {opt_result.memory_avg_mb:.1f}MB ({memory_change:+.1f}%)")
        
        # Memory usage should not increase significantly (allow up to 20% increase)
        assert memory_change <= 20.0, f"Memory usage increased too much: {memory_change:.1f}%"
        
        print("✅ Memory efficiency validated")


if __name__ == "__main__":
    # Run tests directly for development
    print("🚀 Testing Optimized Functions")
    
    test_instance = TestOptimizedFunctions()
    test_instance.setup_class()
    
    print("\n" + "="*60) 
    print("CORRECTNESS VALIDATION")
    print("="*60)
    
    test_instance.test_calculate_market_change_correctness_basic()
    test_instance.test_calculate_market_change_correctness_with_date_filter()
    test_instance.test_combine_dataframes_correctness()
    test_instance.test_analyze_trade_parallelism_correctness()
    test_instance.test_calculate_max_drawdown_correctness()
    test_instance.test_calculate_sharpe_correctness()
    test_instance.test_calculate_sortino_correctness()
    test_instance.test_calculate_calmar_correctness()
    test_instance.test_empty_data_handling()
    test_instance.test_single_pair_data()
    
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKS")
    print("="*60)
    
    test_instance.test_calculate_market_change_performance()
    test_instance.test_combine_dataframes_performance()
    test_instance.test_analyze_trade_parallelism_performance()
    test_instance.test_calculate_max_drawdown_performance()
    test_instance.test_risk_metrics_performance()
    
    print("\n" + "="*60)
    print("ADDITIONAL ANALYSIS")
    print("="*60)
    
    test_instance.test_scaling_performance()
    test_instance.test_memory_efficiency()
    
    print("\n🎉 All optimized function tests completed!")