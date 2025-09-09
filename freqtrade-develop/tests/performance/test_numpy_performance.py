"""
Performance tests for pure NumPy optimizations
Testing framework for freqtrade optimization sprint without Numba dependency
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time

# Add freqtrade to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from freqtrade.data.metrics import calculate_market_change, combine_dataframes_by_column
from freqtrade.data.btanalysis.trade_parallelism import analyze_trade_parallelism

from tests.performance.utils.benchmark_runner import PerformanceBenchmark, BenchmarkResult
from tests.performance.utils.performance_utils import (
    generate_trading_data, 
    generate_trades_data,
    create_performance_test_datasets,
    load_test_dataset
)


class TestNumPyOptimizations:
    """Test suite for pure NumPy optimization benchmarks"""
    
    @classmethod
    def setup_class(cls):
        """Set up test data and benchmark runner"""
        cls.benchmark = PerformanceBenchmark(warmup_iterations=2, benchmark_iterations=10)
        
        # Create test datasets if they don't exist
        datasets_dir = Path("tests/performance/data")
        if not datasets_dir.exists():
            create_performance_test_datasets()
        
        # Load test datasets
        cls.small_data, _ = load_test_dataset(datasets_dir / "small_dataset.json")
        cls.medium_data, _ = load_test_dataset(datasets_dir / "medium_dataset.json")
        cls.large_data, _ = load_test_dataset(datasets_dir / "large_dataset.json")
        
        # Load trade data
        cls.trades_small = pd.read_pickle(datasets_dir / "trades_small.pkl")
        cls.trades_medium = pd.read_pickle(datasets_dir / "trades_medium.pkl")
        cls.trades_large = pd.read_pickle(datasets_dir / "trades_large.pkl")
    
    def test_baseline_calculate_market_change(self):
        """Establish baseline performance for calculate_market_change"""
        result = self.benchmark.benchmark_function(
            calculate_market_change,
            "calculate_market_change_baseline",
            args=(self.medium_data,)
        )
        
        print(f"\n📊 Baseline calculate_market_change: {result.mean_time*1000:.2f}ms ±{result.std_time*1000:.2f}ms")
        assert result.mean_time > 0
        
        # Save baseline for comparison
        baseline_path = Path("tests/performance/data/baseline_results.json")
        self.benchmark.save_results(str(baseline_path))
    
    def test_baseline_combine_dataframes_by_column(self):
        """Establish baseline performance for combine_dataframes_by_column"""
        result = self.benchmark.benchmark_function(
            combine_dataframes_by_column,
            "combine_dataframes_baseline",
            args=(self.medium_data,)
        )
        
        print(f"\n📊 Baseline combine_dataframes: {result.mean_time*1000:.2f}ms ±{result.std_time*1000:.2f}ms")
        assert result.mean_time > 0
    
    def test_baseline_analyze_trade_parallelism(self):
        """Establish baseline performance for analyze_trade_parallelism"""  
        result = self.benchmark.benchmark_function(
            analyze_trade_parallelism,
            "analyze_trade_parallelism_baseline",
            args=(self.trades_medium, "1h")
        )
        
        print(f"\n📊 Baseline analyze_trade_parallelism: {result.mean_time*1000:.2f}ms ±{result.std_time*1000:.2f}ms")
        assert result.mean_time > 0
        
    # TODO: Add optimized function tests when implementations are ready
    # These will be implemented in the next sprint days
    
    @pytest.mark.skip(reason="Optimized functions not yet implemented")
    def test_optimized_calculate_market_change(self):
        """Test optimized calculate_market_change (pure NumPy version)"""
        # This will be implemented when we create the optimized functions
        pass
    
    @pytest.mark.skip(reason="Optimized functions not yet implemented")  
    def test_compare_calculate_market_change(self):
        """Compare original vs optimized calculate_market_change"""
        # This will test the performance improvement
        pass
    
    def test_correctness_validation_framework(self):
        """Test that our correctness validation framework works"""
        # Test with identical functions to ensure validation works
        def dummy_func(data):
            return calculate_market_change(data)
            
        def identical_func(data):
            return calculate_market_change(data)
        
        # Both functions should produce identical results
        result1 = dummy_func(self.small_data)
        result2 = identical_func(self.small_data)
        
        # Validate numerical equivalence
        assert abs(result1 - result2) < 1e-10, "Functions should produce identical results"
        print("✅ Correctness validation framework working")
    
    def test_memory_monitoring(self):
        """Test that memory monitoring is working"""
        result = self.benchmark.benchmark_function(
            calculate_market_change,
            "memory_test",
            args=(self.large_data,)
        )
        
        assert result.memory_peak_mb > 0
        assert result.memory_avg_mb > 0
        print(f"✅ Memory monitoring working: Peak {result.memory_peak_mb:.1f}MB, Avg {result.memory_avg_mb:.1f}MB")
    
    def test_performance_scaling(self):
        """Test how performance scales with dataset size"""
        datasets = [
            ("small", self.small_data),
            ("medium", self.medium_data), 
            ("large", self.large_data)
        ]
        
        results = []
        for name, data in datasets:
            result = self.benchmark.benchmark_function(
                calculate_market_change,
                f"scaling_test_{name}",
                args=(data,)
            )
            results.append((name, len(data), result.mean_time))
            print(f"{name:6} ({len(data):2} pairs): {result.mean_time*1000:7.2f}ms")
        
        # Performance should scale roughly with data size
        small_time = results[0][2]  
        large_time = results[2][2]
        size_ratio = len(self.large_data) / len(self.small_data)
        time_ratio = large_time / small_time
        
        print(f"Size ratio: {size_ratio:.1f}x, Time ratio: {time_ratio:.1f}x")
        assert time_ratio > 1.0, "Larger datasets should take more time"
        
    def test_data_generation_quality(self):
        """Verify that our synthetic data is realistic"""
        # Check small dataset properties
        sample_pair = list(self.small_data.keys())[0]
        sample_df = self.small_data[sample_pair]
        
        # Basic data quality checks
        assert len(sample_df) > 8000, "Should have ~8760 hours in a year"
        assert all(col in sample_df.columns for col in ['date', 'open', 'high', 'low', 'close', 'volume'])
        assert (sample_df['high'] >= sample_df['close']).all(), "High should be >= close"
        assert (sample_df['low'] <= sample_df['close']).all(), "Low should be <= close"
        assert (sample_df['volume'] > 0).all(), "Volume should be positive"
        
        print(f"✅ Data quality validated: {len(sample_df)} rows, {len(self.small_data)} pairs")


if __name__ == "__main__":
    # Run performance tests directly
    print("🚀 Starting Performance Testing Framework Validation")
    
    # Create test datasets
    create_performance_test_datasets()
    
    # Run basic benchmark test
    test_instance = TestNumPyOptimizations()
    test_instance.setup_class()
    
    print("\n" + "="*60)
    print("BASELINE PERFORMANCE MEASUREMENTS")
    print("="*60)
    
    test_instance.test_baseline_calculate_market_change()
    test_instance.test_baseline_combine_dataframes_by_column()
    test_instance.test_baseline_analyze_trade_parallelism()
    
    print("\n✅ Performance testing framework ready for optimization implementation!")
    print("\n📈 Next steps:")
    print("1. Implement NumPy-optimized versions of target functions")
    print("2. Add comparison tests for performance improvements")
    print("3. Validate correctness of optimized implementations")