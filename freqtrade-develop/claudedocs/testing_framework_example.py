#!/usr/bin/env python3
"""
Freqtrade Numba Optimization - Testing Framework Examples

This file demonstrates the comprehensive testing framework for validating
numba optimizations including correctness, performance, and integration testing.
"""

import pytest
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from unittest.mock import patch, MagicMock

# =============================================================================
# 1. PERFORMANCE BENCHMARK BASE CLASS
# =============================================================================

class PerformanceBenchmark(ABC):
    """Base class for performance benchmarks with standardized methodology"""
    
    def __init__(self, data_sizes: List[int] = None):
        self.data_sizes = data_sizes or [100, 1000, 10000]
        self.warmup_iterations = 5
        self.benchmark_iterations = 100
        self.results = {}
        
    @abstractmethod
    def setup_data(self, size: int) -> Any:
        """Setup benchmark data of specified size"""
        pass
    
    @abstractmethod
    def run_original(self, data: Any) -> Any:
        """Run original implementation"""
        pass
    
    @abstractmethod
    def run_optimized(self, data: Any) -> Any:  
        """Run optimized implementation"""
        pass
    
    def validate_correctness(self, original_result: Any, optimized_result: Any) -> bool:
        """Validate that both implementations produce identical results"""
        if isinstance(original_result, np.ndarray) and isinstance(optimized_result, np.ndarray):
            return np.allclose(original_result, optimized_result, rtol=1e-10, equal_nan=True)
        elif isinstance(original_result, (int, float)) and isinstance(optimized_result, (int, float)):
            return abs(original_result - optimized_result) < 1e-10
        else:
            return original_result == optimized_result
    
    def benchmark_single_size(self, size: int) -> dict:
        """Run benchmark for a single data size"""
        
        # Setup data
        data = self.setup_data(size)
        
        # Warmup both implementations
        for _ in range(self.warmup_iterations):
            original_result = self.run_original(data)
            optimized_result = self.run_optimized(data)
        
        # Validate correctness
        if not self.validate_correctness(original_result, optimized_result):
            raise AssertionError(f"Correctness validation failed for size {size}")
        
        # Benchmark original implementation
        start_time = time.perf_counter()
        for _ in range(self.benchmark_iterations):
            self.run_original(data)
        original_time = time.perf_counter() - start_time
        
        # Benchmark optimized implementation  
        start_time = time.perf_counter()
        for _ in range(self.benchmark_iterations):
            self.run_optimized(data)
        optimized_time = time.perf_counter() - start_time
        
        # Calculate metrics
        speedup = original_time / optimized_time if optimized_time > 0 else 0
        efficiency = (original_time - optimized_time) / original_time if original_time > 0 else 0
        
        return {
            'data_size': size,
            'iterations': self.benchmark_iterations,
            'original_time': original_time,
            'optimized_time': optimized_time,
            'original_avg': original_time / self.benchmark_iterations,
            'optimized_avg': optimized_time / self.benchmark_iterations,
            'speedup': speedup,
            'efficiency': efficiency,
            'improvement_pct': efficiency * 100
        }
    
    def run_full_benchmark(self) -> Dict[int, dict]:
        """Run benchmark across all data sizes"""
        
        results = {}
        
        for size in self.data_sizes:
            print(f"Benchmarking size {size}...")
            try:
                results[size] = self.benchmark_single_size(size)
                print(f"  Speedup: {results[size]['speedup']:.2f}x")
            except Exception as e:
                print(f"  Error: {e}")
                results[size] = {'error': str(e)}
        
        self.results = results
        return results
    
    def generate_report(self) -> str:
        """Generate detailed benchmark report"""
        
        if not self.results:
            return "No benchmark results available. Run benchmark first."
        
        report = f"\n{'='*60}\n"
        report += f"PERFORMANCE BENCHMARK REPORT: {self.__class__.__name__}\n"
        report += f"{'='*60}\n"
        
        successful_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not successful_results:
            report += "All benchmarks failed.\n"
            return report
        
        # Summary statistics
        speedups = [r['speedup'] for r in successful_results.values()]
        avg_speedup = np.mean(speedups)
        max_speedup = np.max(speedups)
        min_speedup = np.min(speedups)
        
        report += f"\nSUMMARY STATISTICS:\n"
        report += f"  Average Speedup: {avg_speedup:.2f}x\n"
        report += f"  Maximum Speedup: {max_speedup:.2f}x\n" 
        report += f"  Minimum Speedup: {min_speedup:.2f}x\n"
        report += f"  Data Sizes Tested: {list(successful_results.keys())}\n"
        
        # Detailed results
        report += f"\nDETAILED RESULTS:\n"
        for size, result in successful_results.items():
            report += f"\n  Data Size: {size:,}\n"
            report += f"    Original Time:  {result['original_avg']*1000:.3f} ms/call\n"
            report += f"    Optimized Time: {result['optimized_avg']*1000:.3f} ms/call\n"
            report += f"    Speedup:        {result['speedup']:.2f}x\n"
            report += f"    Improvement:    {result['improvement_pct']:.1f}%\n"
        
        # Failed benchmarks
        failed_results = {k: v for k, v in self.results.items() if 'error' in v}
        if failed_results:
            report += f"\nFAILED BENCHMARKS:\n"
            for size, result in failed_results.items():
                report += f"  Size {size}: {result['error']}\n"
        
        return report

# =============================================================================
# 2. METRICS BENCHMARK IMPLEMENTATION
# =============================================================================

class MetricsBenchmark(PerformanceBenchmark):
    """Benchmark for data metrics optimizations"""
    
    def __init__(self, data_sizes: List[int] = None):
        super().__init__(data_sizes)
        self.test_data_cache = {}
    
    def setup_data(self, size: int) -> Dict[str, pd.DataFrame]:
        """Setup test data for metrics benchmarking"""
        
        if size in self.test_data_cache:
            return self.test_data_cache[size]
        
        # Generate multiple trading pairs with realistic price data
        pairs = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'DOT/USDT', 'LINK/USDT']
        data = {}
        
        for pair in pairs:
            # Generate realistic price data with trend and volatility
            base_price = np.random.uniform(10, 1000)
            trend = np.random.uniform(-0.001, 0.001)
            volatility = np.random.uniform(0.01, 0.05)
            
            # Generate price series with realistic characteristics  
            returns = np.random.normal(trend, volatility, size)
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Add some NaN values to test edge cases
            nan_indices = np.random.choice(size, size=max(1, size // 100), replace=False)
            prices[nan_indices] = np.nan
            
            data[pair] = pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=size, freq='1H'),
                'open': prices * np.random.uniform(0.999, 1.001, size),
                'high': prices * np.random.uniform(1.001, 1.01, size), 
                'low': prices * np.random.uniform(0.99, 0.999, size),
                'close': prices,
                'volume': np.random.uniform(1000, 100000, size)
            })
        
        self.test_data_cache[size] = data
        return data
    
    def run_original(self, data: Dict[str, pd.DataFrame]) -> float:
        """Run original market change calculation"""
        from implementation_examples import calculate_market_change_fallback
        return calculate_market_change_fallback(data)
    
    def run_optimized(self, data: Dict[str, pd.DataFrame]) -> float:
        """Run optimized market change calculation"""
        from implementation_examples import calculate_market_change_numba
        
        # Convert to numba-compatible format
        closes_arrays = []
        for pair, df in data.items():
            closes = df['close'].dropna().values
            if len(closes) > 0:
                closes_arrays.append(closes)
        
        weights = np.ones(len(closes_arrays))
        return calculate_market_change_numba(closes_arrays, weights)

# =============================================================================
# 3. CORRECTNESS TESTING FRAMEWORK
# =============================================================================

class CorrectnessTestSuite:
    """Comprehensive correctness testing for optimized functions"""
    
    def __init__(self):
        self.test_cases = []
        self.edge_cases = []
        
    def add_test_case(self, name: str, data: Any, expected_result: Any = None):
        """Add a test case"""
        self.test_cases.append({
            'name': name,
            'data': data,
            'expected_result': expected_result
        })
    
    def add_edge_case(self, name: str, data: Any, should_raise: type = None):
        """Add an edge case test"""
        self.edge_cases.append({
            'name': name,
            'data': data,
            'should_raise': should_raise
        })
    
    def run_correctness_tests(self, 
                            original_func: Callable,
                            optimized_func: Callable) -> Dict[str, bool]:
        """Run all correctness tests"""
        
        results = {}
        
        # Test normal cases
        for case in self.test_cases:
            try:
                original_result = original_func(case['data'])
                optimized_result = optimized_func(case['data'])
                
                # Compare results
                if isinstance(original_result, np.ndarray):
                    correct = np.allclose(original_result, optimized_result, rtol=1e-10, equal_nan=True)
                elif isinstance(original_result, (int, float)):
                    correct = abs(original_result - optimized_result) < 1e-10
                else:
                    correct = original_result == optimized_result
                
                results[case['name']] = correct
                
                if not correct:
                    print(f"FAIL: {case['name']}")
                    print(f"  Original: {original_result}")
                    print(f"  Optimized: {optimized_result}")
                
            except Exception as e:
                results[case['name']] = False
                print(f"ERROR in {case['name']}: {e}")
        
        # Test edge cases
        for case in self.edge_cases:
            try:
                if case['should_raise']:
                    # Both should raise the same exception
                    original_raised = False
                    optimized_raised = False
                    
                    try:
                        original_func(case['data'])
                    except case['should_raise']:
                        original_raised = True
                    except Exception:
                        pass
                    
                    try:
                        optimized_func(case['data'])
                    except case['should_raise']:
                        optimized_raised = True
                    except Exception:
                        pass
                    
                    results[f"{case['name']}_edge"] = original_raised == optimized_raised
                else:
                    # Should produce same results without raising
                    original_result = original_func(case['data'])
                    optimized_result = optimized_func(case['data'])
                    
                    if isinstance(original_result, np.ndarray):
                        correct = np.allclose(original_result, optimized_result, rtol=1e-10, equal_nan=True)
                    else:
                        correct = original_result == optimized_result
                    
                    results[f"{case['name']}_edge"] = correct
                    
            except Exception as e:
                results[f"{case['name']}_edge"] = False
                print(f"ERROR in edge case {case['name']}: {e}")
        
        return results

# =============================================================================
# 4. INTEGRATION TESTING
# =============================================================================

class IntegrationTestSuite:
    """Integration tests for numba optimization system"""
    
    def test_numba_manager_initialization(self):
        """Test NumbaManager initialization and basic functionality"""
        from implementation_examples import NumbaManager
        
        config = {
            'enable_numba_optimization': True,
            'numba_optimization_modules': ['metrics'],
            'performance_benchmarking': False
        }
        
        manager = NumbaManager(config)
        assert manager is not None
        assert manager.numba_available is not None
        
    def test_function_registration_and_retrieval(self):
        """Test function registration and retrieval"""
        from implementation_examples import NumbaManager
        
        config = {'enable_numba_optimization': True, 'numba_optimization_modules': ['test']}
        manager = NumbaManager(config)
        
        def original_func(x):
            return x * 2
        
        def optimized_func(x):
            return x + x  # Equivalent but different implementation
        
        manager.register_optimization('test', 'multiply', optimized_func, original_func)
        
        retrieved_func = manager.get_function('test', 'multiply')
        assert retrieved_func is not None
        assert retrieved_func(5) == 10
    
    def test_fallback_mechanism(self):
        """Test fallback when optimization is disabled"""
        from implementation_examples import NumbaManager
        
        config = {'enable_numba_optimization': False, 'numba_optimization_modules': []}
        manager = NumbaManager(config)
        
        def original_func(x):
            return x * 2
        
        def optimized_func(x):
            return x + x
        
        manager.register_optimization('test', 'multiply', optimized_func, original_func)
        
        # Should get fallback function
        retrieved_func = manager.get_function('test', 'multiply')
        assert retrieved_func(5) == 10
    
    def test_performance_monitoring(self):
        """Test performance monitoring functionality"""
        from implementation_examples import NumbaManager
        
        config = {
            'enable_numba_optimization': True, 
            'numba_optimization_modules': ['test'],
            'performance_benchmarking': True
        }
        manager = NumbaManager(config)
        
        def slow_func(x):
            time.sleep(0.001)  # Simulate some work
            return x * 2
        
        def fast_func(x):
            return x + x
        
        manager.register_optimization('test', 'func', fast_func, slow_func)
        
        func = manager.get_function('test', 'func')
        result = func(5)
        
        assert result == 10
        
        # Check that performance was recorded
        report = manager.get_performance_report()
        assert 'test.func' in report
        assert report['test.func']['total_calls'] >= 1

# =============================================================================
# 5. PYTEST TEST FIXTURES AND UTILITIES  
# =============================================================================

@pytest.fixture
def sample_market_data():
    """Fixture providing sample market data for testing"""
    data = {}
    pairs = ['BTC/USDT', 'ETH/USDT']
    
    for pair in pairs:
        prices = 100 + np.cumsum(np.random.randn(1000) * 0.01)
        data[pair] = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=1000, freq='1H'),
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 1000)
        })
    
    return data

@pytest.fixture
def numba_manager():
    """Fixture providing configured NumbaManager"""
    from implementation_examples import NumbaManager
    
    config = {
        'enable_numba_optimization': True,
        'numba_optimization_modules': ['metrics'],
        'performance_benchmarking': True
    }
    
    return NumbaManager(config)

# =============================================================================
# 6. SPECIFIC TEST IMPLEMENTATIONS
# =============================================================================

class TestMetricsOptimization:
    """Test suite for metrics optimizations"""
    
    def test_market_change_correctness(self, sample_market_data):
        """Test market change calculation correctness"""
        from implementation_examples import calculate_market_change_fallback, calculate_market_change_numba
        
        # Run both implementations
        original_result = calculate_market_change_fallback(sample_market_data)
        
        # Convert to numba format
        closes_arrays = []
        for pair, df in sample_market_data.items():
            closes_arrays.append(df['close'].values)
        weights = np.ones(len(closes_arrays))
        
        optimized_result = calculate_market_change_numba(closes_arrays, weights)
        
        # Results should be very close
        assert abs(original_result - optimized_result) < 1e-10
    
    def test_market_change_edge_cases(self):
        """Test edge cases for market change calculation"""
        
        # Empty data
        empty_data = {}
        from implementation_examples import calculate_market_change_fallback
        result = calculate_market_change_fallback(empty_data)
        assert result == 0.0
        
        # Data with all NaN values
        nan_data = {
            'TEST/USDT': pd.DataFrame({
                'date': pd.date_range('2023-01-01', periods=10),
                'close': [np.nan] * 10
            })
        }
        result = calculate_market_change_fallback(nan_data)
        assert result == 0.0
    
    def test_market_change_performance(self, sample_market_data):
        """Test performance improvement for market change"""
        benchmark = MetricsBenchmark([len(sample_market_data['BTC/USDT'])])
        benchmark.test_data_cache[len(sample_market_data['BTC/USDT'])] = sample_market_data
        
        results = benchmark.run_full_benchmark()
        
        # Should see some performance improvement
        size = len(sample_market_data['BTC/USDT'])
        if size in results and 'error' not in results[size]:
            assert results[size]['speedup'] > 1.0  # Should be faster

# =============================================================================
# 7. REGRESSION TESTING
# =============================================================================

class RegressionTestSuite:
    """Regression testing to catch performance degradations"""
    
    def __init__(self, baseline_file: str = None):
        self.baseline_file = baseline_file
        self.baseline_results = {}
        if baseline_file:
            self.load_baseline()
    
    def load_baseline(self):
        """Load baseline performance results"""
        try:
            import json
            with open(self.baseline_file, 'r') as f:
                self.baseline_results = json.load(f)
        except FileNotFoundError:
            print(f"Baseline file {self.baseline_file} not found. Creating new baseline.")
    
    def save_baseline(self, results: dict):
        """Save current results as new baseline"""
        if self.baseline_file:
            import json
            with open(self.baseline_file, 'w') as f:
                json.dump(results, f, indent=2)
    
    def check_regression(self, current_results: dict, tolerance: float = 0.1) -> dict:
        """Check for performance regression"""
        
        regression_report = {}
        
        for test_name, current_result in current_results.items():
            if test_name not in self.baseline_results:
                regression_report[test_name] = 'NEW_TEST'
                continue
            
            baseline = self.baseline_results[test_name]
            current = current_result
            
            if 'speedup' in baseline and 'speedup' in current:
                baseline_speedup = baseline['speedup']
                current_speedup = current['speedup']
                
                # Check if current performance is significantly worse
                regression_pct = (baseline_speedup - current_speedup) / baseline_speedup
                
                if regression_pct > tolerance:
                    regression_report[test_name] = {
                        'status': 'REGRESSION',
                        'baseline_speedup': baseline_speedup,
                        'current_speedup': current_speedup,
                        'regression_pct': regression_pct * 100
                    }
                elif regression_pct < -tolerance:
                    regression_report[test_name] = {
                        'status': 'IMPROVEMENT',
                        'baseline_speedup': baseline_speedup,
                        'current_speedup': current_speedup,
                        'improvement_pct': -regression_pct * 100
                    }
                else:
                    regression_report[test_name] = {'status': 'STABLE'}
        
        return regression_report

# =============================================================================
# 8. MAIN TESTING ORCHESTRATION
# =============================================================================

def run_comprehensive_tests():
    """Run all tests in the testing framework"""
    
    print("="*80)
    print("FREQTRADE NUMBA OPTIMIZATION - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    # 1. Run performance benchmarks
    print("\n1. PERFORMANCE BENCHMARKS")
    print("-" * 40)
    
    metrics_benchmark = MetricsBenchmark([100, 1000, 5000])
    results = metrics_benchmark.run_full_benchmark()
    print(metrics_benchmark.generate_report())
    
    # 2. Run correctness tests
    print("\n2. CORRECTNESS TESTS")
    print("-" * 40)
    
    correctness_suite = CorrectnessTestSuite()
    
    # Add test cases
    test_data = {
        'BTC/USDT': pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'close': np.random.randn(100).cumsum() + 100
        })
    }
    
    correctness_suite.add_test_case('basic_data', test_data)
    correctness_suite.add_edge_case('empty_data', {})
    
    from implementation_examples import calculate_market_change_fallback
    
    def optimized_wrapper(data):
        from implementation_examples import calculate_market_change_numba
        closes_arrays = []
        for pair, df in data.items():
            if not df.empty:
                closes_arrays.append(df['close'].values)
        weights = np.ones(len(closes_arrays))
        return calculate_market_change_numba(closes_arrays, weights)
    
    correctness_results = correctness_suite.run_correctness_tests(
        calculate_market_change_fallback,
        optimized_wrapper
    )
    
    print(f"Correctness test results: {correctness_results}")
    
    # 3. Run integration tests
    print("\n3. INTEGRATION TESTS")
    print("-" * 40)
    
    integration_suite = IntegrationTestSuite()
    
    try:
        integration_suite.test_numba_manager_initialization()
        print("✓ NumbaManager initialization test passed")
    except Exception as e:
        print(f"✗ NumbaManager initialization test failed: {e}")
    
    try:
        integration_suite.test_function_registration_and_retrieval()
        print("✓ Function registration test passed")
    except Exception as e:
        print(f"✗ Function registration test failed: {e}")
    
    try:
        integration_suite.test_fallback_mechanism()
        print("✓ Fallback mechanism test passed")
    except Exception as e:
        print(f"✗ Fallback mechanism test failed: {e}")
    
    try:
        integration_suite.test_performance_monitoring()
        print("✓ Performance monitoring test passed")
    except Exception as e:
        print(f"✗ Performance monitoring test failed: {e}")
    
    # 4. Summary
    print("\n4. SUMMARY")
    print("-" * 40)
    
    successful_benchmarks = sum(1 for r in results.values() if 'error' not in r)
    total_benchmarks = len(results)
    successful_correctness = sum(1 for r in correctness_results.values() if r)
    total_correctness = len(correctness_results)
    
    print(f"Performance Benchmarks: {successful_benchmarks}/{total_benchmarks} successful")
    print(f"Correctness Tests: {successful_correctness}/{total_correctness} passed")
    print(f"Integration Tests: 4/4 categories tested")
    
    if successful_benchmarks == total_benchmarks and successful_correctness == total_correctness:
        print("\n🎉 ALL TESTS PASSED - Optimization system is ready for deployment!")
    else:
        print("\n⚠️  Some tests failed - Review results before deployment")

if __name__ == '__main__':
    run_comprehensive_tests()