"""
Benchmark Runner for Freqtrade Performance Optimizations
Adapted for pure NumPy vectorization approach (no Numba dependency)
"""

import time
import numpy as np
import pandas as pd
import psutil
import gc
from typing import Callable, Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    function_name: str
    mean_time: float
    std_time: float  
    min_time: float
    max_time: float
    iterations: int
    memory_peak_mb: float
    memory_avg_mb: float
    improvement_pct: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'function_name': self.function_name,
            'mean_time_ms': self.mean_time * 1000,
            'std_time_ms': self.std_time * 1000,
            'min_time_ms': self.min_time * 1000,
            'max_time_ms': self.max_time * 1000,
            'iterations': self.iterations,
            'memory_peak_mb': self.memory_peak_mb,
            'memory_avg_mb': self.memory_avg_mb,
            'improvement_pct': self.improvement_pct
        }


class PerformanceBenchmark:
    """Performance benchmarking framework for numerical optimizations"""
    
    def __init__(self, warmup_iterations: int = 3, benchmark_iterations: int = 20):
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.results: Dict[str, BenchmarkResult] = {}
        
    def benchmark_function(self, 
                          func: Callable, 
                          func_name: str,
                          args: Tuple = (), 
                          kwargs: Dict = None) -> BenchmarkResult:
        """
        Benchmark a single function with memory monitoring
        
        Args:
            func: Function to benchmark
            func_name: Name for results tracking  
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            BenchmarkResult with timing and memory statistics
        """
        if kwargs is None:
            kwargs = {}
            
        # Force garbage collection before benchmark
        gc.collect()
        
        # Warmup runs
        for _ in range(self.warmup_iterations):
            func(*args, **kwargs)
            
        # Benchmark runs with memory monitoring
        times = []
        memory_samples = []
        process = psutil.Process()
        
        for _ in range(self.benchmark_iterations):
            gc.collect()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            times.append(end_time - start_time)
            memory_samples.append(memory_after)
            
        # Calculate statistics
        times_array = np.array(times)
        memory_array = np.array(memory_samples)
        
        benchmark_result = BenchmarkResult(
            function_name=func_name,
            mean_time=float(np.mean(times_array)),
            std_time=float(np.std(times_array)),
            min_time=float(np.min(times_array)), 
            max_time=float(np.max(times_array)),
            iterations=self.benchmark_iterations,
            memory_peak_mb=float(np.max(memory_array)),
            memory_avg_mb=float(np.mean(memory_array))
        )
        
        self.results[func_name] = benchmark_result
        return benchmark_result
    
    def compare_functions(self, 
                         original_func: Callable,
                         optimized_func: Callable, 
                         func_name: str,
                         args: Tuple = (),
                         kwargs: Dict = None) -> Dict[str, BenchmarkResult]:
        """
        Compare performance between original and optimized functions
        
        Returns:
            Dict with 'original' and 'optimized' benchmark results
        """
        if kwargs is None:
            kwargs = {}
            
        original_result = self.benchmark_function(
            original_func, f"{func_name}_original", args, kwargs
        )
        
        optimized_result = self.benchmark_function(
            optimized_func, f"{func_name}_optimized", args, kwargs  
        )
        
        # Calculate improvement percentage
        improvement = ((original_result.mean_time - optimized_result.mean_time) 
                      / original_result.mean_time) * 100
        optimized_result.improvement_pct = improvement
        
        return {
            'original': original_result,
            'optimized': optimized_result
        }
    
    def save_results(self, filepath: str):
        """Save benchmark results to JSON file"""
        results_dict = {name: result.to_dict() for name, result in self.results.items()}
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
            
    def load_baseline_results(self, filepath: str) -> Dict[str, Any]:
        """Load baseline results for comparison"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def print_comparison_report(self, original_result: BenchmarkResult, 
                               optimized_result: BenchmarkResult):
        """Print a formatted comparison report"""
        print(f"\n{'='*60}")
        print(f"Performance Comparison: {original_result.function_name}")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'Original':<15} {'Optimized':<15} {'Change':<15}")
        print(f"{'-'*60}")
        
        mean_change = ((original_result.mean_time - optimized_result.mean_time) 
                      / original_result.mean_time) * 100
        
        print(f"{'Mean Time (ms)':<20} {original_result.mean_time*1000:<15.3f} "
              f"{optimized_result.mean_time*1000:<15.3f} {mean_change:+.1f}%")
        
        print(f"{'Std Time (ms)':<20} {original_result.std_time*1000:<15.3f} "
              f"{optimized_result.std_time*1000:<15.3f}")
        
        memory_change = ((optimized_result.memory_avg_mb - original_result.memory_avg_mb) 
                        / original_result.memory_avg_mb) * 100
        
        print(f"{'Avg Memory (MB)':<20} {original_result.memory_avg_mb:<15.1f} "
              f"{optimized_result.memory_avg_mb:<15.1f} {memory_change:+.1f}%")
        
        print(f"{'Iterations':<20} {original_result.iterations:<15} "
              f"{optimized_result.iterations:<15}")
        
        print(f"\n🎯 **Performance Improvement: {mean_change:+.1f}%**")
        
        # Performance assessment
        if mean_change >= 20:
            print("✅ **EXCELLENT**: Target improvement achieved!")
        elif mean_change >= 10:
            print("✅ **GOOD**: Significant improvement")
        elif mean_change >= 5:
            print("⚠️  **MODERATE**: Some improvement")
        else:
            print("❌ **INSUFFICIENT**: Below target improvement")