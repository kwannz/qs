"""
Advanced Performance Monitoring Framework for Freqtrade

This module provides comprehensive performance tracking and analysis capabilities
specifically designed for monitoring optimization effectiveness.

Features:
- Real-time performance metrics collection
- Automatic baseline comparison
- Performance regression detection  
- Memory usage tracking
- Statistical analysis and reporting
- Export capabilities for analysis
"""

import logging
import time
import threading
import psutil
import statistics
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    function_name: str
    execution_time_ms: float
    memory_usage_mb: float
    timestamp: datetime
    optimization_used: bool
    args_count: int = 0
    result_size_bytes: int = 0
    error: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class FunctionStats:
    """Aggregated statistics for a function"""
    name: str
    total_calls: int = 0
    optimized_calls: int = 0
    original_calls: int = 0
    
    # Timing statistics (milliseconds)
    optimized_times: List[float] = field(default_factory=list)
    original_times: List[float] = field(default_factory=list)
    
    # Memory statistics (MB)
    optimized_memory: List[float] = field(default_factory=list)
    original_memory: List[float] = field(default_factory=list)
    
    # Error tracking
    optimized_errors: int = 0
    original_errors: int = 0
    
    # Last measurements
    last_optimized_time: Optional[float] = None
    last_original_time: Optional[float] = None
    
    def add_metric(self, metric: PerformanceMetric):
        """Add a performance metric to statistics"""
        self.total_calls += 1
        
        if metric.optimization_used:
            self.optimized_calls += 1
            self.optimized_times.append(metric.execution_time_ms)
            self.optimized_memory.append(metric.memory_usage_mb)
            self.last_optimized_time = metric.execution_time_ms
            if metric.error:
                self.optimized_errors += 1
        else:
            self.original_calls += 1
            self.original_times.append(metric.execution_time_ms)
            self.original_memory.append(metric.memory_usage_mb)
            self.last_original_time = metric.execution_time_ms
            if metric.error:
                self.original_errors += 1
        
        # Keep only recent measurements to prevent memory bloat
        max_samples = 1000
        if len(self.optimized_times) > max_samples:
            self.optimized_times = self.optimized_times[-max_samples:]
            self.optimized_memory = self.optimized_memory[-max_samples:]
        if len(self.original_times) > max_samples:
            self.original_times = self.original_times[-max_samples:]
            self.original_memory = self.original_memory[-max_samples:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        summary = {
            'function': self.name,
            'total_calls': self.total_calls,
            'optimization_adoption_rate': (
                self.optimized_calls / self.total_calls * 100 
                if self.total_calls > 0 else 0
            ),
            'error_rates': {
                'optimized': (
                    self.optimized_errors / self.optimized_calls * 100
                    if self.optimized_calls > 0 else 0
                ),
                'original': (
                    self.original_errors / self.original_calls * 100
                    if self.original_calls > 0 else 0
                )
            }
        }
        
        # Timing statistics
        if self.optimized_times and self.original_times:
            opt_mean = statistics.mean(self.optimized_times)
            orig_mean = statistics.mean(self.original_times)
            
            summary['performance'] = {
                'optimized_mean_ms': round(opt_mean, 3),
                'original_mean_ms': round(orig_mean, 3),
                'improvement_percent': round(((orig_mean - opt_mean) / orig_mean) * 100, 2),
                'speedup_factor': round(orig_mean / opt_mean, 2) if opt_mean > 0 else 0,
                'optimized_std_ms': round(statistics.stdev(self.optimized_times), 3) if len(self.optimized_times) > 1 else 0,
                'original_std_ms': round(statistics.stdev(self.original_times), 3) if len(self.original_times) > 1 else 0
            }
        elif self.optimized_times:
            summary['performance'] = {
                'optimized_mean_ms': round(statistics.mean(self.optimized_times), 3),
                'optimized_std_ms': round(statistics.stdev(self.optimized_times), 3) if len(self.optimized_times) > 1 else 0
            }
        elif self.original_times:
            summary['performance'] = {
                'original_mean_ms': round(statistics.mean(self.original_times), 3),
                'original_std_ms': round(statistics.stdev(self.original_times), 3) if len(self.original_times) > 1 else 0
            }
        
        # Memory statistics
        if self.optimized_memory and self.original_memory:
            opt_mem_mean = statistics.mean(self.optimized_memory)
            orig_mem_mean = statistics.mean(self.original_memory)
            
            summary['memory'] = {
                'optimized_mean_mb': round(opt_mem_mean, 2),
                'original_mean_mb': round(orig_mem_mean, 2),
                'memory_change_percent': round(((opt_mem_mean - orig_mem_mean) / orig_mem_mean) * 100, 2) if orig_mem_mean > 0 else 0
            }
        
        return summary


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system
    
    Tracks execution times, memory usage, error rates, and provides
    detailed analysis and reporting capabilities.
    """
    
    def __init__(
        self,
        max_metrics: int = 10000,
        analysis_window_minutes: int = 60,
        enable_memory_tracking: bool = True
    ):
        self.max_metrics = max_metrics
        self.analysis_window_minutes = analysis_window_minutes
        self.enable_memory_tracking = enable_memory_tracking
        
        # Storage
        self._metrics: deque = deque(maxlen=max_metrics)
        self._function_stats: Dict[str, FunctionStats] = defaultdict(lambda: FunctionStats(""))
        self._lock = threading.RLock()
        
        # System monitoring
        self._process = psutil.Process()
        self._start_time = datetime.now()
        
        # Performance baselines (loaded from previous runs)
        self._baselines: Dict[str, Dict[str, float]] = {}
    
    def record_execution(
        self,
        function_name: str,
        execution_time_ms: float,
        optimization_used: bool,
        args_count: int = 0,
        result_size_bytes: int = 0,
        error: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """Record a function execution"""
        
        # Get memory usage
        memory_mb = 0.0
        if self.enable_memory_tracking:
            try:
                memory_mb = self._process.memory_info().rss / (1024 * 1024)
            except Exception:
                pass  # Memory tracking is best-effort
        
        # Create metric
        metric = PerformanceMetric(
            function_name=function_name,
            execution_time_ms=execution_time_ms,
            memory_usage_mb=memory_mb,
            timestamp=datetime.now(),
            optimization_used=optimization_used,
            args_count=args_count,
            result_size_bytes=result_size_bytes,
            error=error,
            context=context or {}
        )
        
        with self._lock:
            self._metrics.append(metric)
            
            # Update function statistics
            if function_name not in self._function_stats:
                self._function_stats[function_name] = FunctionStats(function_name)
            self._function_stats[function_name].add_metric(metric)
    
    def get_function_stats(self, function_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for specific function or all functions"""
        with self._lock:
            if function_name:
                if function_name in self._function_stats:
                    return self._function_stats[function_name].get_summary()
                else:
                    return {}
            else:
                return {
                    name: stats.get_summary() 
                    for name, stats in self._function_stats.items()
                }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide performance statistics"""
        with self._lock:
            # Calculate aggregate metrics
            total_calls = sum(stats.total_calls for stats in self._function_stats.values())
            total_optimized = sum(stats.optimized_calls for stats in self._function_stats.values())
            total_errors = sum(stats.optimized_errors + stats.original_errors for stats in self._function_stats.values())
            
            # Recent performance (last hour)
            cutoff_time = datetime.now() - timedelta(minutes=self.analysis_window_minutes)
            recent_metrics = [m for m in self._metrics if m.timestamp >= cutoff_time]
            
            system_stats = {
                'monitoring_duration_minutes': round((datetime.now() - self._start_time).total_seconds() / 60, 1),
                'total_function_calls': total_calls,
                'optimization_adoption_rate': round(total_optimized / total_calls * 100, 2) if total_calls > 0 else 0,
                'overall_error_rate': round(total_errors / total_calls * 100, 4) if total_calls > 0 else 0,
                'functions_monitored': len(self._function_stats),
                'recent_activity': {
                    'window_minutes': self.analysis_window_minutes,
                    'calls_in_window': len(recent_metrics),
                    'avg_calls_per_minute': round(len(recent_metrics) / self.analysis_window_minutes, 1) if recent_metrics else 0
                }
            }
            
            # System resource usage
            try:
                cpu_percent = self._process.cpu_percent()
                memory_info = self._process.memory_info()
                
                system_stats['system_resources'] = {
                    'cpu_percent': round(cpu_percent, 1),
                    'memory_rss_mb': round(memory_info.rss / (1024 * 1024), 1),
                    'memory_vms_mb': round(memory_info.vms / (1024 * 1024), 1)
                }
            except Exception:
                system_stats['system_resources'] = {'error': 'Unable to collect system metrics'}
            
            return system_stats
    
    def detect_performance_regressions(self, threshold_percent: float = 10.0) -> List[Dict[str, Any]]:
        """
        Detect functions that have performance regressions
        
        Args:
            threshold_percent: Minimum performance degradation to flag as regression
            
        Returns:
            List of regression alerts
        """
        regressions = []
        
        with self._lock:
            for func_name, stats in self._function_stats.items():
                if not stats.optimized_times or not stats.original_times:
                    continue
                
                # Calculate recent vs baseline performance
                recent_optimized = stats.optimized_times[-50:] if len(stats.optimized_times) >= 50 else stats.optimized_times
                recent_original = stats.original_times[-50:] if len(stats.original_times) >= 50 else stats.original_times
                
                if len(recent_optimized) < 5 or len(recent_original) < 5:
                    continue  # Need sufficient samples
                
                recent_opt_mean = statistics.mean(recent_optimized)
                recent_orig_mean = statistics.mean(recent_original)
                
                # Check if optimization is actually slower than original
                if recent_opt_mean > recent_orig_mean:
                    degradation = ((recent_opt_mean - recent_orig_mean) / recent_orig_mean) * 100
                    
                    if degradation >= threshold_percent:
                        regressions.append({
                            'function': func_name,
                            'degradation_percent': round(degradation, 2),
                            'optimized_mean_ms': round(recent_opt_mean, 3),
                            'original_mean_ms': round(recent_orig_mean, 3),
                            'recommendation': 'Consider disabling optimization or investigating implementation'
                        })
                
                # Check for significant slowdowns in optimized version
                baseline_key = f"{func_name}_optimized"
                if baseline_key in self._baselines:
                    baseline_time = self._baselines[baseline_key].get('mean_time_ms', recent_opt_mean)
                    
                    if recent_opt_mean > baseline_time * (1 + threshold_percent / 100):
                        slowdown = ((recent_opt_mean - baseline_time) / baseline_time) * 100
                        regressions.append({
                            'function': func_name,
                            'type': 'baseline_regression',
                            'slowdown_percent': round(slowdown, 2),
                            'current_mean_ms': round(recent_opt_mean, 3),
                            'baseline_mean_ms': round(baseline_time, 3),
                            'recommendation': 'Performance has degraded from baseline - investigate recent changes'
                        })
        
        return sorted(regressions, key=lambda x: x.get('degradation_percent', x.get('slowdown_percent', 0)), reverse=True)
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'system_overview': self.get_system_stats(),
            'function_performance': self.get_function_stats(),
            'performance_regressions': self.detect_performance_regressions(),
            'top_performers': self._get_top_performers(),
            'optimization_recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _get_top_performers(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get functions with best performance improvements"""
        performers = []
        
        with self._lock:
            for func_name, stats in self._function_stats.items():
                if not stats.optimized_times or not stats.original_times:
                    continue
                
                opt_mean = statistics.mean(stats.optimized_times)
                orig_mean = statistics.mean(stats.original_times)
                
                if opt_mean < orig_mean:  # Optimization is faster
                    improvement = ((orig_mean - opt_mean) / orig_mean) * 100
                    speedup = orig_mean / opt_mean
                    
                    performers.append({
                        'function': func_name,
                        'improvement_percent': round(improvement, 2),
                        'speedup_factor': round(speedup, 2),
                        'optimized_mean_ms': round(opt_mean, 3),
                        'original_mean_ms': round(orig_mean, 3),
                        'calls': stats.optimized_calls
                    })
        
        return sorted(performers, key=lambda x: x['improvement_percent'], reverse=True)[:limit]
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on collected data"""
        recommendations = []
        
        with self._lock:
            system_stats = self.get_system_stats()
            
            # Overall adoption recommendations
            if system_stats['optimization_adoption_rate'] < 50:
                recommendations.append(
                    "Low optimization adoption rate. Consider enabling more optimizations or "
                    "investigating why optimizations are being bypassed."
                )
            
            # Function-specific recommendations
            for func_name, stats in self._function_stats.items():
                if stats.total_calls < 10:
                    continue  # Skip functions with insufficient data
                
                error_rate_opt = (stats.optimized_errors / stats.optimized_calls * 100) if stats.optimized_calls > 0 else 0
                error_rate_orig = (stats.original_errors / stats.original_calls * 100) if stats.original_calls > 0 else 0
                
                if error_rate_opt > error_rate_orig * 2 and error_rate_opt > 5:
                    recommendations.append(
                        f"Function '{func_name}' has high optimization error rate ({error_rate_opt:.1f}%). "
                        "Consider investigating implementation or reducing rollout percentage."
                    )
                
                if stats.optimized_times and stats.original_times:
                    opt_mean = statistics.mean(stats.optimized_times)
                    orig_mean = statistics.mean(stats.original_times)
                    
                    if opt_mean > orig_mean * 1.1:  # Optimization is 10% slower
                        recommendations.append(
                            f"Function '{func_name}' optimization appears to be slower than original. "
                            "Consider disabling or re-implementing."
                        )
                    elif opt_mean < orig_mean * 0.5:  # Very fast optimization
                        recommendations.append(
                            f"Function '{func_name}' shows excellent performance improvement. "
                            "Consider increasing rollout percentage to 100%."
                        )
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export collected metrics to file"""
        
        with self._lock:
            if format.lower() == 'json':
                report = self.generate_performance_report()
                with open(filepath, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
            
            elif format.lower() == 'csv':
                import csv
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'timestamp', 'function_name', 'execution_time_ms', 
                        'optimization_used', 'memory_usage_mb', 'error'
                    ])
                    
                    for metric in self._metrics:
                        writer.writerow([
                            metric.timestamp.isoformat(),
                            metric.function_name,
                            metric.execution_time_ms,
                            metric.optimization_used,
                            metric.memory_usage_mb,
                            metric.error or ''
                        ])
            
        logger.info(f"Exported performance metrics to {filepath}")
    
    def reset_metrics(self):
        """Reset all collected metrics (useful for testing)"""
        with self._lock:
            self._metrics.clear()
            self._function_stats.clear()
            self._start_time = datetime.now()


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def monitor_performance(
    function_name: Optional[str] = None,
    enable_monitoring: bool = True
):
    """
    Decorator to monitor function performance
    
    Args:
        function_name: Override name for monitoring (defaults to function.__name__)
        enable_monitoring: Enable/disable monitoring for this function
    """
    def decorator(func: Callable) -> Callable:
        if not enable_monitoring:
            return func
        
        monitored_name = function_name or func.__name__
        
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            error = None
            result = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = str(e)
                raise
            finally:
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                
                # Estimate result size
                result_size = 0
                try:
                    if result is not None:
                        import sys
                        result_size = sys.getsizeof(result)
                except Exception:
                    pass
                
                performance_monitor.record_execution(
                    function_name=monitored_name,
                    execution_time_ms=execution_time_ms,
                    optimization_used=False,  # Default to False, can be overridden
                    args_count=len(args) + len(kwargs),
                    result_size_bytes=result_size,
                    error=error
                )
        
        return wrapper
    return decorator


def get_performance_summary() -> Dict[str, Any]:
    """Get quick performance summary for status checking"""
    return {
        'system_stats': performance_monitor.get_system_stats(),
        'top_performers': performance_monitor._get_top_performers(3),
        'regressions': performance_monitor.detect_performance_regressions(5.0)  # 5% threshold
    }