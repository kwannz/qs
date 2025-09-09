"""
High-Performance Caching Layer for Freqtrade

This module provides intelligent caching for frequently-called functions
with focus on backtesting and analysis performance optimization.

Features:
- LRU cache with automatic eviction
- Memory-aware cache sizing  
- Time-based cache expiration
- Cache warming for predictable workloads
- Performance metrics and monitoring
- Thread-safe operations
"""

import logging
import hashlib
import pickle
import threading
import time
import weakref
from collections import OrderedDict
from typing import Any, Dict, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from functools import wraps
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0  
    total_requests: int = 0
    total_memory_mb: float = 0.0
    avg_lookup_time_ms: float = 0.0
    hit_rate: float = 0.0
    last_reset: float = field(default_factory=time.time)
    
    def update_hit_rate(self):
        """Update calculated hit rate"""
        if self.total_requests > 0:
            self.hit_rate = (self.hits / self.total_requests) * 100
    
    def reset(self):
        """Reset all statistics"""
        self.hits = 0
        self.misses = 0  
        self.evictions = 0
        self.total_requests = 0
        self.total_memory_mb = 0.0
        self.avg_lookup_time_ms = 0.0
        self.hit_rate = 0.0
        self.last_reset = time.time()


class SmartLRUCache:
    """
    Memory-aware LRU cache with intelligent eviction policies
    
    Features:
    - Automatic memory monitoring and management
    - Time-based expiration for data freshness
    - Efficient key hashing for complex objects
    - Thread-safe operations
    - Detailed performance metrics
    """
    
    def __init__(
        self, 
        max_size: int = 1000,
        max_memory_mb: float = 500.0,
        ttl_seconds: int = 3600,  # 1 hour default
        enable_metrics: bool = True
    ):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.ttl_seconds = ttl_seconds
        self.enable_metrics = enable_metrics
        
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, float] = {}
        self._memory_usage: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._stats = CacheStats()
        
    def _hash_key(self, args: Tuple, kwargs: Dict[str, Any]) -> str:
        """
        Create hash key from function arguments
        Handles complex objects like DataFrames intelligently
        """
        key_data = []
        
        # Process positional arguments
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                # Use shape, columns, and sample of data for DataFrame hash
                key_data.append(('df', arg.shape, tuple(arg.columns), arg.iloc[:5].values.tobytes() if len(arg) > 0 else b''))
            elif isinstance(arg, np.ndarray):
                # Use shape and sample for numpy arrays
                key_data.append(('array', arg.shape, arg.dtype, arg.flat[:20].tobytes() if arg.size > 0 else b''))
            elif isinstance(arg, dict):
                # Sort dict items for consistent hashing
                key_data.append(('dict', tuple(sorted(arg.items()))))
            else:
                key_data.append(arg)
        
        # Process keyword arguments  
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (pd.DataFrame, np.ndarray)):
                key_data.append((k, type(v).__name__, str(v.shape) if hasattr(v, 'shape') else str(v)))
            else:
                key_data.append((k, v))
        
        # Create hash
        key_str = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(key_str).hexdigest()[:32]  # Use first 32 chars for efficiency
    
    def _estimate_memory(self, obj: Any) -> float:
        """Estimate memory usage in MB for an object"""
        try:
            if isinstance(obj, pd.DataFrame):
                return obj.memory_usage(deep=True).sum() / (1024 * 1024)
            elif isinstance(obj, np.ndarray):
                return obj.nbytes / (1024 * 1024)
            elif isinstance(obj, (list, tuple)):
                # Estimate for collections
                if len(obj) > 0:
                    sample_size = self._estimate_memory(obj[0]) if len(obj) > 0 else 0.001
                    return sample_size * len(obj)
            else:
                # Fallback estimation
                return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)) / (1024 * 1024)
        except Exception:
            # Conservative fallback
            return 0.1
    
    def _cleanup_expired(self):
        """Remove expired entries from cache"""
        if self.ttl_seconds <= 0:
            return
            
        current_time = time.time()
        expired_keys = []
        
        for key, timestamp in self._timestamps.items():
            if current_time - timestamp > self.ttl_seconds:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self._cache:
                self._evict_key(key)
    
    def _evict_key(self, key: str):
        """Remove specific key from cache"""
        if key in self._cache:
            del self._cache[key]
        if key in self._timestamps:
            del self._timestamps[key]
        if key in self._memory_usage:
            self._stats.total_memory_mb -= self._memory_usage[key]
            del self._memory_usage[key]
        
        if self.enable_metrics:
            self._stats.evictions += 1
    
    def _enforce_limits(self):
        """Enforce cache size and memory limits"""
        # Enforce memory limit
        while (self._stats.total_memory_mb > self.max_memory_mb and 
               len(self._cache) > 0):
            # Evict oldest entry
            oldest_key = next(iter(self._cache))
            self._evict_key(oldest_key)
        
        # Enforce size limit  
        while len(self._cache) > self.max_size:
            oldest_key = next(iter(self._cache))
            self._evict_key(oldest_key)
    
    def get(self, key: str) -> Tuple[bool, Any]:
        """
        Get value from cache
        Returns (found, value) tuple
        """
        with self._lock:
            start_time = time.time()
            
            # Cleanup expired entries periodically
            if len(self._cache) > 100 and time.time() % 60 < 1:  # Every ~60 seconds
                self._cleanup_expired()
            
            if self.enable_metrics:
                self._stats.total_requests += 1
            
            # Check if key exists and not expired
            if key in self._cache:
                current_time = time.time()
                if (self.ttl_seconds <= 0 or 
                    current_time - self._timestamps.get(key, 0) <= self.ttl_seconds):
                    
                    # Move to end (most recently used)
                    value = self._cache.pop(key)
                    self._cache[key] = value
                    
                    if self.enable_metrics:
                        self._stats.hits += 1
                        lookup_time = (time.time() - start_time) * 1000
                        # Running average of lookup times
                        if self._stats.avg_lookup_time_ms == 0:
                            self._stats.avg_lookup_time_ms = lookup_time
                        else:
                            self._stats.avg_lookup_time_ms = (
                                self._stats.avg_lookup_time_ms * 0.9 + lookup_time * 0.1
                            )
                        self._stats.update_hit_rate()
                    
                    return True, value
                else:
                    # Expired - remove it
                    self._evict_key(key)
            
            if self.enable_metrics:
                self._stats.misses += 1
                self._stats.update_hit_rate()
            
            return False, None
    
    def put(self, key: str, value: Any):
        """Store value in cache"""
        with self._lock:
            # Estimate memory usage
            memory_mb = self._estimate_memory(value)
            
            # Remove existing entry if it exists
            if key in self._cache:
                self._evict_key(key)
            
            # Store new entry
            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._memory_usage[key] = memory_mb
            self._stats.total_memory_mb += memory_mb
            
            # Enforce limits
            self._enforce_limits()
    
    def clear(self):
        """Clear entire cache"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear() 
            self._memory_usage.clear()
            self._stats.total_memory_mb = 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_mb': round(self._stats.total_memory_mb, 2),
                'max_memory_mb': self.max_memory_mb,
                'hit_rate': round(self._stats.hit_rate, 2),
                'hits': self._stats.hits,
                'misses': self._stats.misses,
                'evictions': self._stats.evictions,
                'total_requests': self._stats.total_requests,
                'avg_lookup_time_ms': round(self._stats.avg_lookup_time_ms, 3),
                'uptime_minutes': round((time.time() - self._stats.last_reset) / 60, 1)
            }


class CacheManager:
    """
    Global cache manager for Freqtrade optimizations
    
    Manages multiple specialized caches for different use cases:
    - OHLCV data cache (large, long TTL)
    - Calculation results cache (medium, medium TTL)  
    - Temporary computation cache (small, short TTL)
    """
    
    def __init__(self):
        self._caches: Dict[str, SmartLRUCache] = {}
        self._create_default_caches()
    
    def _create_default_caches(self):
        """Create default caches for common use cases"""
        
        # OHLCV data cache - large datasets, long retention
        self._caches['ohlcv'] = SmartLRUCache(
            max_size=100,  # Fewer entries but larger
            max_memory_mb=1000.0,  # 1GB for OHLCV data
            ttl_seconds=7200,  # 2 hours
            enable_metrics=True
        )
        
        # Analysis results cache - medium size, medium retention
        self._caches['analysis'] = SmartLRUCache(
            max_size=500,
            max_memory_mb=200.0,  # 200MB for analysis results
            ttl_seconds=3600,  # 1 hour  
            enable_metrics=True
        )
        
        # Temporary computation cache - small, fast turnover
        self._caches['temp'] = SmartLRUCache(
            max_size=1000,
            max_memory_mb=50.0,  # 50MB for temporary results
            ttl_seconds=300,  # 5 minutes
            enable_metrics=True
        )
        
        # Backtesting cache - specialized for backtesting operations
        self._caches['backtest'] = SmartLRUCache(
            max_size=200,
            max_memory_mb=500.0,  # 500MB for backtesting data
            ttl_seconds=1800,  # 30 minutes
            enable_metrics=True
        )
    
    def get_cache(self, cache_name: str) -> SmartLRUCache:
        """Get specific cache by name"""
        return self._caches.get(cache_name, self._caches['temp'])
    
    def create_cache(
        self, 
        name: str, 
        max_size: int = 100, 
        max_memory_mb: float = 50.0,
        ttl_seconds: int = 3600
    ) -> SmartLRUCache:
        """Create a new named cache"""
        cache = SmartLRUCache(
            max_size=max_size,
            max_memory_mb=max_memory_mb, 
            ttl_seconds=ttl_seconds,
            enable_metrics=True
        )
        self._caches[name] = cache
        return cache
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all caches"""
        return {name: cache.get_stats() for name, cache in self._caches.items()}
    
    def clear_all(self):
        """Clear all caches"""
        for cache in self._caches.values():
            cache.clear()
    
    def warm_cache(self, cache_name: str, key_value_pairs: Dict[str, Any]):
        """Warm up cache with predefined key-value pairs"""
        cache = self.get_cache(cache_name)
        for key, value in key_value_pairs.items():
            cache.put(key, value)
        
        logger.info(f"Warmed cache '{cache_name}' with {len(key_value_pairs)} entries")


# Global cache manager instance
cache_manager = CacheManager()


def cached(
    cache_name: str = 'temp',
    ttl_seconds: Optional[int] = None,
    max_memory_mb: Optional[float] = None
):
    """
    Decorator to add caching to any function
    
    Args:
        cache_name: Name of cache to use ('ohlcv', 'analysis', 'temp', 'backtest')
        ttl_seconds: Override TTL for this function
        max_memory_mb: Override memory limit for this function
        
    Example:
        @cached('analysis', ttl_seconds=7200)
        def expensive_calculation(data):
            # ... complex computation
            return result
    """
    def decorator(func: Callable) -> Callable:
        cache = cache_manager.get_cache(cache_name)
        
        # Override cache settings if specified
        if ttl_seconds is not None:
            cache.ttl_seconds = ttl_seconds
        if max_memory_mb is not None:
            cache.max_memory_mb = max_memory_mb
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}:{cache._hash_key(args, kwargs)}"
            
            # Try cache first
            found, cached_result = cache.get(cache_key)
            if found:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            
            return result
        
        # Add cache control methods to function
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_stats = lambda: cache.get_stats()
        
        return wrapper
    
    return decorator


# Pre-configured cache decorators for common use cases
ohlcv_cached = lambda ttl=7200: cached('ohlcv', ttl_seconds=ttl)
analysis_cached = lambda ttl=3600: cached('analysis', ttl_seconds=ttl) 
backtest_cached = lambda ttl=1800: cached('backtest', ttl_seconds=ttl)
temp_cached = lambda ttl=300: cached('temp', ttl_seconds=ttl)


def get_cache_status() -> Dict[str, Any]:
    """Get comprehensive cache status for monitoring"""
    stats = cache_manager.get_all_stats()
    
    # Calculate aggregate statistics
    total_memory = sum(cache_stats['memory_usage_mb'] for cache_stats in stats.values())
    total_entries = sum(cache_stats['size'] for cache_stats in stats.values())
    avg_hit_rate = np.mean([cache_stats['hit_rate'] for cache_stats in stats.values()])
    
    return {
        'caches': stats,
        'aggregate': {
            'total_memory_mb': round(total_memory, 2),
            'total_entries': total_entries,
            'average_hit_rate': round(avg_hit_rate, 2),
            'cache_count': len(stats)
        },
        'recommendations': _generate_cache_recommendations(stats)
    }


def _generate_cache_recommendations(stats: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Generate optimization recommendations based on cache statistics"""
    recommendations = {}
    
    for cache_name, cache_stats in stats.items():
        if cache_stats['hit_rate'] < 30:
            recommendations[cache_name] = "Consider increasing cache size or TTL - low hit rate"
        elif cache_stats['hit_rate'] > 95:
            recommendations[cache_name] = "Excellent performance - consider expanding to cache more operations"
        elif cache_stats['memory_usage_mb'] > cache_stats['max_memory_mb'] * 0.9:
            recommendations[cache_name] = "Near memory limit - consider increasing max_memory_mb"
        elif cache_stats['evictions'] > cache_stats['hits'] * 0.1:
            recommendations[cache_name] = "High eviction rate - consider increasing cache size"
        else:
            recommendations[cache_name] = "Performing well"
    
    return recommendations


# Example usage for OHLCV caching:
@ohlcv_cached(ttl=7200)  # 2 hour TTL for OHLCV data
def cached_get_ohlcv_as_lists(processed_data, *args, **kwargs):
    """Example of how to use caching with OHLCV processing"""
    # This would wrap the actual processing function
    pass


# Example usage for analysis caching:
@analysis_cached(ttl=3600)  # 1 hour TTL for analysis
def cached_calculate_indicators(dataframe, *args, **kwargs):
    """Example of how to use caching with indicator calculations"""
    # This would wrap indicator calculations
    pass