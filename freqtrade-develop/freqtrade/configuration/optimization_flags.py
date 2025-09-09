"""
Optimization Feature Flags System

This module provides a centralized way to enable/disable performance optimizations
with gradual rollout capabilities and safety switches.

Design principles:
1. Safe defaults - optimizations are opt-in
2. Granular control - individual function control
3. Runtime switching - no restart required  
4. Telemetry support - track adoption and performance
5. Fallback support - automatic rollback on errors
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Optimization levels for gradual rollout"""
    DISABLED = "disabled"       # Use original functions only
    CONSERVATIVE = "conservative"  # Low-risk optimizations only  
    STANDARD = "standard"       # Most optimizations enabled
    AGGRESSIVE = "aggressive"   # All optimizations enabled
    EXPERIMENTAL = "experimental"  # Include experimental features


@dataclass
class OptimizationConfig:
    """Configuration for individual optimizations"""
    enabled: bool = False
    level: OptimizationLevel = OptimizationLevel.DISABLED
    rollout_percentage: float = 0.0  # 0-100% of executions use optimization
    max_errors: int = 5  # Max errors before auto-disable
    error_window_minutes: int = 60  # Time window for error counting
    performance_threshold: float = 1.0  # Min performance ratio (optimized/original)
    last_error: Optional[datetime] = None
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class OptimizationFlags:
    """
    Centralized optimization feature flags manager
    
    Provides runtime control over performance optimizations with:
    - Gradual percentage-based rollout  
    - Automatic error-based fallback
    - Performance monitoring integration
    - Environment variable overrides
    """
    
    def __init__(self):
        self._flags: Dict[str, OptimizationConfig] = {}
        self._global_level = OptimizationLevel.DISABLED
        self._load_default_config()
        self._load_env_overrides()
        
    def _load_default_config(self):
        """Load default optimization configurations"""
        
        # Sprint 2 functions - proven stable, higher rollout by default
        self._flags['calculate_market_change'] = OptimizationConfig(
            enabled=True,
            level=OptimizationLevel.STANDARD,
            rollout_percentage=100.0,  # Proven stable in Sprint 2
            max_errors=10,
            performance_threshold=0.5,  # Must be at least 50% faster
            metadata={'sprint': 2, 'stability': 'high', 'impact': 'medium'}
        )
        
        self._flags['analyze_trade_parallelism'] = OptimizationConfig(
            enabled=True,
            level=OptimizationLevel.STANDARD, 
            rollout_percentage=100.0,  # Major breakthrough in Sprint 2
            max_errors=5,
            performance_threshold=0.1,  # Should be dramatically faster
            metadata={'sprint': 2, 'stability': 'high', 'impact': 'very_high'}
        )
        
        self._flags['combine_dataframes_by_column'] = OptimizationConfig(
            enabled=True,
            level=OptimizationLevel.STANDARD,
            rollout_percentage=100.0,  # Stable optimization
            max_errors=10,
            performance_threshold=0.6,
            metadata={'sprint': 2, 'stability': 'high', 'impact': 'medium'}
        )
        
        self._flags['calculate_max_drawdown'] = OptimizationConfig(
            enabled=True,
            level=OptimizationLevel.STANDARD,
            rollout_percentage=90.0,  # Gradual rollout
            max_errors=8,
            performance_threshold=0.7,
            metadata={'sprint': 2, 'stability': 'medium', 'impact': 'medium'}
        )
        
        # Risk metrics - newer, more conservative rollout
        for func in ['calculate_sharpe', 'calculate_sortino', 'calculate_calmar']:
            self._flags[func] = OptimizationConfig(
                enabled=True,
                level=OptimizationLevel.CONSERVATIVE,
                rollout_percentage=50.0,  # Conservative rollout
                max_errors=3,  # Lower tolerance for newer optimizations
                performance_threshold=0.8,
                metadata={'sprint': 2, 'stability': 'medium', 'impact': 'low'}
            )
        
        # Sprint 3 functions - new and experimental
        self._flags['get_ohlcv_as_lists'] = OptimizationConfig(
            enabled=False,  # Start disabled
            level=OptimizationLevel.EXPERIMENTAL,
            rollout_percentage=10.0,  # Very conservative rollout
            max_errors=2,  # Low error tolerance
            performance_threshold=0.6,  # Should be significantly faster
            metadata={'sprint': 3, 'stability': 'experimental', 'impact': 'high'}
        )
        
        self._flags['get_ohlcv_as_lists_vectorized'] = OptimizationConfig(
            enabled=False,  # Start disabled  
            level=OptimizationLevel.EXPERIMENTAL,
            rollout_percentage=5.0,  # Very limited rollout
            max_errors=1,  # Very low tolerance
            performance_threshold=0.7,
            metadata={'sprint': 3, 'stability': 'experimental', 'impact': 'high'}
        )
        
    def _load_env_overrides(self):
        """Load configuration overrides from environment variables"""
        
        # Global level override
        global_level = os.getenv('FREQTRADE_OPT_LEVEL', '').lower()
        if global_level in [level.value for level in OptimizationLevel]:
            self._global_level = OptimizationLevel(global_level)
            logger.info(f"Optimization level set to {self._global_level.value} via environment")
        
        # Individual function overrides
        for func_name in self._flags:
            env_key = f'FREQTRADE_OPT_{func_name.upper()}'
            env_value = os.getenv(env_key, '').lower()
            
            if env_value == 'true':
                self._flags[func_name].enabled = True
                self._flags[func_name].rollout_percentage = 100.0
                logger.info(f"Optimization {func_name} force-enabled via environment")
            elif env_value == 'false':
                self._flags[func_name].enabled = False
                self._flags[func_name].rollout_percentage = 0.0
                logger.info(f"Optimization {func_name} force-disabled via environment")
                
            # Rollout percentage override
            percentage_key = f'FREQTRADE_OPT_{func_name.upper()}_PERCENTAGE'
            percentage_str = os.getenv(percentage_key)
            if percentage_str:
                try:
                    percentage = float(percentage_str)
                    if 0.0 <= percentage <= 100.0:
                        self._flags[func_name].rollout_percentage = percentage
                        logger.info(f"Optimization {func_name} rollout set to {percentage}% via environment")
                except ValueError:
                    logger.warning(f"Invalid percentage value for {percentage_key}: {percentage_str}")
    
    def is_optimization_enabled(
        self, 
        function_name: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if an optimization should be used
        
        Args:
            function_name: Name of the function to check
            context: Optional context for decision making (user_id, request_id, etc.)
            
        Returns:
            True if optimization should be used, False otherwise
        """
        config = self._flags.get(function_name)
        if not config:
            return False
            
        # Global disable check
        if self._global_level == OptimizationLevel.DISABLED:
            return False
            
        # Function disabled check
        if not config.enabled:
            return False
            
        # Level compatibility check
        if not self._is_level_compatible(config.level):
            return False
            
        # Error threshold check
        if self._is_error_threshold_exceeded(config):
            logger.warning(f"Optimization {function_name} disabled due to error threshold")
            return False
            
        # Rollout percentage check
        if config.rollout_percentage < 100.0:
            import random
            if random.random() * 100 > config.rollout_percentage:
                return False
                
        return True
    
    def _is_level_compatible(self, optimization_level: OptimizationLevel) -> bool:
        """Check if optimization level is compatible with global level"""
        level_hierarchy = [
            OptimizationLevel.DISABLED,
            OptimizationLevel.CONSERVATIVE, 
            OptimizationLevel.STANDARD,
            OptimizationLevel.AGGRESSIVE,
            OptimizationLevel.EXPERIMENTAL
        ]
        
        try:
            global_index = level_hierarchy.index(self._global_level)
            opt_index = level_hierarchy.index(optimization_level)
            return opt_index <= global_index
        except ValueError:
            return False
    
    def _is_error_threshold_exceeded(self, config: OptimizationConfig) -> bool:
        """Check if error threshold has been exceeded in the time window"""
        if config.last_error is None:
            return False
            
        # Reset error count if outside time window
        window_start = datetime.now() - timedelta(minutes=config.error_window_minutes)
        if config.last_error < window_start:
            config.error_count = 0
            config.last_error = None
            return False
            
        return config.error_count >= config.max_errors
    
    def record_optimization_error(self, function_name: str, error: Exception):
        """Record an error for automatic fallback logic"""
        config = self._flags.get(function_name)
        if not config:
            return
            
        config.error_count += 1
        config.last_error = datetime.now()
        
        logger.error(f"Optimization error in {function_name} (count: {config.error_count}): {error}")
        
        # Auto-disable if threshold exceeded
        if config.error_count >= config.max_errors:
            config.enabled = False
            logger.critical(f"Auto-disabling optimization {function_name} due to error threshold")
    
    def record_performance_metric(
        self, 
        function_name: str, 
        original_time: float, 
        optimized_time: float
    ):
        """Record performance metrics for monitoring"""
        config = self._flags.get(function_name)
        if not config:
            return
            
        ratio = optimized_time / original_time if original_time > 0 else 1.0
        improvement_pct = ((original_time - optimized_time) / original_time) * 100
        
        # Check if performance threshold is met
        if ratio > config.performance_threshold:
            logger.warning(
                f"Optimization {function_name} below performance threshold: "
                f"{improvement_pct:.1f}% improvement (threshold: {(1-config.performance_threshold)*100:.1f}%)"
            )
        
        # Store in metadata for telemetry
        if 'performance_history' not in config.metadata:
            config.metadata['performance_history'] = []
            
        config.metadata['performance_history'].append({
            'timestamp': datetime.now().isoformat(),
            'original_time': original_time,
            'optimized_time': optimized_time, 
            'improvement_pct': improvement_pct
        })
        
        # Keep only last 100 measurements
        if len(config.metadata['performance_history']) > 100:
            config.metadata['performance_history'] = config.metadata['performance_history'][-100:]
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current status of all optimizations for debugging/monitoring"""
        status = {
            'global_level': self._global_level.value,
            'functions': {}
        }
        
        for func_name, config in self._flags.items():
            status['functions'][func_name] = {
                'enabled': config.enabled,
                'level': config.level.value,
                'rollout_percentage': config.rollout_percentage,
                'error_count': config.error_count,
                'last_error': config.last_error.isoformat() if config.last_error else None,
                'metadata': config.metadata
            }
            
        return status
    
    def set_global_level(self, level: OptimizationLevel):
        """Set global optimization level"""
        self._global_level = level
        logger.info(f"Global optimization level set to {level.value}")
    
    def enable_function(self, function_name: str, rollout_percentage: float = 100.0):
        """Enable optimization for a specific function"""
        if function_name not in self._flags:
            # Create new entry for functions not in default config
            self._flags[function_name] = OptimizationConfig(
                enabled=True,
                level=OptimizationLevel.STANDARD,
                rollout_percentage=rollout_percentage,
                max_errors=5,  # Default error threshold
                performance_threshold=0.5,  # Default performance threshold
                metadata={'dynamic': True}
            )
            logger.info(f"Created and enabled optimization {function_name} at {rollout_percentage}% rollout")
        else:
            # Update existing entry
            self._flags[function_name].enabled = True
            self._flags[function_name].rollout_percentage = rollout_percentage
            logger.info(f"Enabled optimization {function_name} at {rollout_percentage}% rollout")
    
    def register_function(
        self, 
        function_name: str, 
        enabled: bool = False,
        level: OptimizationLevel = OptimizationLevel.STANDARD,
        rollout_percentage: float = 0.0,
        max_errors: int = 5,
        performance_threshold: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a new function for optimization control"""
        self._flags[function_name] = OptimizationConfig(
            enabled=enabled,
            level=level,
            rollout_percentage=rollout_percentage,
            max_errors=max_errors,
            performance_threshold=performance_threshold,
            metadata=metadata or {'dynamic': True}
        )
        logger.info(f"Registered function {function_name} for optimization control")
    
    def disable_function(self, function_name: str):
        """Disable optimization for a specific function"""
        if function_name in self._flags:
            self._flags[function_name].enabled = False
            logger.info(f"Disabled optimization {function_name}")


# Global optimization flags instance
optimization_flags = OptimizationFlags()


def should_use_optimization(function_name: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """
    Convenience function to check if optimization should be used
    
    Args:
        function_name: Name of function to check  
        context: Optional context information
        
    Returns:
        True if optimization should be used
    """
    return optimization_flags.is_optimization_enabled(function_name, context)


def with_optimization_fallback(
    function_name: str,
    optimized_func: Callable,
    original_func: Callable,
    *args,
    **kwargs
) -> Any:
    """
    Execute optimized function with automatic fallback to original on error
    
    Args:
        function_name: Name of function for telemetry
        optimized_func: Optimized function to try first
        original_func: Original function to fall back to
        *args, **kwargs: Arguments to pass to functions
        
    Returns:
        Result from optimized function, or original function if optimization fails
    """
    if not should_use_optimization(function_name):
        return original_func(*args, **kwargs)
    
    import time
    
    try:
        # Time optimized execution
        start_time = time.perf_counter()
        result = optimized_func(*args, **kwargs)
        opt_time = time.perf_counter() - start_time
        
        # Time original for comparison (lightweight sampling)
        if optimization_flags._flags.get(function_name, {}).get('rollout_percentage', 0) < 100:
            start_time = time.perf_counter()
            original_func(*args, **kwargs)  # Don't use result, just measure
            orig_time = time.perf_counter() - start_time
            
            optimization_flags.record_performance_metric(function_name, orig_time, opt_time)
        
        return result
        
    except Exception as e:
        optimization_flags.record_optimization_error(function_name, e)
        logger.warning(f"Optimization {function_name} failed, falling back to original: {e}")
        return original_func(*args, **kwargs)