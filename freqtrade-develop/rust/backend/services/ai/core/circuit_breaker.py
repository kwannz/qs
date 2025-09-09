"""
Circuit Breaker Implementation
断路器实现

Provides fault tolerance for AI API calls with configurable thresholds,
automatic recovery, and comprehensive monitoring capabilities.
"""

import asyncio
import time
from enum import Enum
from typing import Dict, Any, Callable, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import structlog
from functools import wraps

from config import settings


logger = structlog.get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Circuit is open, requests fail fast
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: int = 60
    monitoring_window_seconds: int = 300  # 5 minutes
    max_requests_half_open: int = 5


@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics for monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_open_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    recent_failures: List[datetime] = field(default_factory=list)
    recent_successes: List[datetime] = field(default_factory=list)
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    Features:
    - Configurable failure and success thresholds
    - Automatic state transitions (closed -> open -> half-open -> closed)
    - Time-based recovery with exponential backoff
    - Request filtering in open state
    - Comprehensive metrics collection
    - Thread-safe operation
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker identifier
            config: Configuration parameters
        """
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.next_attempt_time: Optional[float] = None
        self.half_open_requests = 0
        
        self.lock = asyncio.Lock()
        
        logger.info(
            "Circuit breaker initialized",
            name=self.name,
            failure_threshold=config.failure_threshold,
            success_threshold=config.success_threshold,
            timeout_seconds=config.timeout_seconds
        )
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerOpenException: When circuit is open
            Original exception from function call
        """
        async with self.lock:
            await self._check_state_transition()
            
            # Fail fast if circuit is open
            if self.state == CircuitState.OPEN:
                self._record_blocked_request()
                raise CircuitBreakerOpenException(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Next attempt allowed at {self._format_next_attempt_time()}"
                )
            
            # Limit requests in half-open state
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_requests >= self.config.max_requests_half_open:
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker '{self.name}' is HALF_OPEN with max requests exceeded"
                    )
                self.half_open_requests += 1
        
        # Execute the function
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            
            # Record success
            duration = time.time() - start_time
            await self._record_success(duration)
            
            return result
            
        except Exception as e:
            # Record failure
            duration = time.time() - start_time
            await self._record_failure(e, duration)
            raise
    
    async def _check_state_transition(self):
        """Check and perform state transitions based on current conditions."""
        current_time = time.time()
        
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed to move to half-open
            if (self.next_attempt_time and 
                current_time >= self.next_attempt_time):
                await self._transition_to_half_open()
                
        elif self.state == CircuitState.HALF_OPEN:
            # Check if we should close the circuit (enough successes)
            if self.success_count >= self.config.success_threshold:
                await self._transition_to_closed()
                
        elif self.state == CircuitState.CLOSED:
            # Check if we should open the circuit (too many failures)
            if self.failure_count >= self.config.failure_threshold:
                await self._transition_to_open()
    
    async def _transition_to_open(self):
        """Transition circuit breaker to OPEN state."""
        self.state = CircuitState.OPEN
        self.last_failure_time = time.time()
        self.next_attempt_time = self.last_failure_time + self.config.timeout_seconds
        self.metrics.circuit_open_count += 1
        
        logger.warning(
            "Circuit breaker opened",
            name=self.name,
            failure_count=self.failure_count,
            failure_threshold=self.config.failure_threshold,
            next_attempt=self._format_next_attempt_time()
        )
    
    async def _transition_to_half_open(self):
        """Transition circuit breaker to HALF_OPEN state."""
        self.state = CircuitState.HALF_OPEN
        self.half_open_requests = 0
        self.success_count = 0
        self.failure_count = 0
        
        logger.info("Circuit breaker transitioned to HALF_OPEN", name=self.name)
    
    async def _transition_to_closed(self):
        """Transition circuit breaker to CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_requests = 0
        self.next_attempt_time = None
        
        logger.info("Circuit breaker closed (recovered)", name=self.name)
    
    async def _record_success(self, duration: float):
        """Record successful request."""
        async with self.lock:
            self.success_count += 1
            self.metrics.successful_requests += 1
            self.metrics.total_requests += 1
            self.metrics.last_success_time = datetime.utcnow()
            
            # Clean old successes from monitoring window
            self._clean_recent_events(self.metrics.recent_successes)
            self.metrics.recent_successes.append(datetime.utcnow())
            
            # Reset failure count on success in closed state
            if self.state == CircuitState.CLOSED:
                self.failure_count = 0
            
            logger.debug(
                "Circuit breaker success recorded",
                name=self.name,
                state=self.state.value,
                success_count=self.success_count,
                duration=duration
            )
    
    async def _record_failure(self, exception: Exception, duration: float):
        """Record failed request."""
        async with self.lock:
            self.failure_count += 1
            self.metrics.failed_requests += 1
            self.metrics.total_requests += 1
            self.metrics.last_failure_time = datetime.utcnow()
            
            # Clean old failures from monitoring window
            self._clean_recent_events(self.metrics.recent_failures)
            self.metrics.recent_failures.append(datetime.utcnow())
            
            # Reset success count on failure in half-open state
            if self.state == CircuitState.HALF_OPEN:
                self.success_count = 0
                # Immediately transition to open on failure in half-open state
                await self._transition_to_open()
            
            logger.warning(
                "Circuit breaker failure recorded",
                name=self.name,
                state=self.state.value,
                failure_count=self.failure_count,
                exception=str(exception),
                duration=duration
            )
    
    def _record_blocked_request(self):
        """Record request blocked by open circuit."""
        self.metrics.total_requests += 1
        # Note: Not incrementing failed_requests as the request wasn't attempted
        
        logger.debug("Request blocked by open circuit breaker", name=self.name)
    
    def _clean_recent_events(self, events: List[datetime]):
        """Remove events outside monitoring window."""
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.config.monitoring_window_seconds)
        events[:] = [event for event in events if event > cutoff_time]
    
    def _format_next_attempt_time(self) -> str:
        """Format next attempt time for display."""
        if self.next_attempt_time:
            next_time = datetime.fromtimestamp(self.next_attempt_time)
            return next_time.strftime("%H:%M:%S")
        return "unknown"
    
    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self.state
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get circuit breaker metrics for monitoring.
        
        Returns:
            Dict containing comprehensive metrics
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "failure_rate": self.metrics.failure_rate,
            "success_rate": self.metrics.success_rate,
            "circuit_open_count": self.metrics.circuit_open_count,
            "last_failure_time": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
            "last_success_time": self.metrics.last_success_time.isoformat() if self.metrics.last_success_time else None,
            "next_attempt_time": self._format_next_attempt_time() if self.next_attempt_time else None,
            "half_open_requests": self.half_open_requests if self.state == CircuitState.HALF_OPEN else 0,
            "recent_failures_count": len(self.metrics.recent_failures),
            "recent_successes_count": len(self.metrics.recent_successes),
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "success_threshold": self.config.success_threshold,
                "timeout_seconds": self.config.timeout_seconds,
                "monitoring_window_seconds": self.config.monitoring_window_seconds
            }
        }
    
    async def reset(self):
        """Reset circuit breaker to initial state."""
        async with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.half_open_requests = 0
            self.next_attempt_time = None
            
            # Reset metrics
            self.metrics = CircuitBreakerMetrics()
            
        logger.info("Circuit breaker reset", name=self.name)
    
    async def force_open(self):
        """Force circuit breaker to OPEN state (for testing/maintenance)."""
        async with self.lock:
            await self._transition_to_open()
        
        logger.warning("Circuit breaker forced open", name=self.name)
    
    async def force_close(self):
        """Force circuit breaker to CLOSED state (for testing/maintenance)."""
        async with self.lock:
            await self._transition_to_closed()
        
        logger.info("Circuit breaker forced closed", name=self.name)


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers.
    
    Provides centralized management, configuration, and monitoring
    of circuit breakers across different services and endpoints.
    """
    
    def __init__(self):
        """Initialize circuit breaker manager."""
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.default_config = CircuitBreakerConfig(
            failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            success_threshold=settings.CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
            timeout_seconds=settings.CIRCUIT_BREAKER_TIMEOUT_SECONDS
        )
        
        logger.info("Circuit breaker manager initialized")
    
    def get_circuit_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """
        Get or create circuit breaker.
        
        Args:
            name: Circuit breaker name
            config: Optional custom configuration
            
        Returns:
            CircuitBreaker instance
        """
        if name not in self.circuit_breakers:
            circuit_config = config or self.default_config
            self.circuit_breakers[name] = CircuitBreaker(name, circuit_config)
            
            logger.info("Circuit breaker created", name=name)
        
        return self.circuit_breakers[name]
    
    async def call_with_circuit_breaker(
        self,
        name: str,
        func: Callable,
        *args,
        config: Optional[CircuitBreakerConfig] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with circuit breaker protection.
        
        Args:
            name: Circuit breaker name
            func: Function to execute
            config: Optional circuit breaker configuration
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        circuit_breaker = self.get_circuit_breaker(name, config)
        return await circuit_breaker.call(func, *args, **kwargs)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all circuit breakers.
        
        Returns:
            Dict mapping circuit breaker names to their metrics
        """
        return {
            name: cb.get_metrics()
            for name, cb in self.circuit_breakers.items()
        }
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Get summary metrics across all circuit breakers.
        
        Returns:
            Dict containing summary statistics
        """
        if not self.circuit_breakers:
            return {
                "total_circuit_breakers": 0,
                "open_circuits": 0,
                "half_open_circuits": 0,
                "closed_circuits": 0
            }
        
        state_counts = {state: 0 for state in CircuitState}
        total_requests = 0
        total_failures = 0
        
        for cb in self.circuit_breakers.values():
            metrics = cb.get_metrics()
            state_counts[cb.get_state()] += 1
            total_requests += metrics["total_requests"]
            total_failures += metrics["failed_requests"]
        
        overall_failure_rate = (total_failures / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "total_circuit_breakers": len(self.circuit_breakers),
            "open_circuits": state_counts[CircuitState.OPEN],
            "half_open_circuits": state_counts[CircuitState.HALF_OPEN],
            "closed_circuits": state_counts[CircuitState.CLOSED],
            "total_requests": total_requests,
            "total_failures": total_failures,
            "overall_failure_rate": overall_failure_rate
        }
    
    async def reset_all(self):
        """Reset all circuit breakers."""
        for cb in self.circuit_breakers.values():
            await cb.reset()
        
        logger.info("All circuit breakers reset")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all circuit breakers.
        
        Returns:
            Dict containing health status
        """
        health_status = {
            "healthy": True,
            "circuit_breakers": {}
        }
        
        for name, cb in self.circuit_breakers.items():
            cb_health = {
                "state": cb.get_state().value,
                "healthy": cb.get_state() != CircuitState.OPEN
            }
            
            if not cb_health["healthy"]:
                health_status["healthy"] = False
            
            health_status["circuit_breakers"][name] = cb_health
        
        return health_status


# Decorator for automatic circuit breaker protection
def circuit_breaker(
    name: Optional[str] = None,
    failure_threshold: int = None,
    success_threshold: int = None,
    timeout_seconds: int = None
):
    """
    Decorator to add circuit breaker protection to async functions.
    
    Args:
        name: Circuit breaker name (defaults to function name)
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes needed to close circuit
        timeout_seconds: Timeout before attempting half-open
        
    Example:
        @circuit_breaker(name="deepseek_api", failure_threshold=5)
        async def call_deepseek_api():
            # API call logic
            pass
    """
    def decorator(func: Callable):
        circuit_name = name or f"{func.__module__}.{func.__name__}"
        
        # Create custom config if parameters provided
        config = None
        if any(param is not None for param in [failure_threshold, success_threshold, timeout_seconds]):
            config = CircuitBreakerConfig(
                failure_threshold=failure_threshold or settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                success_threshold=success_threshold or settings.CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
                timeout_seconds=timeout_seconds or settings.CIRCUIT_BREAKER_TIMEOUT_SECONDS
            )
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get global circuit breaker manager (would be injected in real app)
            # For this implementation, we'll create a module-level manager
            global _global_circuit_manager
            if '_global_circuit_manager' not in globals():
                _global_circuit_manager = CircuitBreakerManager()
            
            return await _global_circuit_manager.call_with_circuit_breaker(
                circuit_name, func, *args, config=config, **kwargs
            )
        
        return wrapper
    return decorator


# Module-level circuit breaker manager instance
_global_circuit_manager: Optional[CircuitBreakerManager] = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get global circuit breaker manager instance."""
    global _global_circuit_manager
    if _global_circuit_manager is None:
        _global_circuit_manager = CircuitBreakerManager()
    return _global_circuit_manager