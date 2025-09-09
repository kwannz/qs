"""
Rate Limiter Implementation
速率限制器实现

Provides intelligent rate limiting for AI API calls with token bucket algorithm,
adaptive throttling, and burst capacity management.
"""

import asyncio
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import structlog

from config import settings


logger = structlog.get_logger(__name__)


@dataclass
class RateLimiterConfig:
    """Rate limiter configuration."""
    rate: int = 60  # requests per time period
    per: int = 60   # time period in seconds
    burst: int = 10 # burst capacity
    
    @property
    def requests_per_second(self) -> float:
        """Calculate requests per second."""
        return self.rate / self.per


@dataclass
class RateLimiterMetrics:
    """Rate limiter metrics for monitoring."""
    total_requests: int = 0
    allowed_requests: int = 0
    denied_requests: int = 0
    current_tokens: float = 0
    last_refill_time: float = field(default_factory=time.time)
    wait_times: List[float] = field(default_factory=list)
    
    @property
    def denial_rate(self) -> float:
        """Calculate request denial rate."""
        if self.total_requests == 0:
            return 0.0
        return (self.denied_requests / self.total_requests) * 100
    
    @property
    def average_wait_time(self) -> float:
        """Calculate average wait time."""
        if not self.wait_times:
            return 0.0
        return sum(self.wait_times) / len(self.wait_times)


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter for API calls.
    
    Features:
    - Token bucket algorithm with configurable rate and burst
    - Smooth rate limiting with sub-second precision
    - Adaptive waiting with predictive scheduling
    - Comprehensive metrics collection
    - Thread-safe operation
    """
    
    def __init__(self, name: str, config: RateLimiterConfig):
        """
        Initialize rate limiter.
        
        Args:
            name: Rate limiter identifier
            config: Rate limiter configuration
        """
        self.name = name
        self.config = config
        self.metrics = RateLimiterMetrics()
        
        # Token bucket state
        self.tokens = float(config.burst)  # Start with full burst capacity
        self.last_update = time.time()
        
        # Synchronization
        self.lock = asyncio.Lock()
        
        logger.info(
            "Rate limiter initialized",
            name=self.name,
            rate=config.rate,
            per=config.per,
            burst=config.burst,
            requests_per_second=config.requests_per_second
        )
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens acquired, False if would need to wait
        """
        async with self.lock:
            await self._refill_bucket()
            
            self.metrics.total_requests += 1
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                self.metrics.allowed_requests += 1
                self.metrics.current_tokens = self.tokens
                
                logger.debug(
                    "Rate limit tokens acquired",
                    name=self.name,
                    tokens_requested=tokens,
                    tokens_remaining=self.tokens
                )
                
                return True
            else:
                self.metrics.denied_requests += 1
                
                logger.debug(
                    "Rate limit tokens denied",
                    name=self.name,
                    tokens_requested=tokens,
                    tokens_available=self.tokens
                )
                
                return False
    
    async def wait_for_tokens(self, tokens: int = 1) -> float:
        """
        Wait until tokens are available.
        
        Args:
            tokens: Number of tokens needed
            
        Returns:
            Time waited in seconds
        """
        start_time = time.time()
        
        while True:
            if await self.acquire(tokens):
                wait_time = time.time() - start_time
                
                if wait_time > 0:
                    self.metrics.wait_times.append(wait_time)
                    # Keep only recent wait times for metrics
                    if len(self.metrics.wait_times) > 1000:
                        self.metrics.wait_times = self.metrics.wait_times[-500:]
                
                return wait_time
            
            # Calculate optimal wait time
            wait_time = await self._calculate_wait_time(tokens)
            
            logger.debug(
                "Rate limited, waiting",
                name=self.name,
                tokens_needed=tokens,
                wait_time=wait_time
            )
            
            await asyncio.sleep(wait_time)
    
    async def _refill_bucket(self):
        """Refill token bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_update
        
        if elapsed > 0:
            # Calculate tokens to add based on rate
            tokens_to_add = elapsed * self.config.requests_per_second
            
            # Add tokens but don't exceed burst capacity
            self.tokens = min(self.config.burst, self.tokens + tokens_to_add)
            self.last_update = now
            self.metrics.current_tokens = self.tokens
            self.metrics.last_refill_time = now
    
    async def _calculate_wait_time(self, tokens_needed: int) -> float:
        """
        Calculate optimal wait time for required tokens.
        
        Args:
            tokens_needed: Number of tokens required
            
        Returns:
            Time to wait in seconds
        """
        async with self.lock:
            await self._refill_bucket()
            
            if self.tokens >= tokens_needed:
                return 0.0
            
            tokens_deficit = tokens_needed - self.tokens
            time_to_generate = tokens_deficit / self.config.requests_per_second
            
            # Add small buffer to ensure tokens are available
            return time_to_generate + 0.01
    
    def get_current_rate(self) -> float:
        """
        Get current effective rate.
        
        Returns:
            Current requests per second rate
        """
        return min(self.config.requests_per_second, self.tokens)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get rate limiter metrics.
        
        Returns:
            Dict containing comprehensive metrics
        """
        return {
            "name": self.name,
            "config": {
                "rate": self.config.rate,
                "per": self.config.per,
                "burst": self.config.burst,
                "requests_per_second": self.config.requests_per_second
            },
            "current_tokens": self.tokens,
            "total_requests": self.metrics.total_requests,
            "allowed_requests": self.metrics.allowed_requests,
            "denied_requests": self.metrics.denied_requests,
            "denial_rate": self.metrics.denial_rate,
            "average_wait_time": self.metrics.average_wait_time,
            "last_refill_time": datetime.fromtimestamp(self.metrics.last_refill_time).isoformat()
        }
    
    async def reset(self):
        """Reset rate limiter to initial state."""
        async with self.lock:
            self.tokens = float(self.config.burst)
            self.last_update = time.time()
            self.metrics = RateLimiterMetrics()
        
        logger.info("Rate limiter reset", name=self.name)


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts based on success/failure patterns.
    
    Features:
    - Automatic rate adjustment based on error rates
    - Backoff on failures, speedup on successes
    - Circuit breaker integration
    - Predictive rate control
    """
    
    def __init__(self, name: str, base_config: RateLimiterConfig):
        """
        Initialize adaptive rate limiter.
        
        Args:
            name: Rate limiter identifier
            base_config: Base rate limiter configuration
        """
        self.name = name
        self.base_config = base_config
        self.current_config = RateLimiterConfig(
            rate=base_config.rate,
            per=base_config.per,
            burst=base_config.burst
        )
        
        self.base_limiter = TokenBucketRateLimiter(f"{name}_adaptive", self.current_config)
        
        # Adaptation parameters
        self.success_count = 0
        self.failure_count = 0
        self.adaptation_window = 100  # Adjust every 100 requests
        self.min_rate_multiplier = 0.1  # Minimum 10% of base rate
        self.max_rate_multiplier = 2.0  # Maximum 200% of base rate
        
        logger.info(
            "Adaptive rate limiter initialized",
            name=self.name,
            base_rate=base_config.rate
        )
    
    async def acquire_with_feedback(self, success: Optional[bool] = None) -> bool:
        """
        Acquire token and provide feedback for adaptation.
        
        Args:
            success: Feedback on previous request (True/False/None)
            
        Returns:
            True if token acquired
        """
        # Record feedback
        if success is not None:
            if success:
                self.success_count += 1
            else:
                self.failure_count += 1
            
            # Check if adaptation is needed
            total_feedback = self.success_count + self.failure_count
            if total_feedback >= self.adaptation_window:
                await self._adapt_rate()
        
        return await self.base_limiter.acquire()
    
    async def wait_with_feedback(self, success: Optional[bool] = None) -> float:
        """
        Wait for token with feedback for adaptation.
        
        Args:
            success: Feedback on previous request
            
        Returns:
            Time waited in seconds
        """
        await self.acquire_with_feedback(success)
        return await self.base_limiter.wait_for_tokens()
    
    async def _adapt_rate(self):
        """Adapt rate based on success/failure ratio."""
        if self.failure_count + self.success_count == 0:
            return
        
        failure_rate = self.failure_count / (self.failure_count + self.success_count)
        
        # Calculate adaptation multiplier
        if failure_rate > 0.1:  # More than 10% failures
            multiplier = max(0.8, 1.0 - failure_rate)  # Reduce rate
        elif failure_rate < 0.05:  # Less than 5% failures
            multiplier = min(1.2, 1.0 + (0.05 - failure_rate) * 4)  # Increase rate
        else:
            multiplier = 1.0  # No change
        
        # Apply multiplier with bounds
        new_rate = int(self.base_config.rate * multiplier)
        new_rate = max(
            int(self.base_config.rate * self.min_rate_multiplier),
            min(int(self.base_config.rate * self.max_rate_multiplier), new_rate)
        )
        
        # Update configuration if changed
        if new_rate != self.current_config.rate:
            old_rate = self.current_config.rate
            self.current_config.rate = new_rate
            
            # Create new limiter with adapted rate
            self.base_limiter = TokenBucketRateLimiter(
                f"{self.name}_adaptive", self.current_config
            )
            
            logger.info(
                "Rate adapted",
                name=self.name,
                old_rate=old_rate,
                new_rate=new_rate,
                failure_rate=failure_rate,
                multiplier=multiplier
            )
        
        # Reset counters
        self.success_count = 0
        self.failure_count = 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get adaptive rate limiter metrics."""
        base_metrics = self.base_limiter.get_metrics()
        base_metrics.update({
            "adaptive": True,
            "base_rate": self.base_config.rate,
            "current_rate": self.current_config.rate,
            "rate_multiplier": self.current_config.rate / self.base_config.rate,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "adaptation_window": self.adaptation_window
        })
        return base_metrics


class AsyncRateLimiter:
    """
    Async-compatible rate limiter for AI API calls.
    
    Simple wrapper around TokenBucketRateLimiter for easy integration.
    """
    
    def __init__(self, rate: int, per: int = 60, burst: Optional[int] = None):
        """
        Initialize async rate limiter.
        
        Args:
            rate: Requests per time period
            per: Time period in seconds
            burst: Burst capacity (defaults to rate/2)
        """
        config = RateLimiterConfig(
            rate=rate,
            per=per,
            burst=burst or max(1, rate // 2)
        )
        
        self.limiter = TokenBucketRateLimiter("async_limiter", config)
    
    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        await self.limiter.wait_for_tokens(1)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiter metrics."""
        return self.limiter.get_metrics()


class RateLimiter:
    """
    Centralized rate limiter manager for multiple services.
    
    Manages rate limiters for different AI services with
    service-specific configurations and monitoring.
    """
    
    def __init__(self):
        """Initialize rate limiter manager."""
        self.limiters: Dict[str, TokenBucketRateLimiter] = {}
        self.adaptive_limiters: Dict[str, AdaptiveRateLimiter] = {}
        
        # Default configurations for different services
        self.default_configs = {
            "deepseek": RateLimiterConfig(
                rate=settings.DEEPSEEK_RATE_LIMIT,
                per=60,
                burst=min(10, settings.DEEPSEEK_RATE_LIMIT // 6)
            ),
            "gemini": RateLimiterConfig(
                rate=settings.GEMINI_RATE_LIMIT,
                per=60,
                burst=min(10, settings.GEMINI_RATE_LIMIT // 6)
            ),
            "api": RateLimiterConfig(
                rate=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
                per=60,
                burst=settings.RATE_LIMIT_BURST_SIZE
            )
        }
        
        logger.info("Rate limiter manager initialized")
    
    def get_rate_limiter(
        self,
        service: str,
        config: Optional[RateLimiterConfig] = None
    ) -> TokenBucketRateLimiter:
        """
        Get or create rate limiter for service.
        
        Args:
            service: Service name (deepseek, gemini, api, etc.)
            config: Optional custom configuration
            
        Returns:
            TokenBucketRateLimiter instance
        """
        if service not in self.limiters:
            limiter_config = config or self.default_configs.get(service, RateLimiterConfig())
            self.limiters[service] = TokenBucketRateLimiter(service, limiter_config)
            
            logger.info("Rate limiter created", service=service, config=limiter_config)
        
        return self.limiters[service]
    
    def get_adaptive_limiter(
        self,
        service: str,
        config: Optional[RateLimiterConfig] = None
    ) -> AdaptiveRateLimiter:
        """
        Get or create adaptive rate limiter for service.
        
        Args:
            service: Service name
            config: Optional custom configuration
            
        Returns:
            AdaptiveRateLimiter instance
        """
        if service not in self.adaptive_limiters:
            limiter_config = config or self.default_configs.get(service, RateLimiterConfig())
            self.adaptive_limiters[service] = AdaptiveRateLimiter(service, limiter_config)
            
            logger.info("Adaptive rate limiter created", service=service)
        
        return self.adaptive_limiters[service]
    
    async def acquire(self, service: str, tokens: int = 1) -> bool:
        """
        Acquire tokens from service rate limiter.
        
        Args:
            service: Service name
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens acquired
        """
        limiter = self.get_rate_limiter(service)
        return await limiter.acquire(tokens)
    
    async def wait_for_service(self, service: str, tokens: int = 1) -> float:
        """
        Wait for tokens from service rate limiter.
        
        Args:
            service: Service name
            tokens: Number of tokens needed
            
        Returns:
            Time waited in seconds
        """
        limiter = self.get_rate_limiter(service)
        return await limiter.wait_for_tokens(tokens)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all rate limiters.
        
        Returns:
            Dict mapping service names to their metrics
        """
        metrics = {}
        
        # Regular rate limiters
        for service, limiter in self.limiters.items():
            metrics[service] = limiter.get_metrics()
        
        # Adaptive rate limiters
        for service, limiter in self.adaptive_limiters.items():
            metrics[f"{service}_adaptive"] = limiter.get_metrics()
        
        return metrics
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Get summary metrics across all rate limiters.
        
        Returns:
            Dict containing summary statistics
        """
        total_requests = 0
        total_denied = 0
        total_wait_time = 0.0
        
        for limiter in list(self.limiters.values()) + [al.base_limiter for al in self.adaptive_limiters.values()]:
            metrics = limiter.get_metrics()
            total_requests += metrics["total_requests"]
            total_denied += metrics["denied_requests"]
            total_wait_time += metrics["average_wait_time"] * metrics["total_requests"]
        
        avg_wait_time = (total_wait_time / total_requests) if total_requests > 0 else 0
        denial_rate = (total_denied / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "total_limiters": len(self.limiters) + len(self.adaptive_limiters),
            "total_requests": total_requests,
            "total_denied": total_denied,
            "overall_denial_rate": denial_rate,
            "average_wait_time": avg_wait_time
        }
    
    async def reset_all(self):
        """Reset all rate limiters."""
        for limiter in self.limiters.values():
            await limiter.reset()
        
        for adaptive_limiter in self.adaptive_limiters.values():
            await adaptive_limiter.base_limiter.reset()
        
        logger.info("All rate limiters reset")