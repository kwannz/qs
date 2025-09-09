"""
Metrics Collection System
指标收集系统

Comprehensive metrics collection for AI service monitoring,
including Prometheus integration and custom business metrics.
"""

import time
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import structlog
from prometheus_client import Counter, Histogram, Gauge, Info, CollectorRegistry

from config import settings


logger = structlog.get_logger(__name__)


@dataclass
class AIRequestMetrics:
    """AI request metrics tracking."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens_used: int = 0
    total_response_time: float = 0.0
    avg_response_time: float = 0.0
    
    def record_request(self, success: bool, tokens: int, response_time: float):
        """Record a request with metrics."""
        self.total_requests += 1
        self.total_tokens_used += tokens
        self.total_response_time += response_time
        
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.avg_response_time = self.total_response_time / self.total_requests
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100


class MetricsCollector:
    """
    Comprehensive metrics collector for AI service monitoring.
    
    Features:
    - Prometheus metrics integration
    - Custom business metrics tracking
    - System resource monitoring
    - AI model performance metrics
    - Cache and rate limiter metrics
    - Health status tracking
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.registry = CollectorRegistry()
        self.start_time = time.time()
        
        # AI Request Metrics
        self.ai_requests = Counter(
            'ai_requests_total',
            'Total number of AI requests',
            ['provider', 'model', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.ai_request_duration = Histogram(
            'ai_request_duration_seconds',
            'Duration of AI requests',
            ['provider', 'model', 'endpoint'],
            registry=self.registry
        )
        
        self.ai_tokens_used = Counter(
            'ai_tokens_used_total',
            'Total tokens used by AI requests',
            ['provider', 'model'],
            registry=self.registry
        )
        
        # Cache Metrics
        self.cache_operations = Counter(
            'cache_operations_total',
            'Total cache operations',
            ['operation', 'status'],
            registry=self.registry
        )
        
        self.cache_hit_rate = Gauge(
            'cache_hit_rate',
            'Cache hit rate percentage',
            registry=self.registry
        )
        
        # Rate Limiter Metrics
        self.rate_limit_requests = Counter(
            'rate_limit_requests_total',
            'Total rate limit requests',
            ['service', 'status'],
            registry=self.registry
        )
        
        # Circuit Breaker Metrics
        self.circuit_breaker_state = Gauge(
            'circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=half-open, 2=open)',
            ['service'],
            registry=self.registry
        )
        
        # System Metrics
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage',
            registry=self.registry
        )
        
        # Application Metrics
        self.app_uptime = Gauge(
            'app_uptime_seconds',
            'Application uptime in seconds',
            registry=self.registry
        )
        
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        # Business Metrics
        self.analysis_accuracy = Histogram(
            'analysis_accuracy_score',
            'Analysis accuracy scores',
            ['analysis_type'],
            registry=self.registry
        )
        
        self.trading_signals_generated = Counter(
            'trading_signals_generated_total',
            'Total trading signals generated',
            ['signal_type', 'confidence_level'],
            registry=self.registry
        )
        
        # Service Info
        self.service_info = Info(
            'service_info',
            'Service information',
            registry=self.registry
        )
        
        self.service_info.info({
            'version': '1.0.0',
            'environment': settings.ENVIRONMENT,
            'ai_providers': 'deepseek,gemini'
        })
        
        # Internal metrics tracking
        self.ai_metrics: Dict[str, AIRequestMetrics] = {}
        self.custom_metrics: Dict[str, Any] = {}
        
        logger.info("Metrics collector initialized")
    
    def record_ai_request(
        self,
        provider: str,
        model: str,
        endpoint: str,
        success: bool,
        duration: float,
        tokens_used: int = 0,
        error_type: Optional[str] = None
    ):
        """
        Record AI request metrics.
        
        Args:
            provider: AI provider name (deepseek, gemini)
            model: Model name used
            endpoint: API endpoint called
            success: Whether request was successful
            duration: Request duration in seconds
            tokens_used: Number of tokens consumed
            error_type: Type of error if request failed
        """
        status = "success" if success else "error"
        
        # Prometheus metrics
        self.ai_requests.labels(
            provider=provider,
            model=model,
            endpoint=endpoint,
            status=status
        ).inc()
        
        self.ai_request_duration.labels(
            provider=provider,
            model=model,
            endpoint=endpoint
        ).observe(duration)
        
        if tokens_used > 0:
            self.ai_tokens_used.labels(
                provider=provider,
                model=model
            ).inc(tokens_used)
        
        # Internal metrics tracking
        key = f"{provider}_{model}_{endpoint}"
        if key not in self.ai_metrics:
            self.ai_metrics[key] = AIRequestMetrics()
        
        self.ai_metrics[key].record_request(success, tokens_used, duration)
        
        logger.debug(
            "AI request metrics recorded",
            provider=provider,
            model=model,
            endpoint=endpoint,
            success=success,
            duration=duration,
            tokens_used=tokens_used
        )
    
    def record_cache_operation(self, operation: str, success: bool, hit_rate: Optional[float] = None):
        """
        Record cache operation metrics.
        
        Args:
            operation: Cache operation (get, set, delete)
            success: Whether operation was successful
            hit_rate: Current cache hit rate
        """
        status = "success" if success else "error"
        
        self.cache_operations.labels(
            operation=operation,
            status=status
        ).inc()
        
        if hit_rate is not None:
            self.cache_hit_rate.set(hit_rate)
    
    def record_rate_limit_request(self, service: str, allowed: bool):
        """
        Record rate limit request.
        
        Args:
            service: Service name
            allowed: Whether request was allowed
        """
        status = "allowed" if allowed else "denied"
        
        self.rate_limit_requests.labels(
            service=service,
            status=status
        ).inc()
    
    def update_circuit_breaker_state(self, service: str, state: str):
        """
        Update circuit breaker state.
        
        Args:
            service: Service name
            state: Circuit breaker state (closed, half_open, open)
        """
        state_value = {
            "closed": 0,
            "half_open": 1,
            "open": 2
        }.get(state, 0)
        
        self.circuit_breaker_state.labels(service=service).set(state_value)
    
    def update_health_status(self, **health_checks: bool):
        """
        Update health status for various components.
        
        Args:
            **health_checks: Component health statuses
        """
        for component, healthy in health_checks.items():
            self.custom_metrics[f"{component}_healthy"] = healthy
    
    async def collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.system_memory_usage.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.system_disk_usage.set(disk_percent)
            
            # Application uptime
            uptime = time.time() - self.start_time
            self.app_uptime.set(uptime)
            
            # Network connections (approximate active connections)
            connections = len(psutil.net_connections())
            self.active_connections.set(connections)
            
            logger.debug(
                "System metrics collected",
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=disk_percent,
                uptime=uptime
            )
            
        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
    
    def record_analysis_accuracy(self, analysis_type: str, accuracy_score: float):
        """
        Record analysis accuracy metrics.
        
        Args:
            analysis_type: Type of analysis (market, strategy, risk)
            accuracy_score: Accuracy score (0-100)
        """
        self.analysis_accuracy.labels(analysis_type=analysis_type).observe(accuracy_score)
    
    def record_trading_signal(self, signal_type: str, confidence_level: str):
        """
        Record trading signal generation.
        
        Args:
            signal_type: Type of signal (buy, sell, hold)
            confidence_level: Confidence level (low, medium, high)
        """
        self.trading_signals_generated.labels(
            signal_type=signal_type,
            confidence_level=confidence_level
        ).inc()
    
    def get_prometheus_metrics(self) -> str:
        """
        Get Prometheus formatted metrics.
        
        Returns:
            Prometheus formatted metrics string
        """
        from prometheus_client import generate_latest
        return generate_latest(self.registry)
    
    def get_ai_metrics_summary(self) -> Dict[str, Any]:
        """
        Get AI metrics summary.
        
        Returns:
            Dict containing AI metrics summary
        """
        if not self.ai_metrics:
            return {"total_services": 0}
        
        total_requests = sum(m.total_requests for m in self.ai_metrics.values())
        total_successful = sum(m.successful_requests for m in self.ai_metrics.values())
        total_failed = sum(m.failed_requests for m in self.ai_metrics.values())
        total_tokens = sum(m.total_tokens_used for m in self.ai_metrics.values())
        avg_response_time = sum(m.avg_response_time for m in self.ai_metrics.values()) / len(self.ai_metrics)
        
        return {
            "total_services": len(self.ai_metrics),
            "total_requests": total_requests,
            "total_successful": total_successful,
            "total_failed": total_failed,
            "overall_success_rate": (total_successful / total_requests * 100) if total_requests > 0 else 0,
            "total_tokens_used": total_tokens,
            "average_response_time": avg_response_time,
            "services": {
                key: {
                    "requests": metrics.total_requests,
                    "success_rate": metrics.success_rate,
                    "avg_response_time": metrics.avg_response_time,
                    "tokens_used": metrics.total_tokens_used
                }
                for key, metrics in self.ai_metrics.items()
            }
        }
    
    def get_system_metrics_summary(self) -> Dict[str, Any]:
        """
        Get system metrics summary.
        
        Returns:
            Dict containing system metrics
        """
        try:
            return {
                "cpu_usage_percent": psutil.cpu_percent(),
                "memory": {
                    "usage_percent": psutil.virtual_memory().percent,
                    "available_gb": psutil.virtual_memory().available / (1024**3),
                    "total_gb": psutil.virtual_memory().total / (1024**3)
                },
                "disk": {
                    "usage_percent": (psutil.disk_usage('/').used / psutil.disk_usage('/').total) * 100,
                    "free_gb": psutil.disk_usage('/').free / (1024**3),
                    "total_gb": psutil.disk_usage('/').total / (1024**3)
                },
                "uptime_seconds": time.time() - self.start_time,
                "active_connections": len(psutil.net_connections())
            }
        except Exception as e:
            logger.error("Failed to get system metrics summary", error=str(e))
            return {"error": str(e)}
    
    def get_business_metrics_summary(self) -> Dict[str, Any]:
        """
        Get business metrics summary.
        
        Returns:
            Dict containing business metrics
        """
        return {
            "custom_metrics": self.custom_metrics,
            "uptime_hours": (time.time() - self.start_time) / 3600,
            "service_start_time": datetime.fromtimestamp(self.start_time).isoformat()
        }
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics report.
        
        Returns:
            Dict containing all metrics categories
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "ai_metrics": self.get_ai_metrics_summary(),
            "system_metrics": self.get_system_metrics_summary(),
            "business_metrics": self.get_business_metrics_summary()
        }
    
    def export_metrics_to_file(self, filepath: str):
        """
        Export metrics to JSON file.
        
        Args:
            filepath: Path to export file
        """
        import json
        
        try:
            metrics = self.get_comprehensive_metrics()
            
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            
            logger.info("Metrics exported to file", filepath=filepath)
            
        except Exception as e:
            logger.error("Failed to export metrics", filepath=filepath, error=str(e))
    
    def reset_metrics(self):
        """Reset all metrics to initial state."""
        self.ai_metrics.clear()
        self.custom_metrics.clear()
        self.start_time = time.time()
        
        logger.info("Metrics reset to initial state")


# Metrics middleware for automatic request tracking
class MetricsMiddleware:
    """FastAPI middleware for automatic metrics collection."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize metrics middleware.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector
    
    async def __call__(self, request, call_next):
        """Process request with metrics collection."""
        start_time = time.time()
        path = request.url.path
        method = request.method
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Record HTTP request metrics
            self.metrics_collector.custom_metrics[f"http_{method.lower()}_requests"] = \
                self.metrics_collector.custom_metrics.get(f"http_{method.lower()}_requests", 0) + 1
            
            self.metrics_collector.custom_metrics[f"http_response_time_{path}"] = duration
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            self.metrics_collector.custom_metrics["http_errors"] = \
                self.metrics_collector.custom_metrics.get("http_errors", 0) + 1
            
            logger.error(
                "Request failed in metrics middleware",
                path=path,
                method=method,
                duration=duration,
                error=str(e)
            )
            raise


# Context manager for automatic AI request metrics
class AIRequestTracker:
    """Context manager for tracking AI requests."""
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        provider: str,
        model: str,
        endpoint: str
    ):
        """
        Initialize AI request tracker.
        
        Args:
            metrics_collector: Metrics collector instance
            provider: AI provider name
            model: Model name
            endpoint: Endpoint name
        """
        self.metrics_collector = metrics_collector
        self.provider = provider
        self.model = model
        self.endpoint = endpoint
        self.start_time = None
        self.tokens_used = 0
    
    def __enter__(self):
        """Start tracking AI request."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finish tracking AI request."""
        if self.start_time:
            duration = time.time() - self.start_time
            success = exc_type is None
            
            self.metrics_collector.record_ai_request(
                provider=self.provider,
                model=self.model,
                endpoint=self.endpoint,
                success=success,
                duration=duration,
                tokens_used=self.tokens_used,
                error_type=exc_type.__name__ if exc_type else None
            )
    
    def set_tokens_used(self, tokens: int):
        """Set number of tokens used in request."""
        self.tokens_used = tokens


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector