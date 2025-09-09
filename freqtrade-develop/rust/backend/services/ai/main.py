"""
AI Service - Main FastAPI Application
智能分析服务的主要应用入口

This service provides AI-powered market analysis, strategy optimization,
and risk assessment for the cryptocurrency trading platform.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
import structlog

from config import settings
from core.cache import CacheManager
from core.circuit_breaker import CircuitBreakerManager
from core.rate_limiter import RateLimiter
from core.metrics import MetricsCollector
from api.endpoints import market_analysis, strategy_optimization, risk_assessment, news_sentiment
from ai_clients.deepseek_client import DeepSeekClient
from ai_clients.gemini_client import GeminiClient

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    logger.info("Starting AI Service", version="1.0.0", environment=settings.ENVIRONMENT)
    
    try:
        # Initialize core components
        app.state.cache_manager = CacheManager()
        app.state.circuit_breaker = CircuitBreakerManager()
        app.state.rate_limiter = RateLimiter()
        app.state.metrics = MetricsCollector()
        
        # Initialize AI clients
        app.state.deepseek_client = DeepSeekClient()
        app.state.gemini_client = GeminiClient()
        
        # Start background tasks
        asyncio.create_task(background_health_check())
        asyncio.create_task(background_metrics_collection())
        
        logger.info("AI Service started successfully")
        
        yield
        
    except Exception as e:
        logger.error("Failed to start AI Service", error=str(e))
        raise
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down AI Service")
        
        # Close connections and cleanup resources
        await app.state.cache_manager.close()
        await app.state.deepseek_client.close()
        await app.state.gemini_client.close()
        
        logger.info("AI Service shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="AI Analysis Service",
    description="Intelligent market analysis and trading strategy optimization service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Dependency injection
async def get_cache_manager(request: Request) -> CacheManager:
    """Get cache manager instance."""
    return request.app.state.cache_manager


async def get_circuit_breaker(request: Request) -> CircuitBreakerManager:
    """Get circuit breaker manager instance."""
    return request.app.state.circuit_breaker


async def get_rate_limiter(request: Request) -> RateLimiter:
    """Get rate limiter instance."""
    return request.app.state.rate_limiter


async def get_metrics_collector(request: Request) -> MetricsCollector:
    """Get metrics collector instance."""
    return request.app.state.metrics


async def get_deepseek_client(request: Request) -> DeepSeekClient:
    """Get DeepSeek AI client instance."""
    return request.app.state.deepseek_client


async def get_gemini_client(request: Request) -> GeminiClient:
    """Get Gemini AI client instance."""
    return request.app.state.gemini_client


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for service monitoring."""
    try:
        # Basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time(),
            "version": "1.0.0",
            "environment": settings.ENVIRONMENT,
            "dependencies": {}
        }
        
        # Check Redis connection
        try:
            cache_manager = app.state.cache_manager
            await cache_manager.health_check()
            health_status["dependencies"]["redis"] = "healthy"
        except Exception as e:
            health_status["dependencies"]["redis"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check AI service availability
        try:
            deepseek_client = app.state.deepseek_client
            await deepseek_client.health_check()
            health_status["dependencies"]["deepseek"] = "healthy"
        except Exception as e:
            health_status["dependencies"]["deepseek"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
        
        try:
            gemini_client = app.state.gemini_client
            await gemini_client.health_check()
            health_status["dependencies"]["gemini"] = "healthy"
        except Exception as e:
            health_status["dependencies"]["gemini"] = f"unhealthy: {str(e)}"
            # Don't mark as degraded if only one AI provider is down
            
        return JSONResponse(content=health_status)
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=503
        )


# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    try:
        metrics_data = generate_latest()
        return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logger.error("Failed to generate metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate metrics")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with structured logging."""
    logger.warning(
        "HTTP exception occurred",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "path": request.url.path}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with structured logging."""
    logger.error(
        "Unhandled exception occurred",
        error=str(exc),
        path=request.url.path,
        method=request.method,
        exc_info=True
    )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "path": request.url.path}
    )


# Include API routers
app.include_router(
    market_analysis.router,
    prefix="/api/v1/market-analysis",
    tags=["Market Analysis"]
)

app.include_router(
    strategy_optimization.router,
    prefix="/api/v1/strategy-optimization",
    tags=["Strategy Optimization"]
)

app.include_router(
    risk_assessment.router,
    prefix="/api/v1/risk-assessment",
    tags=["Risk Assessment"]
)

app.include_router(
    news_sentiment.router,
    prefix="/api/v1/news-sentiment",
    tags=["News & Sentiment Analysis"]
)


# Background tasks
async def background_health_check():
    """Background task for periodic health monitoring."""
    while True:
        try:
            await asyncio.sleep(settings.HEALTH_CHECK_INTERVAL)
            
            # Perform background health checks
            cache_healthy = await app.state.cache_manager.health_check()
            deepseek_healthy = await app.state.deepseek_client.health_check()
            gemini_healthy = await app.state.gemini_client.health_check()
            
            # Update metrics
            app.state.metrics.update_health_status(
                cache=cache_healthy,
                deepseek=deepseek_healthy,
                gemini=gemini_healthy
            )
            
        except Exception as e:
            logger.error("Background health check failed", error=str(e))


async def background_metrics_collection():
    """Background task for metrics collection."""
    while True:
        try:
            await asyncio.sleep(60)  # Collect metrics every minute
            
            # Collect system metrics
            await app.state.metrics.collect_system_metrics()
            
        except Exception as e:
            logger.error("Background metrics collection failed", error=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.AI_SERVICE_HOST,
        port=settings.AI_SERVICE_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=settings.ENVIRONMENT == "development",
        workers=1 if settings.ENVIRONMENT == "development" else 4
    )