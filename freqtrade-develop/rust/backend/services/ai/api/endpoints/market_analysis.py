"""
Market Analysis API Endpoints
市场分析API端点

Provides intelligent market analysis capabilities using AI models
for trend analysis, pattern recognition, and trading recommendations.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator
import structlog

from config import settings
from core.cache import CacheManager, CacheKey, cache_with_fallback
from core.rate_limiter import RateLimiter
from core.metrics import MetricsCollector, AIRequestTracker
from ai_clients.deepseek_client import DeepSeekClient
from ai_clients.gemini_client import GeminiClient


logger = structlog.get_logger(__name__)
router = APIRouter()


# Request/Response Models
class MarketAnalysisRequest(BaseModel):
    """Market analysis request model."""
    symbol: str = Field(..., description="Trading symbol (e.g., BTCUSDT)")
    timeframe: str = Field(default="1h", description="Analysis timeframe (1m, 5m, 1h, 4h, 1d)")
    analysis_type: str = Field(default="trend", description="Analysis type (trend, pattern, momentum, volatility)")
    market_data: Dict[str, Any] = Field(..., description="Historical market data")
    use_validation: bool = Field(default=True, description="Use secondary AI for validation")
    priority: str = Field(default="normal", description="Request priority (low, normal, high)")
    
    @validator("symbol")
    def validate_symbol(cls, v):
        """Validate trading symbol format."""
        if not v or len(v) < 3:
            raise ValueError("Symbol must be at least 3 characters")
        return v.upper()
    
    @validator("timeframe")
    def validate_timeframe(cls, v):
        """Validate timeframe values."""
        valid_timeframes = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "3d", "1w"]
        if v not in valid_timeframes:
            raise ValueError(f"Timeframe must be one of {valid_timeframes}")
        return v
    
    @validator("analysis_type")
    def validate_analysis_type(cls, v):
        """Validate analysis type values."""
        valid_types = ["trend", "pattern", "momentum", "volatility", "support_resistance", "comprehensive"]
        if v not in valid_types:
            raise ValueError(f"Analysis type must be one of {valid_types}")
        return v


class MarketAnalysisResponse(BaseModel):
    """Market analysis response model."""
    symbol: str
    timeframe: str
    analysis_type: str
    
    # Primary analysis results
    trend_direction: str = Field(description="Trend direction (bullish/bearish/neutral/consolidating)")
    trend_strength: int = Field(description="Trend strength (1-10 scale)")
    support_levels: List[float] = Field(default_factory=list)
    resistance_levels: List[float] = Field(default_factory=list)
    
    # Technical indicators
    indicators: Dict[str, Any] = Field(default_factory=dict)
    
    # Price targets and risk
    price_targets: Dict[str, float] = Field(default_factory=dict)
    risk_level: str = Field(description="Risk level (low/medium/high/extreme)")
    confidence: int = Field(description="Analysis confidence (1-100)")
    
    # Trading recommendations
    recommendation: str = Field(description="Trading recommendation")
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Validation results (if enabled)
    validation: Optional[Dict[str, Any]] = None
    consensus_rating: Optional[int] = None
    
    # Metadata
    analyzed_at: datetime
    model_used: str
    request_id: Optional[str] = None
    processing_time_ms: int


class BulkAnalysisRequest(BaseModel):
    """Bulk market analysis request."""
    symbols: List[str] = Field(..., max_items=50, description="List of trading symbols")
    timeframe: str = Field(default="1h")
    analysis_type: str = Field(default="trend")
    use_validation: bool = Field(default=False, description="Disable validation for bulk requests by default")
    priority: str = Field(default="low")


# Dependency injection
async def get_cache_manager() -> CacheManager:
    """Get cache manager dependency."""
    # This would be injected from the main app state
    from main import app
    return app.state.cache_manager


async def get_rate_limiter() -> RateLimiter:
    """Get rate limiter dependency."""
    from main import app
    return app.state.rate_limiter


async def get_metrics_collector() -> MetricsCollector:
    """Get metrics collector dependency."""
    from main import app
    return app.state.metrics


async def get_deepseek_client() -> DeepSeekClient:
    """Get DeepSeek client dependency."""
    from main import app
    return app.state.deepseek_client


async def get_gemini_client() -> GeminiClient:
    """Get Gemini client dependency."""
    from main import app
    return app.state.gemini_client


@router.post("/analyze", response_model=MarketAnalysisResponse)
async def analyze_market(
    request: MarketAnalysisRequest,
    background_tasks: BackgroundTasks,
    cache_manager: CacheManager = Depends(get_cache_manager),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    metrics: MetricsCollector = Depends(get_metrics_collector),
    deepseek_client: DeepSeekClient = Depends(get_deepseek_client),
    gemini_client: GeminiClient = Depends(get_gemini_client)
):
    """
    Perform intelligent market analysis for a trading symbol.
    
    This endpoint provides comprehensive market analysis using AI models,
    including trend analysis, pattern recognition, and trading recommendations.
    """
    start_time = datetime.utcnow()
    
    try:
        # Apply rate limiting
        await rate_limiter.wait_for_service("market_analysis")
        
        # Generate cache key
        cache_key = CacheKey.market_analysis(
            request.symbol,
            request.timeframe,
            request.analysis_type
        )
        
        # Try to get from cache first
        cached_result = await cache_manager.get(cache_key)
        if cached_result:
            logger.info(
                "Market analysis served from cache",
                symbol=request.symbol,
                timeframe=request.timeframe,
                cache_key=str(cache_key)
            )
            
            # Convert cached result to response model
            cached_result["processing_time_ms"] = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )
            return MarketAnalysisResponse(**cached_result)
        
        # Perform primary analysis with DeepSeek
        with AIRequestTracker(
            metrics, "deepseek", "deepseek-chat", "market_analysis"
        ) as tracker:
            primary_analysis = await deepseek_client.market_analysis(
                symbol=request.symbol,
                timeframe=request.timeframe,
                market_data=request.market_data,
                analysis_type=request.analysis_type
            )
            
            # Track tokens if available
            if "usage" in primary_analysis:
                tracker.set_tokens_used(primary_analysis["usage"].get("total_tokens", 0))
        
        # Perform validation analysis with Gemini if requested
        validation_result = None
        consensus_rating = None
        
        if request.use_validation:
            try:
                with AIRequestTracker(
                    metrics, "gemini", "gemini-1.5-flash", "market_validation"
                ) as tracker:
                    validation_result = await gemini_client.market_analysis_validation(
                        symbol=request.symbol,
                        primary_analysis=primary_analysis,
                        market_data=request.market_data,
                        analysis_type="validation"
                    )
                    consensus_rating = validation_result.get("consensus_rating")
                    
            except Exception as e:
                logger.warning(
                    "Market analysis validation failed, continuing with primary analysis",
                    symbol=request.symbol,
                    error=str(e)
                )
                validation_result = {"error": str(e), "consensus_rating": 50}
                consensus_rating = 50
        
        # Prepare response
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        response_data = {
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "analysis_type": request.analysis_type,
            "trend_direction": primary_analysis.get("trend_direction", "neutral"),
            "trend_strength": primary_analysis.get("trend_strength", 5),
            "support_levels": primary_analysis.get("support_levels", []),
            "resistance_levels": primary_analysis.get("resistance_levels", []),
            "indicators": primary_analysis.get("indicators", {}),
            "price_targets": primary_analysis.get("price_targets", {}),
            "risk_level": primary_analysis.get("risk_level", "medium"),
            "confidence": primary_analysis.get("confidence", 70),
            "recommendation": primary_analysis.get("recommendation", "Hold"),
            "entry_price": primary_analysis.get("entry_price"),
            "stop_loss": primary_analysis.get("stop_loss"),
            "take_profit": primary_analysis.get("take_profit"),
            "validation": validation_result,
            "consensus_rating": consensus_rating,
            "analyzed_at": start_time,
            "model_used": "deepseek-chat",
            "request_id": primary_analysis.get("request_id"),
            "processing_time_ms": processing_time
        }
        
        # Cache the result
        background_tasks.add_task(
            cache_manager.set,
            cache_key,
            response_data,
            settings.MARKET_ANALYSIS_CACHE_TTL
        )
        
        # Record business metrics
        metrics.record_analysis_accuracy(
            analysis_type=request.analysis_type,
            accuracy_score=response_data["confidence"]
        )
        
        logger.info(
            "Market analysis completed",
            symbol=request.symbol,
            timeframe=request.timeframe,
            processing_time_ms=processing_time,
            trend=response_data["trend_direction"],
            confidence=response_data["confidence"]
        )
        
        return MarketAnalysisResponse(**response_data)
        
    except Exception as e:
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        logger.error(
            "Market analysis failed",
            symbol=request.symbol,
            timeframe=request.timeframe,
            processing_time_ms=processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Market analysis failed: {str(e)}"
        )


@router.post("/analyze/bulk")
async def bulk_analyze_market(
    request: BulkAnalysisRequest,
    background_tasks: BackgroundTasks,
    cache_manager: CacheManager = Depends(get_cache_manager),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    metrics: MetricsCollector = Depends(get_metrics_collector),
    deepseek_client: DeepSeekClient = Depends(get_deepseek_client)
):
    """
    Perform bulk market analysis for multiple symbols.
    
    Processes multiple symbols efficiently with intelligent caching
    and parallel processing where possible.
    """
    start_time = datetime.utcnow()
    
    if len(request.symbols) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 symbols allowed in bulk analysis"
        )
    
    try:
        results = []
        cache_hits = 0
        cache_misses = 0
        
        # Check cache for each symbol
        cache_keys = [
            CacheKey.market_analysis(symbol, request.timeframe, request.analysis_type)
            for symbol in request.symbols
        ]
        
        cached_results = await cache_manager.get_many(cache_keys)
        
        # Separate cached and non-cached symbols
        symbols_to_analyze = []
        for i, symbol in enumerate(request.symbols):
            cache_key = str(cache_keys[i])
            if cache_key in cached_results:
                cached_data = cached_results[cache_key]
                cached_data["symbol"] = symbol  # Ensure symbol is set
                cached_data["processing_time_ms"] = 0  # Cached result
                results.append(cached_data)
                cache_hits += 1
            else:
                symbols_to_analyze.append(symbol)
                cache_misses += 1
        
        # Analyze remaining symbols with controlled concurrency
        if symbols_to_analyze:
            semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_AI_REQUESTS)
            
            async def analyze_single_symbol(symbol: str):
                """Analyze single symbol with rate limiting."""
                async with semaphore:
                    # Apply rate limiting per request
                    await rate_limiter.wait_for_service("market_analysis")
                    
                    try:
                        # Mock market data for bulk analysis (would fetch real data in production)
                        market_data = {
                            "current_price": 50000.0,
                            "volume_24h": 1000000000,
                            "price_change_24h": 2.5
                        }
                        
                        with AIRequestTracker(
                            metrics, "deepseek", "deepseek-chat", "bulk_market_analysis"
                        ) as tracker:
                            analysis = await deepseek_client.market_analysis(
                                symbol=symbol,
                                timeframe=request.timeframe,
                                market_data=market_data,
                                analysis_type=request.analysis_type
                            )
                            
                            if "usage" in analysis:
                                tracker.set_tokens_used(analysis["usage"].get("total_tokens", 0))
                        
                        # Format result
                        result = {
                            "symbol": symbol,
                            "timeframe": request.timeframe,
                            "analysis_type": request.analysis_type,
                            "trend_direction": analysis.get("trend_direction", "neutral"),
                            "trend_strength": analysis.get("trend_strength", 5),
                            "confidence": analysis.get("confidence", 70),
                            "recommendation": analysis.get("recommendation", "Hold"),
                            "risk_level": analysis.get("risk_level", "medium"),
                            "analyzed_at": datetime.utcnow(),
                            "model_used": "deepseek-chat",
                            "processing_time_ms": 0  # Will be updated
                        }
                        
                        # Cache the result in background
                        cache_key = CacheKey.market_analysis(symbol, request.timeframe, request.analysis_type)
                        background_tasks.add_task(
                            cache_manager.set,
                            cache_key,
                            result,
                            settings.MARKET_ANALYSIS_CACHE_TTL
                        )
                        
                        return result
                        
                    except Exception as e:
                        logger.error(
                            "Bulk analysis failed for symbol",
                            symbol=symbol,
                            error=str(e)
                        )
                        
                        return {
                            "symbol": symbol,
                            "error": str(e),
                            "trend_direction": "unknown",
                            "confidence": 0,
                            "recommendation": "Analysis failed"
                        }
            
            # Execute analyses concurrently
            analysis_tasks = [analyze_single_symbol(symbol) for symbol in symbols_to_analyze]
            analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Add successful results
            for result in analysis_results:
                if not isinstance(result, Exception):
                    results.append(result)
        
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Update processing time for all results
        for result in results:
            if result.get("processing_time_ms") == 0:
                result["processing_time_ms"] = processing_time
        
        logger.info(
            "Bulk market analysis completed",
            total_symbols=len(request.symbols),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            processing_time_ms=processing_time
        )
        
        return {
            "results": results,
            "summary": {
                "total_symbols": len(request.symbols),
                "successful_analyses": len(results),
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "processing_time_ms": processing_time,
                "analyzed_at": start_time.isoformat()
            }
        }
        
    except Exception as e:
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        logger.error(
            "Bulk market analysis failed",
            symbols_count=len(request.symbols),
            processing_time_ms=processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Bulk market analysis failed: {str(e)}"
        )


@router.get("/supported-symbols")
async def get_supported_symbols():
    """Get list of supported trading symbols for analysis."""
    # This would typically come from your market data service
    supported_symbols = [
        "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
        "BNBUSDT", "SOLUSDT", "XRPUSDT", "MATICUSDT", "AVAXUSDT",
        "LTCUSDT", "BCHUSDT", "FILUSDT", "TRXUSDT", "ETCUSDT"
    ]
    
    return {
        "supported_symbols": supported_symbols,
        "total_count": len(supported_symbols),
        "categories": {
            "major": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            "altcoins": ["ADAUSDT", "DOTUSDT", "LINKUSDT", "SOLUSDT"],
            "defi": ["AVAXUSDT", "MATICUSDT"],
            "legacy": ["LTCUSDT", "BCHUSDT", "ETCUSDT"]
        }
    }


@router.get("/analysis-types")
async def get_analysis_types():
    """Get available analysis types and their descriptions."""
    return {
        "analysis_types": {
            "trend": {
                "description": "Trend direction and strength analysis",
                "outputs": ["trend_direction", "trend_strength", "momentum_indicators"]
            },
            "pattern": {
                "description": "Chart pattern recognition and analysis",
                "outputs": ["pattern_type", "pattern_confidence", "breakout_targets"]
            },
            "momentum": {
                "description": "Momentum indicators and oscillator analysis",
                "outputs": ["rsi", "macd", "stochastic", "momentum_score"]
            },
            "volatility": {
                "description": "Volatility analysis and risk assessment",
                "outputs": ["volatility_score", "bollinger_bands", "atr"]
            },
            "support_resistance": {
                "description": "Key support and resistance level identification",
                "outputs": ["support_levels", "resistance_levels", "pivot_points"]
            },
            "comprehensive": {
                "description": "Complete analysis combining all methods",
                "outputs": ["all_indicators", "weighted_recommendation", "risk_reward_ratio"]
            }
        },
        "timeframes": ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "3d", "1w"],
        "confidence_levels": {
            "high": "> 80%",
            "medium": "50-80%", 
            "low": "< 50%"
        }
    }