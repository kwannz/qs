"""
Strategy Optimization API Endpoints
策略优化API端点

Provides AI-powered trading strategy optimization with parameter tuning,
performance analysis, and risk-adjusted recommendations.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator
import structlog

from config import settings
from core.cache import CacheManager, CacheKey, generate_hash
from core.rate_limiter import RateLimiter
from core.metrics import MetricsCollector, AIRequestTracker
from ai_clients.deepseek_client import DeepSeekClient
from ai_clients.gemini_client import GeminiClient


logger = structlog.get_logger(__name__)
router = APIRouter()


class StrategyConfig(BaseModel):
    """Strategy configuration model."""
    name: str = Field(..., description="Strategy name")
    type: str = Field(..., description="Strategy type (trend_following, mean_reversion, arbitrage)")
    parameters: Dict[str, Any] = Field(..., description="Strategy parameters")
    risk_parameters: Dict[str, Any] = Field(default_factory=dict, description="Risk management parameters")
    timeframe: str = Field(default="1h", description="Strategy timeframe")
    symbols: List[str] = Field(default_factory=list, description="Target symbols")


class BacktestResults(BaseModel):
    """Backtest results model."""
    total_returns: float = Field(..., description="Total returns percentage")
    annualized_returns: float = Field(..., description="Annualized returns")
    sharpe_ratio: float = Field(..., description="Sharpe ratio")
    sortino_ratio: Optional[float] = Field(None, description="Sortino ratio")
    max_drawdown: float = Field(..., description="Maximum drawdown percentage")
    win_rate: float = Field(..., description="Win rate percentage")
    profit_factor: float = Field(..., description="Profit factor")
    total_trades: int = Field(..., description="Total number of trades")
    avg_trade_duration: Optional[float] = Field(None, description="Average trade duration in hours")
    var_95: Optional[float] = Field(None, description="Value at Risk (95%)")
    
    @validator("win_rate", "total_returns", "annualized_returns")
    def validate_percentages(cls, v):
        """Validate percentage values are reasonable."""
        if not -100 <= v <= 1000:  # Allow for extreme cases but catch obvious errors
            raise ValueError("Percentage values must be between -100% and 1000%")
        return v


class MarketConditions(BaseModel):
    """Current market conditions model."""
    volatility_regime: str = Field(..., description="Volatility regime (low, medium, high)")
    trend_regime: str = Field(..., description="Trend regime (trending, ranging, transitional)")
    correlation_environment: str = Field(..., description="Correlation environment (high, medium, low)")
    liquidity_conditions: str = Field(..., description="Liquidity conditions (good, normal, poor)")
    market_sentiment: str = Field(..., description="Market sentiment (bullish, bearish, neutral)")
    economic_indicators: Dict[str, Any] = Field(default_factory=dict)


class StrategyOptimizationRequest(BaseModel):
    """Strategy optimization request model."""
    strategy_config: StrategyConfig
    backtest_results: BacktestResults
    market_conditions: MarketConditions
    optimization_goals: List[str] = Field(
        default=["maximize_sharpe", "minimize_drawdown"],
        description="Optimization goals"
    )
    use_validation: bool = Field(default=True, description="Use secondary AI for validation")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Optimization constraints")


class OptimizationRecommendation(BaseModel):
    """Single optimization recommendation."""
    parameter_name: str
    current_value: Any
    recommended_value: Any
    impact_description: str
    expected_improvement: Dict[str, float]
    confidence: int
    priority: str  # high, medium, low
    implementation_complexity: str  # low, medium, high


class StrategyOptimizationResponse(BaseModel):
    """Strategy optimization response model."""
    strategy_name: str
    optimization_type: str
    
    # Current performance summary
    current_performance: Dict[str, Any]
    
    # Optimization recommendations
    parameter_recommendations: List[OptimizationRecommendation]
    
    # Risk assessment
    risk_assessment: Dict[str, Any]
    
    # Expected improvements
    expected_performance: Dict[str, Any]
    
    # Implementation guidance
    implementation_plan: Dict[str, Any]
    
    # Validation results (if enabled)
    validation: Optional[Dict[str, Any]] = None
    feasibility_score: Optional[int] = None
    
    # Metadata
    optimized_at: datetime
    model_used: str
    processing_time_ms: int


# Dependency injection helpers
async def get_cache_manager() -> CacheManager:
    from main import app
    return app.state.cache_manager


async def get_rate_limiter() -> RateLimiter:
    from main import app
    return app.state.rate_limiter


async def get_metrics_collector() -> MetricsCollector:
    from main import app
    return app.state.metrics


async def get_deepseek_client() -> DeepSeekClient:
    from main import app
    return app.state.deepseek_client


async def get_gemini_client() -> GeminiClient:
    from main import app
    return app.state.gemini_client


@router.post("/optimize", response_model=StrategyOptimizationResponse)
async def optimize_strategy(
    request: StrategyOptimizationRequest,
    background_tasks: BackgroundTasks,
    cache_manager: CacheManager = Depends(get_cache_manager),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    metrics: MetricsCollector = Depends(get_metrics_collector),
    deepseek_client: DeepSeekClient = Depends(get_deepseek_client),
    gemini_client: GeminiClient = Depends(get_gemini_client)
):
    """
    Optimize trading strategy parameters using AI analysis.
    
    Analyzes current strategy performance and provides specific
    parameter optimization recommendations with risk assessment.
    """
    start_time = datetime.utcnow()
    
    try:
        # Apply rate limiting
        await rate_limiter.wait_for_service("strategy_optimization")
        
        # Generate cache key based on strategy config and results
        config_hash = generate_hash(request.strategy_config.dict())
        results_hash = generate_hash(request.backtest_results.dict())
        cache_key = CacheKey.strategy_optimization(
            request.strategy_config.name,
            f"{config_hash}_{results_hash}"
        )
        
        # Check cache
        cached_result = await cache_manager.get(cache_key)
        if cached_result:
            logger.info(
                "Strategy optimization served from cache",
                strategy_name=request.strategy_config.name,
                cache_key=str(cache_key)
            )
            
            cached_result["processing_time_ms"] = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )
            return StrategyOptimizationResponse(**cached_result)
        
        # Perform primary optimization analysis with DeepSeek
        with AIRequestTracker(
            metrics, "deepseek", "deepseek-chat", "strategy_optimization"
        ) as tracker:
            optimization_result = await deepseek_client.strategy_optimization(
                strategy_config=request.strategy_config.dict(),
                backtest_results=request.backtest_results.dict(),
                market_conditions=request.market_conditions.dict()
            )
            
            if "usage" in optimization_result:
                tracker.set_tokens_used(optimization_result["usage"].get("total_tokens", 0))
        
        # Perform validation analysis with Gemini if requested
        validation_result = None
        feasibility_score = None
        
        if request.use_validation:
            try:
                with AIRequestTracker(
                    metrics, "gemini", "gemini-1.5-flash", "strategy_validation"
                ) as tracker:
                    validation_result = await gemini_client.strategy_validation(
                        strategy_config=request.strategy_config.dict(),
                        optimization_recommendations=optimization_result,
                        market_conditions=request.market_conditions.dict()
                    )
                    feasibility_score = validation_result.get("feasibility_score")
                    
            except Exception as e:
                logger.warning(
                    "Strategy optimization validation failed",
                    strategy_name=request.strategy_config.name,
                    error=str(e)
                )
                validation_result = {"error": str(e), "feasibility_score": 50}
                feasibility_score = 50
        
        # Parse optimization recommendations
        recommendations = []
        raw_recommendations = optimization_result.get("optimization_recommendations", [])
        
        for rec in raw_recommendations:
            if isinstance(rec, dict):
                recommendations.append(OptimizationRecommendation(
                    parameter_name=rec.get("parameter_name", "unknown"),
                    current_value=rec.get("current_value"),
                    recommended_value=rec.get("recommended_value"),
                    impact_description=rec.get("impact_description", ""),
                    expected_improvement=rec.get("expected_improvement", {}),
                    confidence=rec.get("confidence", 70),
                    priority=rec.get("priority", "medium"),
                    implementation_complexity=rec.get("implementation_complexity", "medium")
                ))
        
        # Calculate processing time
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Prepare response
        response_data = {
            "strategy_name": request.strategy_config.name,
            "optimization_type": "ai_powered",
            "current_performance": {
                "sharpe_ratio": request.backtest_results.sharpe_ratio,
                "max_drawdown": request.backtest_results.max_drawdown,
                "win_rate": request.backtest_results.win_rate,
                "total_returns": request.backtest_results.total_returns,
                "profit_factor": request.backtest_results.profit_factor
            },
            "parameter_recommendations": [rec.dict() for rec in recommendations],
            "risk_assessment": optimization_result.get("risk_assessment", {}),
            "expected_performance": optimization_result.get("expected_improvements", {}),
            "implementation_plan": {
                "priority_order": [rec.parameter_name for rec in recommendations if rec.priority == "high"],
                "estimated_implementation_time": "2-5 days",
                "testing_requirements": ["backtest_validation", "paper_trading", "gradual_rollout"],
                "rollback_plan": "Keep current parameters as fallback"
            },
            "validation": validation_result,
            "feasibility_score": feasibility_score,
            "optimized_at": start_time,
            "model_used": "deepseek-chat",
            "processing_time_ms": processing_time
        }
        
        # Cache the result
        background_tasks.add_task(
            cache_manager.set,
            cache_key,
            response_data,
            settings.STRATEGY_ANALYSIS_CACHE_TTL
        )
        
        # Record business metrics
        metrics.record_analysis_accuracy(
            analysis_type="strategy_optimization",
            accuracy_score=feasibility_score or 70
        )
        
        logger.info(
            "Strategy optimization completed",
            strategy_name=request.strategy_config.name,
            recommendations_count=len(recommendations),
            feasibility_score=feasibility_score,
            processing_time_ms=processing_time
        )
        
        return StrategyOptimizationResponse(**response_data)
        
    except Exception as e:
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        logger.error(
            "Strategy optimization failed",
            strategy_name=request.strategy_config.name,
            processing_time_ms=processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Strategy optimization failed: {str(e)}"
        )


@router.post("/compare-strategies")
async def compare_strategies(
    strategies: List[StrategyConfig],
    market_conditions: MarketConditions,
    cache_manager: CacheManager = Depends(get_cache_manager),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    deepseek_client: DeepSeekClient = Depends(get_deepseek_client)
):
    """
    Compare multiple trading strategies and recommend the best approach.
    
    Analyzes multiple strategies against current market conditions
    and provides comparative analysis and recommendations.
    """
    if len(strategies) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 strategies allowed for comparison"
        )
    
    start_time = datetime.utcnow()
    
    try:
        # Apply rate limiting
        await rate_limiter.wait_for_service("strategy_comparison")
        
        # Generate cache key for the comparison
        strategies_hash = generate_hash([s.dict() for s in strategies])
        market_hash = generate_hash(market_conditions.dict())
        cache_key = f"strategy_comparison:v1:{strategies_hash}:{market_hash}"
        
        # Check cache
        cached_result = await cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        # Prepare comparison analysis
        comparison_data = {
            "strategies": [s.dict() for s in strategies],
            "market_conditions": market_conditions.dict(),
            "comparison_criteria": [
                "risk_adjusted_returns",
                "market_regime_compatibility", 
                "implementation_complexity",
                "scalability",
                "robustness"
            ]
        }
        
        # Perform AI-powered comparison
        system_prompt = """You are an expert quantitative analyst comparing trading strategies.
        
        Analyze the provided strategies against current market conditions and provide:
        1. Comparative performance analysis
        2. Risk-reward assessment for each strategy
        3. Market regime compatibility
        4. Implementation feasibility
        5. Ranked recommendations
        
        Provide structured JSON response with strategy rankings and detailed analysis."""
        
        messages = [
            {
                "role": "user",
                "content": f"Compare these trading strategies:\n\n{comparison_data}\n\nProvide detailed comparative analysis and rankings."
            }
        ]
        
        response = await deepseek_client.chat_completion(
            messages=messages,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=4000
        )
        
        content = response.get_content()
        
        # Try to parse structured response
        try:
            import json
            comparison_result = json.loads(content)
        except json.JSONDecodeError:
            comparison_result = {
                "raw_analysis": content,
                "rankings": [],
                "recommendation": "Unable to parse structured comparison"
            }
        
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        result = {
            "comparison_result": comparison_result,
            "strategies_analyzed": len(strategies),
            "market_conditions": market_conditions.dict(),
            "compared_at": start_time.isoformat(),
            "processing_time_ms": processing_time
        }
        
        # Cache the result
        await cache_manager.set(
            cache_key,
            result,
            settings.STRATEGY_ANALYSIS_CACHE_TTL
        )
        
        logger.info(
            "Strategy comparison completed",
            strategies_count=len(strategies),
            processing_time_ms=processing_time
        )
        
        return result
        
    except Exception as e:
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        logger.error(
            "Strategy comparison failed",
            strategies_count=len(strategies),
            processing_time_ms=processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Strategy comparison failed: {str(e)}"
        )


@router.get("/optimization-goals")
async def get_optimization_goals():
    """Get available optimization goals and their descriptions."""
    return {
        "optimization_goals": {
            "maximize_sharpe": {
                "description": "Maximize Sharpe ratio (risk-adjusted returns)",
                "priority": "high",
                "measurement": "sharpe_ratio"
            },
            "minimize_drawdown": {
                "description": "Minimize maximum drawdown",
                "priority": "high", 
                "measurement": "max_drawdown"
            },
            "maximize_returns": {
                "description": "Maximize absolute returns",
                "priority": "medium",
                "measurement": "total_returns"
            },
            "maximize_win_rate": {
                "description": "Maximize percentage of winning trades",
                "priority": "low",
                "measurement": "win_rate"
            },
            "minimize_volatility": {
                "description": "Minimize portfolio volatility",
                "priority": "medium",
                "measurement": "volatility"
            },
            "improve_sortino": {
                "description": "Improve Sortino ratio (downside risk focus)",
                "priority": "high",
                "measurement": "sortino_ratio"
            }
        },
        "strategy_types": [
            "trend_following",
            "mean_reversion", 
            "arbitrage",
            "momentum",
            "pairs_trading",
            "market_making"
        ],
        "market_regimes": [
            "trending_bull",
            "trending_bear", 
            "ranging_market",
            "high_volatility",
            "low_volatility"
        ]
    }


@router.get("/parameter-templates/{strategy_type}")
async def get_parameter_templates(strategy_type: str):
    """Get parameter templates for different strategy types."""
    templates = {
        "trend_following": {
            "parameters": {
                "fast_ma_period": {"default": 20, "range": [5, 50], "type": "int"},
                "slow_ma_period": {"default": 50, "range": [20, 200], "type": "int"},
                "rsi_period": {"default": 14, "range": [7, 30], "type": "int"},
                "rsi_overbought": {"default": 70, "range": [60, 80], "type": "float"},
                "rsi_oversold": {"default": 30, "range": [20, 40], "type": "float"}
            },
            "risk_parameters": {
                "stop_loss_pct": {"default": 2.0, "range": [0.5, 5.0], "type": "float"},
                "take_profit_pct": {"default": 4.0, "range": [1.0, 10.0], "type": "float"},
                "position_size_pct": {"default": 10.0, "range": [1.0, 25.0], "type": "float"}
            }
        },
        "mean_reversion": {
            "parameters": {
                "lookback_period": {"default": 20, "range": [10, 60], "type": "int"},
                "zscore_entry": {"default": 2.0, "range": [1.5, 3.0], "type": "float"},
                "zscore_exit": {"default": 0.5, "range": [0.0, 1.0], "type": "float"},
                "bollinger_period": {"default": 20, "range": [10, 50], "type": "int"}
            },
            "risk_parameters": {
                "max_position_size": {"default": 5.0, "range": [1.0, 15.0], "type": "float"},
                "stop_loss_zscore": {"default": 3.0, "range": [2.5, 4.0], "type": "float"}
            }
        }
    }
    
    if strategy_type not in templates:
        raise HTTPException(
            status_code=404,
            detail=f"No parameter template found for strategy type: {strategy_type}"
        )
    
    return {
        "strategy_type": strategy_type,
        "template": templates[strategy_type],
        "optimization_tips": [
            "Start with conservative parameters and gradually optimize",
            "Always backtest parameter changes thoroughly",
            "Consider market regime when selecting parameters",
            "Monitor performance degradation after optimization"
        ]
    }