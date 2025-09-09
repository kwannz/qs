"""
Risk Assessment API Endpoints
风险评估API端点

Provides comprehensive portfolio risk assessment using AI analysis
for position sizing, risk metrics, and risk mitigation recommendations.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from decimal import Decimal

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


class Position(BaseModel):
    """Portfolio position model."""
    symbol: str = Field(..., description="Trading symbol")
    quantity: float = Field(..., description="Position quantity") 
    entry_price: float = Field(..., description="Average entry price")
    current_price: float = Field(..., description="Current market price")
    market_value: float = Field(..., description="Current market value")
    unrealized_pnl: float = Field(..., description="Unrealized P&L")
    position_weight: float = Field(..., description="Position weight in portfolio (%)")
    
    @validator("quantity", "entry_price", "current_price", "market_value")
    def validate_positive_values(cls, v):
        """Validate that financial values are positive."""
        if v <= 0:
            raise ValueError("Financial values must be positive")
        return v


class Portfolio(BaseModel):
    """Portfolio model."""
    total_value: float = Field(..., description="Total portfolio value")
    available_cash: float = Field(..., description="Available cash balance")
    positions: List[Position] = Field(..., description="Current positions")
    leverage_ratio: float = Field(default=1.0, description="Current leverage ratio")
    margin_used: float = Field(default=0.0, description="Margin currently used")
    
    @validator("total_value", "available_cash")
    def validate_portfolio_values(cls, v):
        """Validate portfolio values are non-negative."""
        if v < 0:
            raise ValueError("Portfolio values cannot be negative")
        return v


class MarketData(BaseModel):
    """Market data for risk assessment."""
    volatility_index: float = Field(..., description="Market volatility index (VIX-like)")
    correlation_matrix: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Asset correlation matrix"
    )
    liquidity_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Liquidity scores by symbol"
    )
    beta_values: Dict[str, float] = Field(
        default_factory=dict,
        description="Beta values relative to market"
    )
    historical_volatility: Dict[str, float] = Field(
        default_factory=dict,
        description="30-day historical volatility by symbol"
    )


class RiskParameters(BaseModel):
    """Risk management parameters."""
    max_portfolio_risk: float = Field(default=2.0, description="Maximum portfolio risk (%)")
    max_position_size: float = Field(default=10.0, description="Maximum single position size (%)")
    var_confidence_level: float = Field(default=0.95, description="VaR confidence level")
    max_correlation_exposure: float = Field(default=0.7, description="Maximum correlation exposure")
    liquidity_threshold: float = Field(default=0.3, description="Minimum liquidity threshold")
    
    @validator("var_confidence_level")
    def validate_confidence_level(cls, v):
        """Validate VaR confidence level is between 0 and 1."""
        if not 0 < v < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        return v


class RiskAssessmentRequest(BaseModel):
    """Risk assessment request model."""
    portfolio: Portfolio
    market_data: MarketData
    risk_parameters: RiskParameters
    assessment_type: str = Field(default="comprehensive", description="Type of risk assessment")
    include_stress_testing: bool = Field(default=True, description="Include stress test scenarios")
    use_validation: bool = Field(default=True, description="Use secondary AI for validation")


class RiskBreakdown(BaseModel):
    """Detailed risk breakdown by category."""
    concentration_risk: float = Field(description="Concentration risk score (0-100)")
    correlation_risk: float = Field(description="Correlation risk score (0-100)")
    liquidity_risk: float = Field(description="Liquidity risk score (0-100)")
    volatility_risk: float = Field(description="Volatility risk score (0-100)")
    leverage_risk: float = Field(description="Leverage risk score (0-100)")
    
    @validator("*")
    def validate_risk_scores(cls, v):
        """Validate risk scores are between 0 and 100."""
        if not 0 <= v <= 100:
            raise ValueError("Risk scores must be between 0 and 100")
        return v


class VaRAnalysis(BaseModel):
    """Value at Risk analysis."""
    daily_var: float = Field(description="Daily VaR at specified confidence level")
    weekly_var: float = Field(description="Weekly VaR")
    monthly_var: float = Field(description="Monthly VaR")
    expected_shortfall: float = Field(description="Expected Shortfall (Conditional VaR)")
    var_breakdown: Dict[str, float] = Field(description="VaR contribution by position")


class StressTestScenario(BaseModel):
    """Stress test scenario results."""
    scenario_name: str = Field(description="Scenario description")
    portfolio_impact: float = Field(description="Portfolio impact (%)")
    worst_position: str = Field(description="Position with worst impact")
    recovery_time_estimate: str = Field(description="Estimated recovery time")


class RiskMitigationRecommendation(BaseModel):
    """Risk mitigation recommendation."""
    risk_category: str = Field(description="Category of risk being addressed")
    recommendation: str = Field(description="Specific recommendation")
    priority: str = Field(description="Priority level (high/medium/low)")
    expected_impact: float = Field(description="Expected risk reduction (%)")
    implementation_complexity: str = Field(description="Implementation complexity")


class RiskAssessmentResponse(BaseModel):
    """Risk assessment response model."""
    portfolio_value: float
    assessment_type: str
    
    # Overall risk metrics
    overall_risk_score: int = Field(description="Overall risk score (1-100)")
    risk_rating: str = Field(description="Risk rating (Low/Medium/High/Extreme)")
    
    # Detailed risk breakdown
    risk_breakdown: RiskBreakdown
    
    # Value at Risk analysis
    var_analysis: VaRAnalysis
    
    # Concentration analysis
    concentration_analysis: Dict[str, Any] = Field(description="Position concentration metrics")
    
    # Stress test results
    stress_test_scenarios: List[StressTestScenario] = Field(default_factory=list)
    
    # Risk mitigation recommendations
    risk_mitigation_recommendations: List[RiskMitigationRecommendation]
    
    # Position sizing suggestions
    position_sizing_suggestions: Dict[str, Any] = Field(description="Optimal position sizes")
    
    # Monitoring alerts
    monitoring_alerts: List[str] = Field(description="Risk thresholds to monitor")
    
    # Validation results (if enabled)
    validation: Optional[Dict[str, Any]] = None
    risk_consensus: Optional[int] = None
    
    # Metadata
    assessed_at: datetime
    model_used: str
    processing_time_ms: int


# Dependency injection
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


@router.post("/assess", response_model=RiskAssessmentResponse)
async def assess_portfolio_risk(
    request: RiskAssessmentRequest,
    background_tasks: BackgroundTasks,
    cache_manager: CacheManager = Depends(get_cache_manager),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    metrics: MetricsCollector = Depends(get_metrics_collector),
    deepseek_client: DeepSeekClient = Depends(get_deepseek_client),
    gemini_client: GeminiClient = Depends(get_gemini_client)
):
    """
    Perform comprehensive portfolio risk assessment.
    
    Analyzes portfolio positions, market conditions, and risk parameters
    to provide detailed risk metrics and mitigation recommendations.
    """
    start_time = datetime.utcnow()
    
    try:
        # Apply rate limiting
        await rate_limiter.wait_for_service("risk_assessment")
        
        # Generate cache key
        portfolio_hash = generate_hash(request.portfolio.dict())
        market_hash = generate_hash(request.market_data.dict())
        cache_key = CacheKey.risk_assessment(portfolio_hash, market_hash)
        
        # Check cache
        cached_result = await cache_manager.get(cache_key)
        if cached_result:
            logger.info(
                "Risk assessment served from cache",
                portfolio_value=request.portfolio.total_value,
                cache_key=str(cache_key)
            )
            
            cached_result["processing_time_ms"] = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )
            return RiskAssessmentResponse(**cached_result)
        
        # Perform primary risk assessment with DeepSeek
        with AIRequestTracker(
            metrics, "deepseek", "deepseek-chat", "risk_assessment"
        ) as tracker:
            risk_analysis = await deepseek_client.risk_assessment(
                portfolio=request.portfolio.dict(),
                market_data=request.market_data.dict(),
                risk_params=request.risk_parameters.dict()
            )
            
            if "usage" in risk_analysis:
                tracker.set_tokens_used(risk_analysis["usage"].get("total_tokens", 0))
        
        # Perform validation with Gemini if requested
        validation_result = None
        risk_consensus = None
        
        if request.use_validation:
            try:
                # Prepare validation data (simplified for secondary analysis)
                validation_portfolio = {
                    "total_value": request.portfolio.total_value,
                    "positions_count": len(request.portfolio.positions),
                    "leverage_ratio": request.portfolio.leverage_ratio,
                    "top_positions": [
                        {"symbol": pos.symbol, "weight": pos.position_weight}
                        for pos in sorted(request.portfolio.positions, 
                                        key=lambda x: x.position_weight, reverse=True)[:5]
                    ]
                }
                
                with AIRequestTracker(
                    metrics, "gemini", "gemini-1.5-flash", "risk_validation"
                ) as tracker:
                    validation_result = await gemini_client.risk_assessment(
                        portfolio=validation_portfolio,
                        market_data=request.market_data.dict(),
                        risk_params=request.risk_parameters.dict()
                    )
                    risk_consensus = validation_result.get("overall_risk_score", 50)
                    
            except Exception as e:
                logger.warning(
                    "Risk assessment validation failed",
                    portfolio_value=request.portfolio.total_value,
                    error=str(e)
                )
                validation_result = {"error": str(e)}
                risk_consensus = 50
        
        # Parse risk analysis results
        overall_risk_score = risk_analysis.get("overall_risk_score", 50)
        
        # Determine risk rating
        if overall_risk_score <= 25:
            risk_rating = "Low"
        elif overall_risk_score <= 50:
            risk_rating = "Medium" 
        elif overall_risk_score <= 75:
            risk_rating = "High"
        else:
            risk_rating = "Extreme"
        
        # Parse risk breakdown
        risk_breakdown_data = risk_analysis.get("risk_breakdown", {})
        risk_breakdown = RiskBreakdown(
            concentration_risk=risk_breakdown_data.get("concentration_risk", 30),
            correlation_risk=risk_breakdown_data.get("correlation_risk", 40),
            liquidity_risk=risk_breakdown_data.get("liquidity_risk", 20),
            volatility_risk=risk_breakdown_data.get("volatility_risk", 35),
            leverage_risk=risk_breakdown_data.get("leverage_risk", 25)
        )
        
        # Parse VaR analysis
        var_data = risk_analysis.get("var_analysis", {})
        var_analysis = VaRAnalysis(
            daily_var=var_data.get("daily_var", request.portfolio.total_value * 0.02),
            weekly_var=var_data.get("weekly_var", request.portfolio.total_value * 0.05),
            monthly_var=var_data.get("monthly_var", request.portfolio.total_value * 0.10),
            expected_shortfall=var_data.get("expected_shortfall", request.portfolio.total_value * 0.03),
            var_breakdown=var_data.get("var_breakdown", {})
        )
        
        # Parse stress test scenarios
        stress_scenarios = []
        for scenario_data in risk_analysis.get("stress_test_scenarios", []):
            if isinstance(scenario_data, dict):
                stress_scenarios.append(StressTestScenario(
                    scenario_name=scenario_data.get("scenario_name", "Market Crash"),
                    portfolio_impact=scenario_data.get("portfolio_impact", -15.0),
                    worst_position=scenario_data.get("worst_position", "Unknown"),
                    recovery_time_estimate=scenario_data.get("recovery_time_estimate", "3-6 months")
                ))
        
        # Parse risk mitigation recommendations
        recommendations = []
        for rec_data in risk_analysis.get("risk_mitigation_recommendations", []):
            if isinstance(rec_data, dict):
                recommendations.append(RiskMitigationRecommendation(
                    risk_category=rec_data.get("risk_category", "General"),
                    recommendation=rec_data.get("recommendation", "Review portfolio allocation"),
                    priority=rec_data.get("priority", "medium"),
                    expected_impact=rec_data.get("expected_impact", 10.0),
                    implementation_complexity=rec_data.get("implementation_complexity", "medium")
                ))
        
        # Calculate processing time
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Prepare response
        response_data = {
            "portfolio_value": request.portfolio.total_value,
            "assessment_type": request.assessment_type,
            "overall_risk_score": overall_risk_score,
            "risk_rating": risk_rating,
            "risk_breakdown": risk_breakdown.dict(),
            "var_analysis": var_analysis.dict(),
            "concentration_analysis": risk_analysis.get("concentration_analysis", {}),
            "stress_test_scenarios": [s.dict() for s in stress_scenarios],
            "risk_mitigation_recommendations": [r.dict() for r in recommendations],
            "position_sizing_suggestions": risk_analysis.get("position_sizing_suggestions", {}),
            "monitoring_alerts": risk_analysis.get("monitoring_alerts", [
                f"Portfolio risk score above {overall_risk_score}",
                "Monitor correlation changes",
                "Watch for liquidity deterioration"
            ]),
            "validation": validation_result,
            "risk_consensus": risk_consensus,
            "assessed_at": start_time,
            "model_used": "deepseek-chat", 
            "processing_time_ms": processing_time
        }
        
        # Cache the result
        background_tasks.add_task(
            cache_manager.set,
            cache_key,
            response_data,
            settings.RISK_ANALYSIS_CACHE_TTL
        )
        
        # Record business metrics
        metrics.record_analysis_accuracy(
            analysis_type="risk_assessment",
            accuracy_score=100 - overall_risk_score  # Higher accuracy for lower risk
        )
        
        logger.info(
            "Risk assessment completed",
            portfolio_value=request.portfolio.total_value,
            positions_count=len(request.portfolio.positions),
            overall_risk_score=overall_risk_score,
            processing_time_ms=processing_time
        )
        
        return RiskAssessmentResponse(**response_data)
        
    except Exception as e:
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        logger.error(
            "Risk assessment failed",
            portfolio_value=request.portfolio.total_value,
            processing_time_ms=processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Risk assessment failed: {str(e)}"
        )


@router.post("/position-sizing")
async def calculate_optimal_position_sizing(
    portfolio_value: float,
    target_positions: List[str],
    risk_tolerance: str = "medium",  # low, medium, high
    max_correlation: float = 0.7,
    cache_manager: CacheManager = Depends(get_cache_manager),
    deepseek_client: DeepSeekClient = Depends(get_deepseek_client)
):
    """
    Calculate optimal position sizing for new trades.
    
    Uses AI analysis to determine appropriate position sizes based on
    portfolio constraints, risk tolerance, and correlation limits.
    """
    start_time = datetime.utcnow()
    
    try:
        # Generate cache key
        cache_key = f"position_sizing:v1:{generate_hash({'portfolio_value': portfolio_value, 'positions': target_positions, 'risk_tolerance': risk_tolerance})}"
        
        # Check cache
        cached_result = await cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        # Risk tolerance mappings
        risk_mappings = {
            "low": {"max_position_pct": 5.0, "max_portfolio_risk": 1.0},
            "medium": {"max_position_pct": 10.0, "max_portfolio_risk": 2.0}, 
            "high": {"max_position_pct": 20.0, "max_portfolio_risk": 5.0}
        }
        
        risk_params = risk_mappings.get(risk_tolerance, risk_mappings["medium"])
        
        # Prepare analysis data
        sizing_data = {
            "portfolio_value": portfolio_value,
            "target_positions": target_positions,
            "risk_tolerance": risk_tolerance,
            "max_position_size_pct": risk_params["max_position_pct"],
            "max_portfolio_risk_pct": risk_params["max_portfolio_risk"],
            "max_correlation": max_correlation
        }
        
        # AI-powered position sizing analysis
        system_prompt = """You are an expert portfolio manager specializing in position sizing and risk management.
        
        Given the portfolio parameters and target positions, calculate optimal position sizes that:
        1. Maximize risk-adjusted returns
        2. Stay within risk tolerance limits
        3. Maintain appropriate diversification
        4. Consider correlation constraints
        
        Provide structured JSON with position sizes and risk metrics."""
        
        messages = [
            {
                "role": "user",
                "content": f"Calculate optimal position sizing:\n\n{sizing_data}\n\nProvide detailed position sizing recommendations with risk analysis."
            }
        ]
        
        response = await deepseek_client.chat_completion(
            messages=messages,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=3000
        )
        
        content = response.get_content()
        
        # Parse response
        try:
            import json
            sizing_result = json.loads(content)
        except json.JSONDecodeError:
            sizing_result = {
                "raw_analysis": content,
                "position_sizes": {},
                "total_allocation": 0
            }
        
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        result = {
            "portfolio_value": portfolio_value,
            "risk_tolerance": risk_tolerance,
            "position_sizing_result": sizing_result,
            "risk_parameters": risk_params,
            "calculated_at": start_time.isoformat(),
            "processing_time_ms": processing_time
        }
        
        # Cache result
        await cache_manager.set(cache_key, result, 1800)  # 30 minutes cache
        
        logger.info(
            "Position sizing calculation completed",
            portfolio_value=portfolio_value,
            positions_count=len(target_positions),
            processing_time_ms=processing_time
        )
        
        return result
        
    except Exception as e:
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        logger.error(
            "Position sizing calculation failed",
            portfolio_value=portfolio_value,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Position sizing calculation failed: {str(e)}"
        )


@router.get("/risk-metrics/definitions")
async def get_risk_metrics_definitions():
    """Get definitions and explanations of risk metrics."""
    return {
        "risk_metrics": {
            "overall_risk_score": {
                "description": "Comprehensive risk score (1-100)",
                "interpretation": {
                    "1-25": "Low Risk - Conservative portfolio with limited downside",
                    "26-50": "Medium Risk - Balanced risk-reward profile",
                    "51-75": "High Risk - Aggressive portfolio with significant volatility",
                    "76-100": "Extreme Risk - Very high potential losses"
                }
            },
            "concentration_risk": {
                "description": "Risk from overconcentration in single positions",
                "mitigation": "Diversify across positions and sectors"
            },
            "correlation_risk": {
                "description": "Risk from high correlation between positions",
                "mitigation": "Include uncorrelated or negatively correlated assets"
            },
            "liquidity_risk": {
                "description": "Risk of inability to exit positions quickly",
                "mitigation": "Maintain adequate cash reserves and liquid positions"
            },
            "volatility_risk": {
                "description": "Risk from price volatility and market fluctuations",
                "mitigation": "Use position sizing and volatility targeting"
            },
            "leverage_risk": {
                "description": "Additional risk from leverage and margin usage",
                "mitigation": "Limit leverage and maintain margin buffers"
            }
        },
        "var_metrics": {
            "daily_var": "Maximum expected loss in 1 day at given confidence level",
            "weekly_var": "Maximum expected loss in 1 week",
            "monthly_var": "Maximum expected loss in 1 month",
            "expected_shortfall": "Average loss when VaR threshold is exceeded"
        },
        "position_sizing_guidelines": {
            "conservative": "1-5% per position, max 20% sector exposure",
            "moderate": "2-10% per position, max 30% sector exposure", 
            "aggressive": "5-20% per position, max 50% sector exposure"
        }
    }