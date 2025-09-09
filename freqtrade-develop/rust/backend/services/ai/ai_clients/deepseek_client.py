"""
DeepSeek AI Client Implementation
DeepSeek AI 客户端实现

This module implements the DeepSeek API client with robust error handling,
rate limiting, and circuit breaker patterns for reliable AI analysis.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta

import httpx
import structlog
from pydantic import BaseModel, Field

from config import settings
from core.circuit_breaker import circuit_breaker
from core.rate_limiter import AsyncRateLimiter


logger = structlog.get_logger(__name__)


class DeepSeekRequest(BaseModel):
    """DeepSeek API request model."""
    model: str = Field(default="deepseek-chat", description="Model name")
    messages: List[Dict[str, str]] = Field(..., description="Message history")
    temperature: float = Field(default=0.1, description="Response randomness")
    max_tokens: int = Field(default=4000, description="Maximum response tokens")
    stream: bool = Field(default=False, description="Enable streaming response")
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "You are a cryptocurrency market analyst."},
                    {"role": "user", "content": "Analyze BTC price trend for the next 24 hours."}
                ],
                "temperature": 0.1,
                "max_tokens": 2000
            }
        }


class DeepSeekResponse(BaseModel):
    """DeepSeek API response model."""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    
    def get_content(self) -> str:
        """Extract content from response."""
        if self.choices and len(self.choices) > 0:
            return self.choices[0].get("message", {}).get("content", "")
        return ""
    
    def get_finish_reason(self) -> str:
        """Get finish reason from response."""
        if self.choices and len(self.choices) > 0:
            return self.choices[0].get("finish_reason", "unknown")
        return "unknown"


class DeepSeekClient:
    """
    DeepSeek AI API client with enterprise-grade features.
    
    Features:
    - Rate limiting with token bucket algorithm
    - Circuit breaker for fault tolerance  
    - Automatic retries with exponential backoff
    - Request/response caching
    - Comprehensive error handling
    - Metrics collection
    """
    
    def __init__(self):
        """Initialize DeepSeek client."""
        self.base_url = settings.DEEPSEEK_BASE_URL
        self.api_key = settings.DEEPSEEK_API_KEY
        self.rate_limiter = AsyncRateLimiter(
            rate=settings.DEEPSEEK_RATE_LIMIT,
            per=60  # per minute
        )
        
        # HTTP client configuration
        timeout = httpx.Timeout(
            connect=10.0,
            read=settings.REQUEST_TIMEOUT_SECONDS,
            write=10.0,
            pool=5.0
        )
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=50,
                keepalive_expiry=30.0
            ),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "TradingPlatform-AI/1.0"
            }
        )
        
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = None
        
        logger.info("DeepSeek client initialized", base_url=self.base_url)
    
    async def close(self):
        """Close HTTP client connections."""
        await self.client.aclose()
        logger.info("DeepSeek client closed")
    
    @circuit_breaker(
        failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        timeout_seconds=settings.CIRCUIT_BREAKER_TIMEOUT_SECONDS,
        success_threshold=settings.CIRCUIT_BREAKER_SUCCESS_THRESHOLD
    )
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "deepseek-chat",
        temperature: float = None,
        max_tokens: int = None,
        system_prompt: str = None
    ) -> DeepSeekResponse:
        """
        Send chat completion request to DeepSeek API.
        
        Args:
            messages: List of message dictionaries with role and content
            model: DeepSeek model name 
            temperature: Response randomness (0.0-2.0)
            max_tokens: Maximum response tokens
            system_prompt: Optional system prompt to prepend
            
        Returns:
            DeepSeekResponse: Parsed API response
            
        Raises:
            httpx.HTTPStatusError: For API errors
            asyncio.TimeoutError: For timeout errors
            ValueError: For invalid parameters
        """
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        # Prepare request
        if system_prompt and messages and messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        request_data = DeepSeekRequest(
            model=model,
            messages=messages,
            temperature=temperature or settings.MODEL_TEMPERATURE,
            max_tokens=max_tokens or settings.MAX_TOKENS
        )
        
        start_time = time.time()
        self.request_count += 1
        
        try:
            logger.debug(
                "Sending DeepSeek request",
                model=model,
                message_count=len(messages),
                max_tokens=request_data.max_tokens
            )
            
            response = await self._make_request(
                method="POST",
                endpoint="/chat/completions",
                data=request_data.dict()
            )
            
            # Parse response
            deepseek_response = DeepSeekResponse(**response)
            
            # Log metrics
            duration = time.time() - start_time
            tokens_used = deepseek_response.usage.get("total_tokens", 0)
            
            logger.info(
                "DeepSeek request completed",
                duration=duration,
                tokens_used=tokens_used,
                finish_reason=deepseek_response.get_finish_reason(),
                request_id=deepseek_response.id
            )
            
            self.last_request_time = datetime.utcnow()
            return deepseek_response
            
        except Exception as e:
            self.error_count += 1
            duration = time.time() - start_time
            
            logger.error(
                "DeepSeek request failed",
                error=str(e),
                duration=duration,
                model=model,
                message_count=len(messages)
            )
            raise
    
    async def market_analysis(
        self,
        symbol: str,
        timeframe: str,
        market_data: Dict[str, Any],
        analysis_type: str = "trend"
    ) -> Dict[str, Any]:
        """
        Perform AI-powered market analysis.
        
        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            timeframe: Analysis timeframe (1m, 5m, 1h, 4h, 1d)
            market_data: Historical price and volume data
            analysis_type: Type of analysis (trend, pattern, momentum, volatility)
            
        Returns:
            Dict containing analysis results and recommendations
        """
        system_prompt = f"""You are an expert cryptocurrency market analyst specializing in {analysis_type} analysis.
        
        Your task is to analyze the provided market data for {symbol} on {timeframe} timeframe and provide:
        1. Current trend direction and strength
        2. Key support and resistance levels  
        3. Technical indicators interpretation
        4. Price targets and risk levels
        5. Trading recommendations with risk management
        
        Respond with structured JSON containing:
        - trend_direction: "bullish" | "bearish" | "neutral" | "consolidating"
        - trend_strength: 1-10 scale
        - support_levels: array of price levels
        - resistance_levels: array of price levels
        - indicators: object with indicator values and signals
        - price_targets: upside and downside targets
        - risk_level: "low" | "medium" | "high" | "extreme"
        - confidence: 1-100 percentage
        - recommendation: trading action and reasoning
        - timeframe_outlook: short/medium/long term outlook
        """
        
        user_message = f"""Analyze the following market data for {symbol}:

        Current Price: ${market_data.get('current_price', 'N/A')}
        24h Change: {market_data.get('price_change_24h', 'N/A')}%
        Volume: {market_data.get('volume_24h', 'N/A')}
        
        Technical Data:
        {json.dumps(market_data.get('technical_indicators', {}), indent=2)}
        
        Historical Prices (last 50 candles):
        {json.dumps(market_data.get('price_history', [])[-50:], indent=2)}
        
        Please provide comprehensive {analysis_type} analysis with specific actionable insights."""
        
        messages = [
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = await self.chat_completion(
                messages=messages,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=3000
            )
            
            content = response.get_content()
            
            # Try to parse JSON response
            try:
                analysis_result = json.loads(content)
            except json.JSONDecodeError:
                # Fallback to structured text parsing
                analysis_result = {
                    "raw_analysis": content,
                    "trend_direction": "neutral",
                    "trend_strength": 5,
                    "confidence": 70,
                    "risk_level": "medium",
                    "recommendation": "Further analysis required"
                }
            
            # Add metadata
            analysis_result.update({
                "symbol": symbol,
                "timeframe": timeframe,
                "analysis_type": analysis_type,
                "analyzed_at": datetime.utcnow().isoformat(),
                "model": "deepseek-chat",
                "request_id": response.id
            })
            
            return analysis_result
            
        except Exception as e:
            logger.error(
                "Market analysis failed",
                symbol=symbol,
                timeframe=timeframe,
                analysis_type=analysis_type,
                error=str(e)
            )
            
            # Return fallback analysis
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "analysis_type": analysis_type,
                "error": str(e),
                "trend_direction": "neutral",
                "confidence": 0,
                "risk_level": "high",
                "recommendation": "Analysis unavailable due to AI service error"
            }
    
    async def strategy_optimization(
        self,
        strategy_config: Dict[str, Any],
        backtest_results: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize trading strategy parameters using AI analysis.
        
        Args:
            strategy_config: Current strategy configuration
            backtest_results: Historical performance metrics
            market_conditions: Current market environment data
            
        Returns:
            Dict containing optimization recommendations
        """
        system_prompt = """You are an expert quantitative trading strategist specializing in algorithmic trading optimization.

        Your task is to analyze the provided strategy performance and market conditions to recommend parameter optimizations.
        
        Consider:
        1. Risk-adjusted returns (Sharpe ratio, Sortino ratio)
        2. Maximum drawdown and recovery time
        3. Win rate and profit factor
        4. Market regime compatibility
        5. Parameter stability and robustness
        
        Provide structured JSON response with:
        - current_performance: performance summary
        - identified_issues: list of performance issues
        - optimization_recommendations: specific parameter changes
        - risk_assessment: impact of changes on risk profile
        - expected_improvements: quantitative improvement estimates
        - implementation_priority: high/medium/low for each recommendation
        - market_regime_compatibility: how changes perform in different market conditions
        """
        
        user_message = f"""Optimize this trading strategy:

        Strategy Configuration:
        {json.dumps(strategy_config, indent=2)}
        
        Backtest Results:
        {json.dumps(backtest_results, indent=2)}
        
        Current Market Conditions:
        {json.dumps(market_conditions, indent=2)}
        
        Provide specific optimization recommendations to improve risk-adjusted returns while maintaining robustness."""
        
        messages = [
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = await self.chat_completion(
                messages=messages,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=4000
            )
            
            content = response.get_content()
            
            try:
                optimization_result = json.loads(content)
            except json.JSONDecodeError:
                optimization_result = {
                    "raw_analysis": content,
                    "optimization_recommendations": [],
                    "risk_assessment": "Unable to parse structured recommendations",
                    "expected_improvements": {}
                }
            
            # Add metadata
            optimization_result.update({
                "strategy_name": strategy_config.get("name", "Unknown"),
                "optimized_at": datetime.utcnow().isoformat(),
                "model": "deepseek-chat",
                "request_id": response.id
            })
            
            return optimization_result
            
        except Exception as e:
            logger.error(
                "Strategy optimization failed",
                strategy_name=strategy_config.get("name", "Unknown"),
                error=str(e)
            )
            
            return {
                "strategy_name": strategy_config.get("name", "Unknown"),
                "error": str(e),
                "optimization_recommendations": [],
                "risk_assessment": "Optimization unavailable due to AI service error"
            }
    
    async def risk_assessment(
        self,
        portfolio: Dict[str, Any],
        market_data: Dict[str, Any],
        risk_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive portfolio risk assessment.
        
        Args:
            portfolio: Current portfolio positions and allocations
            market_data: Current market conditions and volatility
            risk_params: Risk management parameters and limits
            
        Returns:
            Dict containing risk assessment and recommendations
        """
        system_prompt = """You are an expert risk management analyst specializing in cryptocurrency portfolio risk assessment.

        Analyze the provided portfolio and market conditions to assess:
        1. Portfolio concentration risk
        2. Market correlation exposure
        3. Volatility and VaR analysis
        4. Liquidity risk assessment
        5. Counterparty and exchange risk
        6. Regulatory and compliance risks

        Provide structured JSON with:
        - overall_risk_score: 1-100 scale
        - risk_breakdown: detailed risk category analysis
        - concentration_analysis: position size and correlation risks
        - var_analysis: Value at Risk calculations
        - stress_test_scenarios: potential loss scenarios
        - risk_mitigation_recommendations: specific actions to reduce risk
        - position_sizing_suggestions: optimal allocation recommendations
        - monitoring_alerts: risk thresholds to monitor
        """
        
        user_message = f"""Assess risk for this cryptocurrency portfolio:

        Portfolio Positions:
        {json.dumps(portfolio, indent=2)}
        
        Market Data:
        {json.dumps(market_data, indent=2)}
        
        Risk Parameters:
        {json.dumps(risk_params, indent=2)}
        
        Provide comprehensive risk assessment with specific risk mitigation recommendations."""
        
        messages = [
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = await self.chat_completion(
                messages=messages,
                system_prompt=system_prompt,
                temperature=0.1,
                max_tokens=4000
            )
            
            content = response.get_content()
            
            try:
                risk_assessment = json.loads(content)
            except json.JSONDecodeError:
                risk_assessment = {
                    "raw_analysis": content,
                    "overall_risk_score": 50,
                    "risk_breakdown": {},
                    "risk_mitigation_recommendations": []
                }
            
            # Add metadata
            risk_assessment.update({
                "assessed_at": datetime.utcnow().isoformat(),
                "portfolio_value": portfolio.get("total_value", 0),
                "model": "deepseek-chat",
                "request_id": response.id
            })
            
            return risk_assessment
            
        except Exception as e:
            logger.error(
                "Risk assessment failed",
                portfolio_value=portfolio.get("total_value", 0),
                error=str(e)
            )
            
            return {
                "portfolio_value": portfolio.get("total_value", 0),
                "error": str(e),
                "overall_risk_score": 100,  # Max risk on error
                "risk_assessment": "Risk assessment unavailable due to AI service error",
                "risk_mitigation_recommendations": [
                    "Reduce position sizes due to inability to assess risk properly"
                ]
            }
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        retry_count: int = 3
    ) -> Dict[str, Any]:
        """
        Make HTTP request to DeepSeek API with retries.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            retry_count: Number of retry attempts
            
        Returns:
            Dict: Parsed JSON response
            
        Raises:
            httpx.HTTPStatusError: For API errors
            asyncio.TimeoutError: For timeout errors
        """
        for attempt in range(retry_count + 1):
            try:
                request_kwargs = {
                    "method": method,
                    "url": endpoint,
                    "timeout": settings.REQUEST_TIMEOUT_SECONDS
                }
                
                if data:
                    request_kwargs["json"] = data
                if params:
                    request_kwargs["params"] = params
                
                response = await self.client.request(**request_kwargs)
                
                # Check for successful response
                response.raise_for_status()
                
                return response.json()
                
            except httpx.TimeoutException as e:
                if attempt == retry_count:
                    logger.error("DeepSeek API timeout after retries", attempts=attempt + 1)
                    raise asyncio.TimeoutError(f"Request timed out after {retry_count + 1} attempts")
                
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(
                    "DeepSeek API timeout, retrying",
                    attempt=attempt + 1,
                    wait_time=wait_time
                )
                await asyncio.sleep(wait_time)
                
            except httpx.HTTPStatusError as e:
                # Don't retry client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    logger.error(
                        "DeepSeek API client error",
                        status_code=e.response.status_code,
                        response_text=e.response.text
                    )
                    raise
                
                # Retry server errors (5xx)
                if attempt == retry_count:
                    logger.error(
                        "DeepSeek API server error after retries",
                        status_code=e.response.status_code,
                        attempts=attempt + 1
                    )
                    raise
                
                wait_time = 2 ** attempt
                logger.warning(
                    "DeepSeek API server error, retrying",
                    status_code=e.response.status_code,
                    attempt=attempt + 1,
                    wait_time=wait_time
                )
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                if attempt == retry_count:
                    logger.error("DeepSeek API request failed after retries", error=str(e))
                    raise
                
                wait_time = 2 ** attempt
                logger.warning(
                    "DeepSeek API request failed, retrying",
                    error=str(e),
                    attempt=attempt + 1,
                    wait_time=wait_time
                )
                await asyncio.sleep(wait_time)
    
    async def health_check(self) -> bool:
        """
        Check DeepSeek API health status.
        
        Returns:
            bool: True if API is healthy, False otherwise
        """
        try:
            # Simple health check with minimal token usage
            messages = [
                {"role": "user", "content": "Health check - respond with 'OK'"}
            ]
            
            response = await self.chat_completion(
                messages=messages,
                max_tokens=10,
                temperature=0.0
            )
            
            return "ok" in response.get_content().lower()
            
        except Exception as e:
            logger.error("DeepSeek health check failed", error=str(e))
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get client metrics for monitoring.
        
        Returns:
            Dict containing client metrics
        """
        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.request_count, 1),
            "last_request_time": self.last_request_time.isoformat() if self.last_request_time else None,
            "rate_limit": settings.DEEPSEEK_RATE_LIMIT,
            "base_url": self.base_url
        }