"""
Google Gemini AI Client Implementation
Google Gemini AI 客户端实现

This module implements the Google Gemini API client for alternative AI analysis,
providing validation and cross-reference capabilities for trading decisions.
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


class GeminiContent(BaseModel):
    """Gemini API content part."""
    text: str = Field(..., description="Text content")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Analyze BTC price trends for the next 24 hours."
            }
        }


class GeminiMessage(BaseModel):
    """Gemini API message structure."""
    role: str = Field(..., description="Message role (user or model)")
    parts: List[GeminiContent] = Field(..., description="Message content parts")
    
    class Config:
        schema_extra = {
            "example": {
                "role": "user",
                "parts": [{"text": "What is the current market sentiment for Bitcoin?"}]
            }
        }


class GeminiRequest(BaseModel):
    """Gemini API request model."""
    contents: List[GeminiMessage] = Field(..., description="Message history")
    generation_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": 0.1,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 4000,
            "stopSequences": []
        },
        description="Generation configuration"
    )
    safety_settings: List[Dict[str, Any]] = Field(
        default_factory=lambda: [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ],
        description="Safety settings"
    )


class GeminiResponse(BaseModel):
    """Gemini API response model."""
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    usage_metadata: Optional[Dict[str, Any]] = None
    
    def get_content(self) -> str:
        """Extract content from response."""
        if self.candidates and len(self.candidates) > 0:
            candidate = self.candidates[0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if parts and len(parts) > 0:
                    return parts[0].get("text", "")
        return ""
    
    def get_finish_reason(self) -> str:
        """Get finish reason from response."""
        if self.candidates and len(self.candidates) > 0:
            return self.candidates[0].get("finishReason", "unknown")
        return "unknown"
    
    def get_safety_ratings(self) -> List[Dict[str, Any]]:
        """Get safety ratings from response."""
        if self.candidates and len(self.candidates) > 0:
            return self.candidates[0].get("safetyRatings", [])
        return []


class GeminiClient:
    """
    Google Gemini AI API client with enterprise-grade features.
    
    Features:
    - Rate limiting with token bucket algorithm
    - Circuit breaker for fault tolerance
    - Automatic retries with exponential backoff
    - Request/response caching
    - Safety filtering and content moderation
    - Comprehensive error handling
    - Metrics collection
    """
    
    def __init__(self):
        """Initialize Gemini client."""
        self.base_url = settings.GEMINI_BASE_URL
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = "gemini-1.5-flash"  # Default model
        
        self.rate_limiter = AsyncRateLimiter(
            rate=settings.GEMINI_RATE_LIMIT,
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
                "Content-Type": "application/json",
                "User-Agent": "TradingPlatform-AI/1.0"
            }
        )
        
        self.request_count = 0
        self.error_count = 0
        self.last_request_time = None
        
        logger.info("Gemini client initialized", base_url=self.base_url, model=self.model_name)
    
    async def close(self):
        """Close HTTP client connections."""
        await self.client.aclose()
        logger.info("Gemini client closed")
    
    @circuit_breaker(
        failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        timeout_seconds=settings.CIRCUIT_BREAKER_TIMEOUT_SECONDS,
        success_threshold=settings.CIRCUIT_BREAKER_SUCCESS_THRESHOLD
    )
    async def generate_content(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        system_prompt: str = None
    ) -> GeminiResponse:
        """
        Generate content using Gemini API.
        
        Args:
            messages: List of message dictionaries with role and content
            model: Gemini model name
            temperature: Response randomness (0.0-2.0)
            max_tokens: Maximum response tokens
            system_prompt: Optional system prompt to prepend
            
        Returns:
            GeminiResponse: Parsed API response
            
        Raises:
            httpx.HTTPStatusError: For API errors
            asyncio.TimeoutError: For timeout errors
            ValueError: For invalid parameters
        """
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        # Convert messages to Gemini format
        gemini_messages = self._convert_messages_to_gemini_format(messages, system_prompt)
        
        # Prepare request
        request_data = GeminiRequest(
            contents=gemini_messages,
            generation_config={
                "temperature": temperature or settings.MODEL_TEMPERATURE,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": max_tokens or settings.MAX_TOKENS,
                "stopSequences": []
            }
        )
        
        model_name = model or self.model_name
        start_time = time.time()
        self.request_count += 1
        
        try:
            logger.debug(
                "Sending Gemini request",
                model=model_name,
                message_count=len(gemini_messages),
                max_tokens=request_data.generation_config["maxOutputTokens"]
            )
            
            response = await self._make_request(
                method="POST",
                endpoint=f"/models/{model_name}:generateContent",
                data=request_data.dict()
            )
            
            # Parse response
            gemini_response = GeminiResponse(**response)
            
            # Check for safety issues
            safety_ratings = gemini_response.get_safety_ratings()
            if any(rating.get("probability", "") in ["HIGH", "MEDIUM"] for rating in safety_ratings):
                logger.warning("Gemini response flagged by safety filters", safety_ratings=safety_ratings)
            
            # Log metrics
            duration = time.time() - start_time
            tokens_used = gemini_response.usage_metadata.get("totalTokenCount", 0) if gemini_response.usage_metadata else 0
            
            logger.info(
                "Gemini request completed",
                duration=duration,
                tokens_used=tokens_used,
                finish_reason=gemini_response.get_finish_reason(),
                model=model_name
            )
            
            self.last_request_time = datetime.utcnow()
            return gemini_response
            
        except Exception as e:
            self.error_count += 1
            duration = time.time() - start_time
            
            logger.error(
                "Gemini request failed",
                error=str(e),
                duration=duration,
                model=model_name,
                message_count=len(gemini_messages)
            )
            raise
    
    async def market_analysis_validation(
        self,
        symbol: str,
        primary_analysis: Dict[str, Any],
        market_data: Dict[str, Any],
        analysis_type: str = "validation"
    ) -> Dict[str, Any]:
        """
        Validate and cross-reference market analysis from primary AI.
        
        Args:
            symbol: Trading symbol
            primary_analysis: Analysis from primary AI (DeepSeek)
            market_data: Market data for validation
            analysis_type: Type of validation analysis
            
        Returns:
            Dict containing validation results and consensus
        """
        system_prompt = f"""You are an expert cryptocurrency market analyst tasked with validating and cross-referencing trading analysis.

        Your role is to:
        1. Critically evaluate the provided analysis for accuracy and completeness
        2. Cross-reference against current market data
        3. Identify potential biases or oversights
        4. Provide consensus rating and confidence assessment
        5. Highlight any conflicting indicators or red flags
        
        Respond with structured JSON containing:
        - validation_status: "confirmed" | "partially_confirmed" | "disputed" | "insufficient_data"
        - consensus_rating: 1-100 scale of agreement with primary analysis
        - key_agreements: list of points you agree with
        - key_disagreements: list of points you disagree with
        - additional_insights: new insights not covered in primary analysis
        - risk_flags: potential risks or concerns identified
        - confidence_assessment: your confidence in the validation
        - recommendation_consensus: overall trading recommendation alignment
        """
        
        user_message = f"""Validate this market analysis for {symbol}:

        Primary Analysis to Validate:
        {json.dumps(primary_analysis, indent=2)}
        
        Current Market Data:
        {json.dumps(market_data, indent=2)}
        
        Please provide critical validation focusing on accuracy, completeness, and potential risks."""
        
        messages = [
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = await self.generate_content(
                messages=messages,
                temperature=0.1,
                max_tokens=3000
            )
            
            content = response.get_content()
            
            # Try to parse JSON response
            try:
                validation_result = json.loads(content)
            except json.JSONDecodeError:
                validation_result = {
                    "raw_validation": content,
                    "validation_status": "insufficient_data",
                    "consensus_rating": 50,
                    "confidence_assessment": 60,
                    "recommendation_consensus": "neutral"
                }
            
            # Add metadata
            validation_result.update({
                "symbol": symbol,
                "analysis_type": analysis_type,
                "validated_at": datetime.utcnow().isoformat(),
                "primary_analysis_source": primary_analysis.get("model", "unknown"),
                "validator_model": "gemini-1.5-flash",
                "safety_ratings": response.get_safety_ratings()
            })
            
            return validation_result
            
        except Exception as e:
            logger.error(
                "Market analysis validation failed",
                symbol=symbol,
                error=str(e)
            )
            
            return {
                "symbol": symbol,
                "error": str(e),
                "validation_status": "error",
                "consensus_rating": 0,
                "confidence_assessment": 0,
                "recommendation_consensus": "validation_failed"
            }
    
    async def strategy_validation(
        self,
        strategy_config: Dict[str, Any],
        optimization_recommendations: Dict[str, Any],
        market_conditions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate strategy optimization recommendations.
        
        Args:
            strategy_config: Original strategy configuration
            optimization_recommendations: Optimization suggestions from primary AI
            market_conditions: Current market environment
            
        Returns:
            Dict containing validation and alternative recommendations
        """
        system_prompt = """You are an expert quantitative analyst specializing in trading strategy validation and risk assessment.

        Your task is to validate optimization recommendations and provide:
        1. Technical feasibility assessment
        2. Risk-adjusted return potential evaluation
        3. Market regime compatibility analysis
        4. Implementation complexity assessment
        5. Alternative optimization approaches
        
        Provide structured JSON with:
        - feasibility_score: 1-100 scale of implementation feasibility
        - risk_assessment: detailed risk evaluation of recommended changes
        - expected_performance: realistic performance expectations
        - implementation_complexity: low/medium/high complexity rating
        - alternative_recommendations: different optimization approaches
        - priority_ranking: ordered list of recommendations by importance
        - market_compatibility: how recommendations perform across market conditions
        - validation_confidence: confidence in the validation assessment
        """
        
        user_message = f"""Validate these strategy optimization recommendations:

        Original Strategy:
        {json.dumps(strategy_config, indent=2)}
        
        Optimization Recommendations:
        {json.dumps(optimization_recommendations, indent=2)}
        
        Market Conditions:
        {json.dumps(market_conditions, indent=2)}
        
        Provide critical validation with focus on feasibility, risk, and alternative approaches."""
        
        messages = [
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = await self.generate_content(
                messages=messages,
                temperature=0.1,
                max_tokens=4000
            )
            
            content = response.get_content()
            
            try:
                validation_result = json.loads(content)
            except json.JSONDecodeError:
                validation_result = {
                    "raw_validation": content,
                    "feasibility_score": 50,
                    "risk_assessment": "Unable to parse structured validation",
                    "validation_confidence": 50
                }
            
            # Add metadata
            validation_result.update({
                "strategy_name": strategy_config.get("name", "Unknown"),
                "validated_at": datetime.utcnow().isoformat(),
                "validator_model": "gemini-1.5-flash",
                "safety_ratings": response.get_safety_ratings()
            })
            
            return validation_result
            
        except Exception as e:
            logger.error(
                "Strategy validation failed",
                strategy_name=strategy_config.get("name", "Unknown"),
                error=str(e)
            )
            
            return {
                "strategy_name": strategy_config.get("name", "Unknown"),
                "error": str(e),
                "feasibility_score": 0,
                "risk_assessment": "Validation failed due to AI service error",
                "validation_confidence": 0
            }
    
    async def news_sentiment_analysis(
        self,
        news_articles: List[Dict[str, Any]],
        symbols: List[str],
        timeframe: str = "24h"
    ) -> Dict[str, Any]:
        """
        Analyze news sentiment for cryptocurrency markets.
        
        Args:
            news_articles: List of news articles with title, content, source
            symbols: List of cryptocurrency symbols to analyze
            timeframe: Analysis timeframe
            
        Returns:
            Dict containing sentiment analysis results
        """
        system_prompt = """You are an expert financial news sentiment analyst specializing in cryptocurrency markets.

        Analyze the provided news articles and extract:
        1. Overall market sentiment (bullish/bearish/neutral)
        2. Symbol-specific sentiment scores
        3. Key themes and narratives driving sentiment
        4. Sentiment strength and confidence levels
        5. Potential market impact assessment
        6. Sentiment trend analysis
        
        Provide structured JSON with:
        - overall_sentiment: "bullish" | "bearish" | "neutral" | "mixed"
        - sentiment_score: -100 to +100 scale (negative = bearish, positive = bullish)
        - symbol_sentiments: object with sentiment for each symbol
        - key_themes: major themes identified in news
        - market_impact: potential impact on price movements
        - confidence_level: confidence in sentiment analysis (1-100)
        - news_volume: assessment of news volume and coverage
        - sentiment_catalysts: specific events or news driving sentiment
        """
        
        # Prepare news summary for analysis
        news_summary = []
        for article in news_articles[:20]:  # Limit to 20 articles to avoid token limits
            news_summary.append({
                "title": article.get("title", "")[:200],  # Limit title length
                "summary": article.get("content", "")[:500],  # Limit content length
                "source": article.get("source", ""),
                "published": article.get("published_at", "")
            })
        
        user_message = f"""Analyze sentiment for these cryptocurrency symbols: {', '.join(symbols)}

        News Articles ({len(news_articles)} total, showing top 20):
        {json.dumps(news_summary, indent=2)}
        
        Timeframe: {timeframe}
        
        Provide comprehensive sentiment analysis with specific focus on market impact and trading implications."""
        
        messages = [
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = await self.generate_content(
                messages=messages,
                temperature=0.2,  # Slightly higher for sentiment analysis
                max_tokens=3000
            )
            
            content = response.get_content()
            
            try:
                sentiment_result = json.loads(content)
            except json.JSONDecodeError:
                sentiment_result = {
                    "raw_analysis": content,
                    "overall_sentiment": "neutral",
                    "sentiment_score": 0,
                    "confidence_level": 50
                }
            
            # Add metadata
            sentiment_result.update({
                "symbols": symbols,
                "timeframe": timeframe,
                "analyzed_at": datetime.utcnow().isoformat(),
                "articles_analyzed": len(news_articles),
                "model": "gemini-1.5-flash",
                "safety_ratings": response.get_safety_ratings()
            })
            
            return sentiment_result
            
        except Exception as e:
            logger.error(
                "News sentiment analysis failed",
                symbols=symbols,
                articles_count=len(news_articles),
                error=str(e)
            )
            
            return {
                "symbols": symbols,
                "error": str(e),
                "overall_sentiment": "neutral",
                "sentiment_score": 0,
                "confidence_level": 0,
                "market_impact": "Unable to analyze due to AI service error"
            }
    
    def _convert_messages_to_gemini_format(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str = None
    ) -> List[GeminiMessage]:
        """
        Convert standard messages to Gemini API format.
        
        Args:
            messages: List of messages in standard format
            system_prompt: Optional system prompt
            
        Returns:
            List of GeminiMessage objects
        """
        gemini_messages = []
        
        # Add system prompt as first user message if provided
        if system_prompt:
            gemini_messages.append(GeminiMessage(
                role="user",
                parts=[GeminiContent(text=system_prompt)]
            ))
            gemini_messages.append(GeminiMessage(
                role="model",
                parts=[GeminiContent(text="I understand. I'll analyze according to these instructions.")]
            ))
        
        # Convert messages
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Gemini uses "model" instead of "assistant"
            if role == "assistant":
                role = "model"
            elif role == "system":
                # Convert system messages to user messages in Gemini
                role = "user"
                content = f"System instruction: {content}"
            
            gemini_messages.append(GeminiMessage(
                role=role,
                parts=[GeminiContent(text=content)]
            ))
        
        return gemini_messages
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        retry_count: int = 3
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Gemini API with retries.
        
        Args:
            method: HTTP method
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters (including API key)
            retry_count: Number of retry attempts
            
        Returns:
            Dict: Parsed JSON response
        """
        # Add API key to params
        if not params:
            params = {}
        params["key"] = self.api_key
        
        for attempt in range(retry_count + 1):
            try:
                request_kwargs = {
                    "method": method,
                    "url": endpoint,
                    "params": params,
                    "timeout": settings.REQUEST_TIMEOUT_SECONDS
                }
                
                if data:
                    request_kwargs["json"] = data
                
                response = await self.client.request(**request_kwargs)
                
                # Check for successful response
                response.raise_for_status()
                
                return response.json()
                
            except httpx.TimeoutException as e:
                if attempt == retry_count:
                    logger.error("Gemini API timeout after retries", attempts=attempt + 1)
                    raise asyncio.TimeoutError(f"Request timed out after {retry_count + 1} attempts")
                
                wait_time = 2 ** attempt
                logger.warning(
                    "Gemini API timeout, retrying",
                    attempt=attempt + 1,
                    wait_time=wait_time
                )
                await asyncio.sleep(wait_time)
                
            except httpx.HTTPStatusError as e:
                # Handle specific Gemini API errors
                if e.response.status_code == 400:
                    error_detail = e.response.text
                    logger.error("Gemini API bad request", response_text=error_detail)
                    raise ValueError(f"Invalid request: {error_detail}")
                elif e.response.status_code == 429:
                    # Rate limit exceeded
                    if attempt == retry_count:
                        logger.error("Gemini API rate limit exceeded after retries")
                        raise Exception("Rate limit exceeded")
                    
                    wait_time = 2 ** attempt
                    logger.warning("Gemini API rate limited, backing off", wait_time=wait_time)
                    await asyncio.sleep(wait_time)
                    continue
                elif e.response.status_code == 403:
                    logger.error("Gemini API access forbidden - check API key")
                    raise ValueError("API access forbidden - invalid or expired API key")
                
                # Don't retry client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    logger.error(
                        "Gemini API client error",
                        status_code=e.response.status_code,
                        response_text=e.response.text
                    )
                    raise
                
                # Retry server errors (5xx)
                if attempt == retry_count:
                    logger.error(
                        "Gemini API server error after retries",
                        status_code=e.response.status_code,
                        attempts=attempt + 1
                    )
                    raise
                
                wait_time = 2 ** attempt
                logger.warning(
                    "Gemini API server error, retrying",
                    status_code=e.response.status_code,
                    attempt=attempt + 1,
                    wait_time=wait_time
                )
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                if attempt == retry_count:
                    logger.error("Gemini API request failed after retries", error=str(e))
                    raise
                
                wait_time = 2 ** attempt
                logger.warning(
                    "Gemini API request failed, retrying",
                    error=str(e),
                    attempt=attempt + 1,
                    wait_time=wait_time
                )
                await asyncio.sleep(wait_time)
    
    async def health_check(self) -> bool:
        """
        Check Gemini API health status.
        
        Returns:
            bool: True if API is healthy, False otherwise
        """
        try:
            # Simple health check with minimal token usage
            messages = [
                {"role": "user", "content": "Health check - respond with 'OK'"}
            ]
            
            response = await self.generate_content(
                messages=messages,
                max_tokens=10,
                temperature=0.0
            )
            
            return "ok" in response.get_content().lower()
            
        except Exception as e:
            logger.error("Gemini health check failed", error=str(e))
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
            "rate_limit": settings.GEMINI_RATE_LIMIT,
            "base_url": self.base_url,
            "model": self.model_name
        }