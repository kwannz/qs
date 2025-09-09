"""
News Sentiment Analysis API Endpoints
新闻情绪分析API端点

Provides AI-powered news sentiment analysis for cryptocurrency markets
with real-time sentiment tracking and market impact assessment.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field, validator
import structlog

from config import settings
from core.cache import CacheManager, CacheKey, generate_hash
from core.rate_limiter import RateLimiter
from core.metrics import MetricsCollector, AIRequestTracker
from ai_clients.gemini_client import GeminiClient
from ai_clients.deepseek_client import DeepSeekClient


logger = structlog.get_logger(__name__)
router = APIRouter()


class NewsArticle(BaseModel):
    """News article model."""
    title: str = Field(..., description="Article title")
    content: str = Field(..., description="Article content/summary")
    source: str = Field(..., description="News source")
    published_at: datetime = Field(..., description="Publication timestamp")
    url: Optional[str] = Field(None, description="Article URL")
    author: Optional[str] = Field(None, description="Article author")
    sentiment_score: Optional[float] = Field(None, description="Pre-computed sentiment score")
    
    @validator("content")
    def validate_content_length(cls, v):
        """Validate content is not too long."""
        if len(v) > 5000:  # Truncate very long content
            return v[:5000] + "..."
        return v


class SentimentAnalysisRequest(BaseModel):
    """Sentiment analysis request model."""
    articles: List[NewsArticle] = Field(..., max_items=100, description="News articles to analyze")
    symbols: List[str] = Field(..., max_items=20, description="Cryptocurrency symbols to analyze")
    timeframe: str = Field(default="24h", description="Analysis timeframe")
    analysis_depth: str = Field(default="standard", description="Analysis depth (basic/standard/deep)")
    include_market_impact: bool = Field(default=True, description="Include market impact analysis")
    weight_by_source: bool = Field(default=True, description="Weight sentiment by source credibility")


class SymbolSentiment(BaseModel):
    """Sentiment analysis for a specific symbol."""
    symbol: str
    sentiment_score: float = Field(description="Sentiment score (-100 to +100)")
    sentiment_label: str = Field(description="Sentiment label (Very Bearish/Bearish/Neutral/Bullish/Very Bullish)")
    confidence: float = Field(description="Analysis confidence (0-100)")
    mention_count: int = Field(description="Number of mentions in articles")
    positive_mentions: int = Field(description="Number of positive mentions")
    negative_mentions: int = Field(description="Number of negative mentions")
    key_themes: List[str] = Field(default_factory=list, description="Key themes mentioned")
    price_impact_prediction: Optional[str] = Field(None, description="Predicted price impact")


class SentimentTheme(BaseModel):
    """Sentiment theme analysis."""
    theme: str = Field(description="Theme or topic")
    sentiment_score: float = Field(description="Theme sentiment score")
    relevance_score: float = Field(description="Theme relevance score")
    article_count: int = Field(description="Number of articles mentioning theme")
    key_keywords: List[str] = Field(default_factory=list, description="Key keywords")


class MarketImpactAnalysis(BaseModel):
    """Market impact analysis."""
    overall_impact: str = Field(description="Overall market impact (Low/Medium/High)")
    price_movement_prediction: str = Field(description="Predicted price movement direction")
    volatility_impact: str = Field(description="Expected volatility impact")
    time_horizon: str = Field(description="Time horizon for impact")
    confidence: float = Field(description="Impact prediction confidence")
    key_drivers: List[str] = Field(default_factory=list, description="Key sentiment drivers")


class SentimentAnalysisResponse(BaseModel):
    """Sentiment analysis response model."""
    timeframe: str
    analysis_depth: str
    articles_analyzed: int
    
    # Overall sentiment metrics
    overall_sentiment: str = Field(description="Overall market sentiment")
    overall_sentiment_score: float = Field(description="Overall sentiment score (-100 to +100)")
    sentiment_strength: str = Field(description="Sentiment strength (Weak/Moderate/Strong)")
    
    # Symbol-specific sentiment
    symbol_sentiments: List[SymbolSentiment]
    
    # Theme analysis
    key_themes: List[SentimentTheme] = Field(default_factory=list)
    
    # Market impact analysis
    market_impact: Optional[MarketImpactAnalysis] = None
    
    # Sentiment trends
    sentiment_trend: str = Field(description="Sentiment trend (Improving/Stable/Deteriorating)")
    momentum: str = Field(description="Sentiment momentum (Accelerating/Steady/Decelerating)")
    
    # Source analysis
    source_breakdown: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Alert conditions
    sentiment_alerts: List[str] = Field(default_factory=list)
    
    # Metadata
    analyzed_at: datetime
    model_used: str
    processing_time_ms: int


# Mock news data for development/testing
SAMPLE_ARTICLES = [
    NewsArticle(
        title="Bitcoin Reaches New All-Time High Amid Institutional Adoption",
        content="Bitcoin has surged to unprecedented levels as major institutions continue to announce adoption plans...",
        source="CoinDesk",
        published_at=datetime.utcnow() - timedelta(hours=2)
    ),
    NewsArticle(
        title="Ethereum Network Upgrade Shows Promise for Scalability",
        content="The latest Ethereum upgrade demonstrates significant improvements in transaction throughput and cost reduction...",
        source="Decrypt", 
        published_at=datetime.utcnow() - timedelta(hours=4)
    ),
    NewsArticle(
        title="Regulatory Concerns Weigh on Cryptocurrency Markets",
        content="New regulatory proposals have created uncertainty in the cryptocurrency space, with traders expressing caution...",
        source="Bloomberg",
        published_at=datetime.utcnow() - timedelta(hours=6)
    )
]


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


async def get_gemini_client() -> GeminiClient:
    from main import app
    return app.state.gemini_client


async def get_deepseek_client() -> DeepSeekClient:
    from main import app
    return app.state.deepseek_client


@router.post("/analyze", response_model=SentimentAnalysisResponse)
async def analyze_news_sentiment(
    request: SentimentAnalysisRequest,
    background_tasks: BackgroundTasks,
    cache_manager: CacheManager = Depends(get_cache_manager),
    rate_limiter: RateLimiter = Depends(get_rate_limiter),
    metrics: MetricsCollector = Depends(get_metrics_collector),
    gemini_client: GeminiClient = Depends(get_gemini_client),
    deepseek_client: DeepSeekClient = Depends(get_deepseek_client)
):
    """
    Analyze news sentiment for cryptocurrency markets.
    
    Processes news articles to extract sentiment signals, market impact predictions,
    and trading-relevant insights for specified cryptocurrency symbols.
    """
    start_time = datetime.utcnow()
    
    if len(request.articles) == 0:
        raise HTTPException(
            status_code=400,
            detail="At least one news article is required"
        )
    
    try:
        # Apply rate limiting
        await rate_limiter.wait_for_service("news_sentiment")
        
        # Generate cache key
        articles_hash = generate_hash([article.dict() for article in request.articles])
        symbols_hash = generate_hash(request.symbols)
        cache_key = CacheKey.news_sentiment(f"{articles_hash}_{symbols_hash}", request.timeframe)
        
        # Check cache
        cached_result = await cache_manager.get(cache_key)
        if cached_result:
            logger.info(
                "News sentiment analysis served from cache",
                articles_count=len(request.articles),
                symbols=request.symbols,
                cache_key=str(cache_key)
            )
            
            cached_result["processing_time_ms"] = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )
            return SentimentAnalysisResponse(**cached_result)
        
        # Perform primary sentiment analysis with Gemini (better for content analysis)
        with AIRequestTracker(
            metrics, "gemini", "gemini-1.5-flash", "news_sentiment"
        ) as tracker:
            sentiment_analysis = await gemini_client.news_sentiment_analysis(
                news_articles=[article.dict() for article in request.articles],
                symbols=request.symbols,
                timeframe=request.timeframe
            )
            
            if "usage_metadata" in sentiment_analysis:
                total_tokens = sentiment_analysis["usage_metadata"].get("totalTokenCount", 0)
                tracker.set_tokens_used(total_tokens)
        
        # Get market impact analysis with DeepSeek if requested
        market_impact = None
        if request.include_market_impact:
            try:
                with AIRequestTracker(
                    metrics, "deepseek", "deepseek-chat", "market_impact_analysis"
                ) as tracker:
                    impact_system_prompt = """You are an expert crypto market analyst specializing in news impact assessment.
                    
                    Analyze the sentiment analysis results and predict:
                    1. Market impact magnitude and direction
                    2. Volatility implications
                    3. Time horizon for impact
                    4. Key risk factors
                    
                    Provide structured analysis focused on trading implications."""
                    
                    impact_messages = [
                        {
                            "role": "user",
                            "content": f"Analyze market impact for sentiment analysis:\n\n{sentiment_analysis}\n\nSymbols: {request.symbols}\nTimeframe: {request.timeframe}"
                        }
                    ]
                    
                    impact_response = await deepseek_client.chat_completion(
                        messages=impact_messages,
                        system_prompt=impact_system_prompt,
                        temperature=0.1,
                        max_tokens=2000
                    )
                    
                    impact_content = impact_response.get_content()
                    
                    # Parse market impact (simplified for demo)
                    market_impact = MarketImpactAnalysis(
                        overall_impact="Medium",
                        price_movement_prediction="Neutral to Slightly Positive",
                        volatility_impact="Moderate",
                        time_horizon="24-48 hours",
                        confidence=75.0,
                        key_drivers=["Institutional adoption news", "Regulatory uncertainty"]
                    )
                    
            except Exception as e:
                logger.warning(
                    "Market impact analysis failed",
                    symbols=request.symbols,
                    error=str(e)
                )
        
        # Parse sentiment analysis results
        overall_sentiment_score = sentiment_analysis.get("sentiment_score", 0)
        
        # Determine overall sentiment label
        if overall_sentiment_score >= 60:
            overall_sentiment = "Very Bullish"
        elif overall_sentiment_score >= 20:
            overall_sentiment = "Bullish"
        elif overall_sentiment_score >= -20:
            overall_sentiment = "Neutral"
        elif overall_sentiment_score >= -60:
            overall_sentiment = "Bearish"
        else:
            overall_sentiment = "Very Bearish"
        
        # Determine sentiment strength
        abs_score = abs(overall_sentiment_score)
        if abs_score >= 70:
            sentiment_strength = "Strong"
        elif abs_score >= 40:
            sentiment_strength = "Moderate"
        else:
            sentiment_strength = "Weak"
        
        # Process symbol-specific sentiments
        symbol_sentiments = []
        symbol_sentiment_data = sentiment_analysis.get("symbol_sentiments", {})
        
        for symbol in request.symbols:
            symbol_data = symbol_sentiment_data.get(symbol, {})
            
            symbol_score = symbol_data.get("sentiment_score", overall_sentiment_score)
            
            # Determine symbol sentiment label
            if symbol_score >= 60:
                symbol_label = "Very Bullish"
            elif symbol_score >= 20:
                symbol_label = "Bullish"
            elif symbol_score >= -20:
                symbol_label = "Neutral"
            elif symbol_score >= -60:
                symbol_label = "Bearish"
            else:
                symbol_label = "Very Bearish"
            
            symbol_sentiments.append(SymbolSentiment(
                symbol=symbol,
                sentiment_score=symbol_score,
                sentiment_label=symbol_label,
                confidence=symbol_data.get("confidence", 70.0),
                mention_count=symbol_data.get("mention_count", 0),
                positive_mentions=symbol_data.get("positive_mentions", 0),
                negative_mentions=symbol_data.get("negative_mentions", 0),
                key_themes=symbol_data.get("key_themes", []),
                price_impact_prediction=symbol_data.get("price_impact", "Neutral")
            ))
        
        # Process key themes
        key_themes = []
        themes_data = sentiment_analysis.get("key_themes", [])
        
        for theme_data in themes_data:
            if isinstance(theme_data, dict):
                key_themes.append(SentimentTheme(
                    theme=theme_data.get("theme", "Unknown"),
                    sentiment_score=theme_data.get("sentiment_score", 0),
                    relevance_score=theme_data.get("relevance_score", 50),
                    article_count=theme_data.get("article_count", 1),
                    key_keywords=theme_data.get("keywords", [])
                ))
        
        # Source breakdown analysis
        source_breakdown = {}
        source_counts = {}
        for article in request.articles:
            source = article.source
            if source not in source_counts:
                source_counts[source] = {"count": 0, "avg_sentiment": 0}
            source_counts[source]["count"] += 1
        
        for source, data in source_counts.items():
            source_breakdown[source] = {
                "article_count": data["count"],
                "credibility_weight": 1.0,  # Would be calculated based on source reputation
                "avg_sentiment": overall_sentiment_score  # Simplified for demo
            }
        
        # Generate sentiment alerts
        sentiment_alerts = []
        if abs(overall_sentiment_score) >= 70:
            sentiment_alerts.append(f"Strong {overall_sentiment.lower()} sentiment detected")
        
        for symbol_sentiment in symbol_sentiments:
            if abs(symbol_sentiment.sentiment_score) >= 80:
                sentiment_alerts.append(
                    f"Extreme sentiment for {symbol_sentiment.symbol}: {symbol_sentiment.sentiment_label}"
                )
        
        # Calculate processing time
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Prepare response
        response_data = {
            "timeframe": request.timeframe,
            "analysis_depth": request.analysis_depth,
            "articles_analyzed": len(request.articles),
            "overall_sentiment": overall_sentiment,
            "overall_sentiment_score": overall_sentiment_score,
            "sentiment_strength": sentiment_strength,
            "symbol_sentiments": [s.dict() for s in symbol_sentiments],
            "key_themes": [t.dict() for t in key_themes],
            "market_impact": market_impact.dict() if market_impact else None,
            "sentiment_trend": "Stable",  # Would be calculated from historical data
            "momentum": "Steady",  # Would be calculated from trend analysis
            "source_breakdown": source_breakdown,
            "sentiment_alerts": sentiment_alerts,
            "analyzed_at": start_time,
            "model_used": "gemini-1.5-flash",
            "processing_time_ms": processing_time
        }
        
        # Cache the result
        background_tasks.add_task(
            cache_manager.set,
            cache_key,
            response_data,
            settings.NEWS_SENTIMENT_CACHE_TTL
        )
        
        # Record business metrics
        metrics.record_analysis_accuracy(
            analysis_type="news_sentiment",
            accuracy_score=sentiment_analysis.get("confidence_level", 70)
        )
        
        logger.info(
            "News sentiment analysis completed",
            articles_count=len(request.articles),
            symbols_count=len(request.symbols),
            overall_sentiment=overall_sentiment,
            sentiment_score=overall_sentiment_score,
            processing_time_ms=processing_time
        )
        
        return SentimentAnalysisResponse(**response_data)
        
    except Exception as e:
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        logger.error(
            "News sentiment analysis failed",
            articles_count=len(request.articles),
            symbols=request.symbols,
            processing_time_ms=processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"News sentiment analysis failed: {str(e)}"
        )


@router.get("/sample-articles")
async def get_sample_articles():
    """Get sample news articles for testing sentiment analysis."""
    return {
        "sample_articles": [article.dict() for article in SAMPLE_ARTICLES],
        "usage_note": "These are sample articles for testing. In production, integrate with real news APIs.",
        "recommended_sources": [
            "CoinDesk",
            "CoinTelegraph",
            "Decrypt", 
            "The Block",
            "Bloomberg Crypto",
            "Reuters Crypto",
            "Yahoo Finance Crypto"
        ]
    }


@router.post("/batch-analyze")
async def batch_sentiment_analysis(
    articles: List[NewsArticle],
    symbols: List[str],
    cache_manager: CacheManager = Depends(get_cache_manager),
    gemini_client: GeminiClient = Depends(get_gemini_client)
):
    """
    Perform batch sentiment analysis for historical data analysis.
    
    Processes large batches of articles efficiently with optimized caching
    and parallel processing where applicable.
    """
    if len(articles) > 500:
        raise HTTPException(
            status_code=400,
            detail="Maximum 500 articles allowed in batch analysis"
        )
    
    start_time = datetime.utcnow()
    
    try:
        # Group articles by publication date for time-series analysis
        from collections import defaultdict
        articles_by_date = defaultdict(list)
        
        for article in articles:
            date_key = article.published_at.strftime("%Y-%m-%d")
            articles_by_date[date_key].append(article)
        
        # Process each date group
        daily_results = {}
        
        for date_str, daily_articles in articles_by_date.items():
            # Generate cache key for this date
            articles_hash = generate_hash([a.dict() for a in daily_articles])
            cache_key = f"batch_sentiment:{date_str}:{articles_hash}"
            
            # Check cache first
            cached_result = await cache_manager.get(cache_key)
            if cached_result:
                daily_results[date_str] = cached_result
                continue
            
            # Analyze this day's articles
            if len(daily_articles) <= 20:  # Small batch, process directly
                sentiment_result = await gemini_client.news_sentiment_analysis(
                    news_articles=[a.dict() for a in daily_articles],
                    symbols=symbols,
                    timeframe="1d"
                )
                
                daily_results[date_str] = {
                    "date": date_str,
                    "articles_count": len(daily_articles),
                    "sentiment_score": sentiment_result.get("sentiment_score", 0),
                    "overall_sentiment": sentiment_result.get("overall_sentiment", "neutral"),
                    "symbol_sentiments": sentiment_result.get("symbol_sentiments", {})
                }
                
                # Cache the result
                await cache_manager.set(cache_key, daily_results[date_str], 3600)  # 1 hour cache
        
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Calculate overall trends
        sentiment_scores = [result.get("sentiment_score", 0) for result in daily_results.values()]
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        
        result = {
            "batch_summary": {
                "total_articles": len(articles),
                "date_range": {
                    "start_date": min(articles, key=lambda x: x.published_at).published_at.strftime("%Y-%m-%d"),
                    "end_date": max(articles, key=lambda x: x.published_at).published_at.strftime("%Y-%m-%d")
                },
                "average_sentiment_score": avg_sentiment,
                "processing_time_ms": processing_time
            },
            "daily_results": daily_results,
            "trend_analysis": {
                "sentiment_trend": "stable",  # Would calculate actual trend
                "volatility": "medium",  # Would calculate sentiment volatility
                "key_events": []  # Would identify significant sentiment shifts
            }
        }
        
        logger.info(
            "Batch sentiment analysis completed",
            total_articles=len(articles),
            date_groups=len(daily_results),
            processing_time_ms=processing_time
        )
        
        return result
        
    except Exception as e:
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        logger.error(
            "Batch sentiment analysis failed",
            articles_count=len(articles),
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail=f"Batch sentiment analysis failed: {str(e)}"
        )


@router.get("/sentiment-sources")
async def get_sentiment_sources():
    """Get information about supported news sources and their characteristics."""
    return {
        "supported_sources": {
            "tier_1": {
                "sources": ["Bloomberg", "Reuters", "Wall Street Journal"],
                "characteristics": {
                    "credibility": "Very High",
                    "response_time": "Fast",
                    "market_impact": "High",
                    "weight_multiplier": 1.5
                }
            },
            "tier_2": {
                "sources": ["CoinDesk", "The Block", "Decrypt"],
                "characteristics": {
                    "credibility": "High",
                    "response_time": "Very Fast",
                    "market_impact": "Medium-High",
                    "weight_multiplier": 1.2
                }
            },
            "tier_3": {
                "sources": ["CoinTelegraph", "CryptoNews", "BeInCrypto"],
                "characteristics": {
                    "credibility": "Medium",
                    "response_time": "Fast",
                    "market_impact": "Medium",
                    "weight_multiplier": 1.0
                }
            }
        },
        "sentiment_ranges": {
            "very_bearish": "< -60",
            "bearish": "-60 to -20",
            "neutral": "-20 to +20",
            "bullish": "+20 to +60",
            "very_bullish": "> +60"
        },
        "confidence_levels": {
            "high": "> 80% - Strong signal with multiple confirmations",
            "medium": "50-80% - Moderate signal with some confirmation",
            "low": "< 50% - Weak signal, requires additional confirmation"
        }
    }