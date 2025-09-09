"""
AI Service Integration Tests
AI服务集成测试

Comprehensive test suite for validating AI service functionality,
API endpoints, and integration with the trading platform.
"""

import asyncio
import pytest
import httpx
import json
from datetime import datetime
from typing import Dict, Any

# Test configuration
AI_SERVICE_URL = "http://localhost:8082"
GATEWAY_SERVICE_URL = "http://localhost:8081"
TEST_TIMEOUT = 30.0


class TestAIServiceIntegration:
    """Test AI service integration and functionality."""
    
    @pytest.fixture(scope="class")
    async def ai_client(self):
        """Create HTTP client for AI service."""
        async with httpx.AsyncClient(
            base_url=AI_SERVICE_URL,
            timeout=TEST_TIMEOUT
        ) as client:
            yield client
    
    @pytest.fixture(scope="class") 
    async def gateway_client(self):
        """Create HTTP client for gateway service."""
        async with httpx.AsyncClient(
            base_url=GATEWAY_SERVICE_URL,
            timeout=TEST_TIMEOUT
        ) as client:
            yield client

    async def test_ai_service_health(self, ai_client):
        """Test AI service health check."""
        response = await ai_client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] in ["healthy", "degraded"]
        assert "dependencies" in health_data
        
        print(f"AI Service Health: {health_data['status']}")

    async def test_gateway_ai_health(self, gateway_client):
        """Test gateway AI integration health."""
        response = await gateway_client.get("/api/ai/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert "ai_service" in health_data
        
        print(f"Gateway AI Integration: {health_data['status']}")

    async def test_market_analysis_endpoint(self, ai_client):
        """Test market analysis endpoint."""
        request_data = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "analysis_type": "trend",
            "market_data": {
                "current_price": 50000.0,
                "volume_24h": 1000000000,
                "price_change_24h": 2.5,
                "technical_indicators": {
                    "rsi": 55.0,
                    "macd": 0.12,
                    "bb_upper": 52000,
                    "bb_lower": 48000
                }
            },
            "use_validation": False,  # Skip validation for faster tests
            "priority": "high"
        }
        
        response = await ai_client.post(
            "/api/v1/market-analysis/analyze",
            json=request_data
        )
        
        assert response.status_code == 200
        analysis = response.json()
        
        # Validate response structure
        required_fields = [
            "symbol", "timeframe", "trend_direction", "trend_strength",
            "confidence", "recommendation", "risk_level"
        ]
        
        for field in required_fields:
            assert field in analysis, f"Missing field: {field}"
        
        # Validate data types and ranges
        assert analysis["symbol"] == "BTCUSDT"
        assert analysis["timeframe"] == "1h"
        assert analysis["trend_direction"] in ["bullish", "bearish", "neutral", "consolidating"]
        assert 1 <= analysis["trend_strength"] <= 10
        assert 0 <= analysis["confidence"] <= 100
        assert analysis["risk_level"] in ["low", "medium", "high", "extreme"]
        
        print(f"Market Analysis - {analysis['symbol']}: {analysis['trend_direction']} "
              f"(confidence: {analysis['confidence']}%)")

    async def test_strategy_optimization_endpoint(self, ai_client):
        """Test strategy optimization endpoint."""
        request_data = {
            "strategy_config": {
                "name": "Test Trend Following",
                "type": "trend_following",
                "parameters": {
                    "sma_fast": 20,
                    "sma_slow": 50,
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30
                },
                "risk_parameters": {
                    "stop_loss_pct": 2.0,
                    "take_profit_pct": 4.0,
                    "position_size_pct": 10.0
                },
                "timeframe": "1h",
                "symbols": ["BTCUSDT", "ETHUSDT"]
            },
            "backtest_results": {
                "total_returns": 25.5,
                "annualized_returns": 18.3,
                "sharpe_ratio": 1.45,
                "sortino_ratio": 1.78,
                "max_drawdown": -8.2,
                "win_rate": 62.5,
                "profit_factor": 1.85,
                "total_trades": 156,
                "avg_trade_duration": 18.5,
                "var_95": -2.1
            },
            "market_conditions": {
                "volatility_regime": "medium",
                "trend_regime": "trending",
                "correlation_environment": "medium",
                "liquidity_conditions": "good",
                "market_sentiment": "bullish",
                "economic_indicators": {
                    "fear_greed_index": 65,
                    "vix_equivalent": 0.45
                }
            },
            "optimization_goals": ["maximize_sharpe", "minimize_drawdown"],
            "use_validation": False,
            "constraints": {
                "max_parameter_change": 0.3,
                "preserve_strategy_type": True
            }
        }
        
        response = await ai_client.post(
            "/api/v1/strategy-optimization/optimize",
            json=request_data
        )
        
        assert response.status_code == 200
        optimization = response.json()
        
        # Validate response structure
        required_fields = [
            "strategy_name", "optimization_type", "current_performance",
            "parameter_recommendations", "risk_assessment", "expected_performance"
        ]
        
        for field in required_fields:
            assert field in optimization, f"Missing field: {field}"
        
        assert optimization["strategy_name"] == "Test Trend Following"
        assert isinstance(optimization["parameter_recommendations"], list)
        
        print(f"Strategy Optimization - {optimization['strategy_name']}: "
              f"{len(optimization['parameter_recommendations'])} recommendations")

    async def test_risk_assessment_endpoint(self, ai_client):
        """Test risk assessment endpoint."""
        request_data = {
            "portfolio": {
                "total_value": 100000.0,
                "available_cash": 15000.0,
                "positions": [
                    {
                        "symbol": "BTCUSDT",
                        "quantity": 1.5,
                        "entry_price": 45000.0,
                        "current_price": 50000.0,
                        "market_value": 75000.0,
                        "unrealized_pnl": 7500.0,
                        "position_weight": 75.0
                    },
                    {
                        "symbol": "ETHUSDT", 
                        "quantity": 3.0,
                        "entry_price": 3000.0,
                        "current_price": 3200.0,
                        "market_value": 9600.0,
                        "unrealized_pnl": 600.0,
                        "position_weight": 9.6
                    }
                ],
                "leverage_ratio": 1.0,
                "margin_used": 0.0
            },
            "market_data": {
                "volatility_index": 0.65,
                "correlation_matrix": {
                    "BTCUSDT": {"ETHUSDT": 0.78},
                    "ETHUSDT": {"BTCUSDT": 0.78}
                },
                "liquidity_scores": {
                    "BTCUSDT": 0.95,
                    "ETHUSDT": 0.88
                },
                "beta_values": {
                    "BTCUSDT": 1.0,
                    "ETHUSDT": 1.2
                },
                "historical_volatility": {
                    "BTCUSDT": 0.6,
                    "ETHUSDT": 0.8
                }
            },
            "risk_parameters": {
                "max_portfolio_risk": 2.0,
                "max_position_size": 25.0,
                "var_confidence_level": 0.95,
                "max_correlation_exposure": 0.7,
                "liquidity_threshold": 0.3
            },
            "assessment_type": "comprehensive",
            "include_stress_testing": True,
            "use_validation": False
        }
        
        response = await ai_client.post(
            "/api/v1/risk-assessment/assess",
            json=request_data
        )
        
        assert response.status_code == 200
        assessment = response.json()
        
        # Validate response structure
        required_fields = [
            "portfolio_value", "overall_risk_score", "risk_rating",
            "risk_breakdown", "var_analysis", "risk_mitigation_recommendations"
        ]
        
        for field in required_fields:
            assert field in assessment, f"Missing field: {field}"
        
        assert assessment["portfolio_value"] == 100000.0
        assert 0 <= assessment["overall_risk_score"] <= 100
        assert assessment["risk_rating"] in ["Low", "Medium", "High", "Extreme"]
        
        print(f"Risk Assessment - Portfolio Value: ${assessment['portfolio_value']:,.0f}, "
              f"Risk Score: {assessment['overall_risk_score']}/100 ({assessment['risk_rating']})")

    async def test_news_sentiment_endpoint(self, ai_client):
        """Test news sentiment analysis endpoint."""
        request_data = {
            "articles": [
                {
                    "title": "Bitcoin Reaches New All-Time High as Institutional Adoption Grows",
                    "content": "Bitcoin has surged to unprecedented levels following announcements from major financial institutions about cryptocurrency adoption plans. The rally has been supported by increased trading volume and positive sentiment across crypto markets.",
                    "source": "CoinDesk",
                    "published_at": datetime.utcnow().isoformat() + "Z"
                },
                {
                    "title": "Ethereum Network Upgrade Shows Promise for Scalability Solutions", 
                    "content": "The latest Ethereum network upgrade has demonstrated significant improvements in transaction throughput and cost reduction, leading to optimism about the network's future scalability.",
                    "source": "Decrypt",
                    "published_at": datetime.utcnow().isoformat() + "Z"
                },
                {
                    "title": "Regulatory Concerns Create Uncertainty in Cryptocurrency Markets",
                    "content": "New regulatory proposals have introduced uncertainty in the cryptocurrency space, with traders expressing caution about potential impacts on market operations.",
                    "source": "Bloomberg",
                    "published_at": datetime.utcnow().isoformat() + "Z"
                }
            ],
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "timeframe": "24h",
            "analysis_depth": "standard",
            "include_market_impact": True,
            "weight_by_source": True
        }
        
        response = await ai_client.post(
            "/api/v1/news-sentiment/analyze",
            json=request_data
        )
        
        assert response.status_code == 200
        sentiment = response.json()
        
        # Validate response structure
        required_fields = [
            "timeframe", "articles_analyzed", "overall_sentiment",
            "overall_sentiment_score", "symbol_sentiments"
        ]
        
        for field in required_fields:
            assert field in sentiment, f"Missing field: {field}"
        
        assert sentiment["articles_analyzed"] == 3
        assert -100 <= sentiment["overall_sentiment_score"] <= 100
        assert sentiment["overall_sentiment"] in [
            "Very Bearish", "Bearish", "Neutral", "Bullish", "Very Bullish"
        ]
        
        print(f"News Sentiment - Articles: {sentiment['articles_analyzed']}, "
              f"Sentiment: {sentiment['overall_sentiment']} "
              f"(Score: {sentiment['overall_sentiment_score']})")

    async def test_bulk_market_analysis(self, ai_client):
        """Test bulk market analysis endpoint."""
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
        
        request_data = {
            "symbols": symbols,
            "timeframe": "1h",
            "analysis_type": "trend",
            "use_validation": False,
            "priority": "low"
        }
        
        response = await ai_client.post(
            "/api/v1/market-analysis/bulk",
            json=request_data
        )
        
        assert response.status_code == 200
        bulk_analysis = response.json()
        
        assert "results" in bulk_analysis
        assert "summary" in bulk_analysis
        
        summary = bulk_analysis["summary"]
        assert summary["total_symbols"] == len(symbols)
        assert summary["successful_analyses"] >= 0
        
        print(f"Bulk Analysis - Symbols: {summary['total_symbols']}, "
              f"Successful: {summary['successful_analyses']}")

    async def test_gateway_integration(self, gateway_client):
        """Test gateway integration with AI services."""
        # Test market analysis through gateway
        request_data = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "analysis_type": "trend",
            "market_data": {
                "current_price": 50000.0,
                "volume_24h": 1000000000,
                "price_change_24h": 2.5
            },
            "use_validation": False,
            "priority": "normal"
        }
        
        response = await gateway_client.post(
            "/api/ai/market-analysis",
            json=request_data
        )
        
        assert response.status_code == 200
        gateway_response = response.json()
        
        assert "analysis" in gateway_response
        assert "request_id" in gateway_response
        assert "processed_at" in gateway_response
        
        print(f"Gateway Integration - Request ID: {gateway_response['request_id']}")

    async def test_error_handling(self, ai_client):
        """Test error handling and validation."""
        # Test invalid symbol
        invalid_request = {
            "symbol": "",  # Invalid empty symbol
            "timeframe": "1h",
            "analysis_type": "trend",
            "market_data": {},
            "use_validation": False
        }
        
        response = await ai_client.post(
            "/api/v1/market-analysis/analyze",
            json=invalid_request
        )
        
        assert response.status_code == 422  # Validation error
        
        # Test invalid timeframe
        invalid_timeframe = {
            "symbol": "BTCUSDT",
            "timeframe": "invalid",  # Invalid timeframe
            "analysis_type": "trend",
            "market_data": {},
            "use_validation": False
        }
        
        response = await ai_client.post(
            "/api/v1/market-analysis/analyze", 
            json=invalid_timeframe
        )
        
        assert response.status_code == 422  # Validation error
        
        print("Error handling tests passed")

    async def test_performance_benchmarks(self, ai_client):
        """Test performance benchmarks."""
        import time
        
        # Test market analysis response time
        start_time = time.time()
        
        request_data = {
            "symbol": "BTCUSDT",
            "timeframe": "1h",
            "analysis_type": "trend",
            "market_data": {
                "current_price": 50000.0,
                "volume_24h": 1000000000
            },
            "use_validation": False
        }
        
        response = await ai_client.post(
            "/api/v1/market-analysis/analyze",
            json=request_data
        )
        
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < 10.0  # Should respond within 10 seconds
        
        analysis = response.json()
        processing_time = analysis.get("processing_time_ms", 0)
        
        print(f"Performance - Response Time: {response_time:.2f}s, "
              f"Processing Time: {processing_time}ms")


@pytest.mark.asyncio
class TestAIServiceMetrics:
    """Test AI service metrics and monitoring."""
    
    async def test_prometheus_metrics(self):
        """Test Prometheus metrics endpoint."""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            response = await client.get(f"{AI_SERVICE_URL}/metrics")
            
            assert response.status_code == 200
            metrics_text = response.text
            
            # Check for expected metrics
            expected_metrics = [
                "ai_requests_total",
                "ai_request_duration_seconds",
                "cache_operations_total",
                "system_cpu_usage_percent"
            ]
            
            for metric in expected_metrics:
                assert metric in metrics_text, f"Missing metric: {metric}"
            
            print(f"Prometheus metrics available: {len(metrics_text)} bytes")


async def run_integration_tests():
    """Run all integration tests."""
    print("🚀 Starting AI Service Integration Tests")
    print("=" * 50)
    
    # Check service availability
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            ai_health = await client.get(f"{AI_SERVICE_URL}/health")
            gateway_health = await client.get(f"{GATEWAY_SERVICE_URL}/health")
            
            if ai_health.status_code != 200:
                print(f"❌ AI Service not available at {AI_SERVICE_URL}")
                return
                
            if gateway_health.status_code != 200:
                print(f"❌ Gateway Service not available at {GATEWAY_SERVICE_URL}")
                return
                
    except Exception as e:
        print(f"❌ Services not available: {e}")
        return
    
    print("✅ Services are available, starting tests...")
    print()
    
    # Run tests using pytest
    import subprocess
    result = subprocess.run([
        "python", "-m", "pytest", __file__, "-v", "--tb=short"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    if result.returncode == 0:
        print("\n🎉 All integration tests passed!")
    else:
        print("\n❌ Some tests failed. Check the output above.")
    
    return result.returncode == 0


if __name__ == "__main__":
    # Run tests when script is executed directly
    success = asyncio.run(run_integration_tests())
    exit(0 if success else 1)