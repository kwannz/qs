"""
Configuration management for AI Service
AI服务的配置管理

This module handles all configuration settings for the AI service,
including environment variables, API keys, and service parameters.
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Application settings with validation and type checking."""
    
    # Service Configuration
    AI_SERVICE_HOST: str = Field(default="0.0.0.0", description="AI service bind address")
    AI_SERVICE_PORT: int = Field(default=8082, description="AI service port")
    ENVIRONMENT: str = Field(default="development", description="Deployment environment")
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # AI Provider Configuration
    DEEPSEEK_API_KEY: str = Field(..., description="DeepSeek API key")
    DEEPSEEK_BASE_URL: str = Field(
        default="https://api.deepseek.com/v1",
        description="DeepSeek API base URL"
    )
    DEEPSEEK_RATE_LIMIT: int = Field(
        default=80,
        description="DeepSeek API rate limit (requests per minute)"
    )
    
    GEMINI_API_KEY: str = Field(..., description="Google Gemini API key")
    GEMINI_BASE_URL: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        description="Google Gemini API base URL"
    )
    GEMINI_RATE_LIMIT: int = Field(
        default=60,
        description="Gemini API rate limit (requests per minute)"
    )
    
    # Redis Configuration
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    REDIS_DB: int = Field(default=1, description="Redis database number")
    REDIS_PASSWORD: Optional[str] = Field(default=None, description="Redis password")
    REDIS_POOL_SIZE: int = Field(default=20, description="Redis connection pool size")
    REDIS_CACHE_TTL: int = Field(default=300, description="Default cache TTL in seconds")
    
    # Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql://postgres:password@localhost:5432/trading_db",
        description="PostgreSQL connection URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=10, description="Database connection pool size")
    
    # Circuit Breaker Configuration
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(
        default=5,
        description="Circuit breaker failure threshold"
    )
    CIRCUIT_BREAKER_TIMEOUT_SECONDS: int = Field(
        default=60,
        description="Circuit breaker timeout in seconds"
    )
    CIRCUIT_BREAKER_SUCCESS_THRESHOLD: int = Field(
        default=3,
        description="Circuit breaker success threshold for recovery"
    )
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(
        default=1000,
        description="API rate limit requests per minute"
    )
    RATE_LIMIT_BURST_SIZE: int = Field(
        default=50,
        description="Rate limiting burst size"
    )
    
    # Cache TTL Configuration (in seconds)
    MARKET_ANALYSIS_CACHE_TTL: int = Field(
        default=60,
        description="Market analysis cache TTL"
    )
    STRATEGY_ANALYSIS_CACHE_TTL: int = Field(
        default=300,
        description="Strategy analysis cache TTL"
    )
    RISK_ANALYSIS_CACHE_TTL: int = Field(
        default=180,
        description="Risk analysis cache TTL"
    )
    NEWS_SENTIMENT_CACHE_TTL: int = Field(
        default=900,
        description="News sentiment analysis cache TTL"
    )
    
    # Performance Settings
    MAX_CONCURRENT_AI_REQUESTS: int = Field(
        default=10,
        description="Maximum concurrent AI API requests"
    )
    REQUEST_TIMEOUT_SECONDS: int = Field(
        default=30,
        description="HTTP request timeout in seconds"
    )
    BATCH_PROCESSING_SIZE: int = Field(
        default=100,
        description="Batch processing size for bulk operations"
    )
    
    # Monitoring Configuration
    ENABLE_METRICS: bool = Field(default=True, description="Enable Prometheus metrics")
    METRICS_PORT: int = Field(default=9090, description="Metrics server port")
    HEALTH_CHECK_INTERVAL: int = Field(
        default=30,
        description="Health check interval in seconds"
    )
    
    # Security Configuration
    SECRET_KEY: str = Field(..., description="Application secret key")
    API_KEY_HEADER: str = Field(
        default="X-API-Key",
        description="API key header name"
    )
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8081"],
        description="CORS allowed origins"
    )
    
    # External Service URLs
    MARKET_DATA_SERVICE_URL: str = Field(
        default="http://localhost:8080",
        description="Market data service URL"
    )
    GATEWAY_SERVICE_URL: str = Field(
        default="http://localhost:8081",
        description="Gateway service URL"
    )
    
    # Optional External APIs
    NEWS_API_KEY: Optional[str] = Field(default=None, description="News API key")
    TWITTER_BEARER_TOKEN: Optional[str] = Field(default=None, description="Twitter Bearer token")
    REDDIT_CLIENT_ID: Optional[str] = Field(default=None, description="Reddit client ID")
    REDDIT_CLIENT_SECRET: Optional[str] = Field(default=None, description="Reddit client secret")
    
    # Model Configuration
    DEFAULT_MODEL: str = Field(default="deepseek", description="Default AI model to use")
    FALLBACK_MODEL: str = Field(default="gemini", description="Fallback AI model")
    MODEL_TEMPERATURE: float = Field(default=0.1, description="AI model temperature")
    MAX_TOKENS: int = Field(default=4000, description="Maximum tokens per AI request")
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_allowed_origins(cls, v):
        """Parse comma-separated allowed origins."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Validate environment."""
        valid_envs = ["development", "testing", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Invalid environment: {v}. Must be one of {valid_envs}")
        return v.lower()
    
    @validator("MODEL_TEMPERATURE")
    def validate_temperature(cls, v):
        """Validate model temperature range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Model temperature must be between 0.0 and 2.0")
        return v
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
        # Environment variable prefixes
        env_prefix = ""
        
        # Validation
        validate_assignment = True
        use_enum_values = True
        
        # Field descriptions in schema
        schema_extra = {
            "example": {
                "AI_SERVICE_HOST": "0.0.0.0",
                "AI_SERVICE_PORT": 8082,
                "ENVIRONMENT": "development",
                "LOG_LEVEL": "INFO",
                "DEEPSEEK_API_KEY": "your_deepseek_api_key",
                "GEMINI_API_KEY": "your_gemini_api_key",
                "REDIS_URL": "redis://localhost:6379",
                "SECRET_KEY": "your-secret-key-here"
            }
        }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance."""
    return settings


# Configuration validation
def validate_configuration():
    """Validate critical configuration settings."""
    errors = []
    
    # Check required API keys
    if not settings.DEEPSEEK_API_KEY or settings.DEEPSEEK_API_KEY == "your_deepseek_api_key_here":
        errors.append("DEEPSEEK_API_KEY is not configured")
    
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == "your_gemini_api_key_here":
        errors.append("GEMINI_API_KEY is not configured")
    
    if not settings.SECRET_KEY or settings.SECRET_KEY == "your-secret-key-change-in-production":
        if settings.ENVIRONMENT == "production":
            errors.append("SECRET_KEY must be changed in production")
    
    # Validate port ranges
    if not 1024 <= settings.AI_SERVICE_PORT <= 65535:
        errors.append("AI_SERVICE_PORT must be between 1024 and 65535")
    
    if settings.ENABLE_METRICS and not 1024 <= settings.METRICS_PORT <= 65535:
        errors.append("METRICS_PORT must be between 1024 and 65535")
    
    # Validate rate limits
    if settings.DEEPSEEK_RATE_LIMIT <= 0:
        errors.append("DEEPSEEK_RATE_LIMIT must be positive")
    
    if settings.GEMINI_RATE_LIMIT <= 0:
        errors.append("GEMINI_RATE_LIMIT must be positive")
    
    # Validate cache TTL values
    ttl_fields = [
        "MARKET_ANALYSIS_CACHE_TTL",
        "STRATEGY_ANALYSIS_CACHE_TTL", 
        "RISK_ANALYSIS_CACHE_TTL",
        "NEWS_SENTIMENT_CACHE_TTL",
        "REDIS_CACHE_TTL"
    ]
    
    for field in ttl_fields:
        value = getattr(settings, field)
        if value <= 0:
            errors.append(f"{field} must be positive")
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
        raise ValueError(error_msg)


# Validate configuration on import
try:
    validate_configuration()
except ValueError as e:
    if settings.ENVIRONMENT != "development":
        raise e
    else:
        print(f"Warning: {e}")
        print("Continuing in development mode with warnings...")


# Export configuration for use in other modules
__all__ = ["settings", "get_settings", "validate_configuration", "Settings"]