"""
Cache Management System
缓存管理系统

Provides Redis-based caching with intelligent key management,
TTL optimization, and performance monitoring for AI responses.
"""

import asyncio
import json
import hashlib
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

import aioredis
import structlog
from pydantic import BaseModel

from config import settings


logger = structlog.get_logger(__name__)


class CacheKey(BaseModel):
    """Cache key structure with metadata."""
    prefix: str
    identifier: str
    version: str = "v1"
    
    def __str__(self) -> str:
        """Generate cache key string."""
        return f"{self.prefix}:{self.version}:{self.identifier}"
    
    @classmethod
    def market_analysis(cls, symbol: str, timeframe: str, analysis_type: str) -> "CacheKey":
        """Generate cache key for market analysis."""
        identifier = f"{symbol}:{timeframe}:{analysis_type}"
        return cls(prefix="market_analysis", identifier=identifier)
    
    @classmethod
    def strategy_optimization(cls, strategy_name: str, config_hash: str) -> "CacheKey":
        """Generate cache key for strategy optimization."""
        identifier = f"{strategy_name}:{config_hash}"
        return cls(prefix="strategy_opt", identifier=identifier)
    
    @classmethod
    def risk_assessment(cls, portfolio_hash: str, market_hash: str) -> "CacheKey":
        """Generate cache key for risk assessment."""
        identifier = f"{portfolio_hash}:{market_hash}"
        return cls(prefix="risk_assessment", identifier=identifier)
    
    @classmethod
    def news_sentiment(cls, symbols_hash: str, timeframe: str) -> "CacheKey":
        """Generate cache key for news sentiment."""
        identifier = f"{symbols_hash}:{timeframe}"
        return cls(prefix="news_sentiment", identifier=identifier)


class CacheEntry(BaseModel):
    """Cache entry with metadata."""
    key: str
    data: Dict[str, Any]
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    size_bytes: int = 0
    
    @classmethod
    def create(cls, key: str, data: Dict[str, Any], ttl_seconds: int) -> "CacheEntry":
        """Create cache entry with expiration."""
        now = datetime.utcnow()
        data_json = json.dumps(data, default=str)
        
        return cls(
            key=key,
            data=data,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
            size_bytes=len(data_json.encode('utf-8'))
        )
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        return datetime.utcnow() > self.expires_at
    
    def refresh_access(self):
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


class CacheStats(BaseModel):
    """Cache statistics for monitoring."""
    total_keys: int = 0
    total_size_bytes: int = 0
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    error_count: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.hit_count + self.miss_count
        if total_requests == 0:
            return 0.0
        return self.hit_count / total_requests
    
    @property
    def average_size_kb(self) -> float:
        """Calculate average entry size in KB."""
        if self.total_keys == 0:
            return 0.0
        return (self.total_size_bytes / 1024) / self.total_keys


class CacheManager:
    """
    Redis-based cache manager with intelligent features.
    
    Features:
    - Automatic key generation and management
    - TTL-based expiration with category-specific policies
    - Cache warming and pre-loading strategies
    - Performance monitoring and statistics
    - Memory usage optimization
    - Batch operations for efficiency
    - Circuit breaker integration
    """
    
    def __init__(self):
        """Initialize cache manager."""
        self.redis_url = settings.REDIS_URL
        self.redis_db = settings.REDIS_DB
        self.pool_size = settings.REDIS_POOL_SIZE
        self.default_ttl = settings.REDIS_CACHE_TTL
        
        self.redis_pool: Optional[aioredis.Redis] = None
        self.stats = CacheStats()
        self.is_connected = False
        
        # TTL policies for different cache types
        self.ttl_policies = {
            "market_analysis": settings.MARKET_ANALYSIS_CACHE_TTL,
            "strategy_opt": settings.STRATEGY_ANALYSIS_CACHE_TTL,
            "risk_assessment": settings.RISK_ANALYSIS_CACHE_TTL,
            "news_sentiment": settings.NEWS_SENTIMENT_CACHE_TTL,
        }
        
        logger.info("Cache manager initialized", redis_url=self.redis_url, db=self.redis_db)
    
    async def connect(self):
        """Establish Redis connection."""
        try:
            self.redis_pool = aioredis.from_url(
                self.redis_url,
                db=self.redis_db,
                password=settings.REDIS_PASSWORD,
                max_connections=self.pool_size,
                retry_on_timeout=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
                encoding='utf-8',
                decode_responses=True
            )
            
            # Test connection
            await self.redis_pool.ping()
            self.is_connected = True
            
            logger.info("Redis connection established", pool_size=self.pool_size)
            
        except Exception as e:
            self.is_connected = False
            logger.error("Failed to connect to Redis", error=str(e))
            raise
    
    async def close(self):
        """Close Redis connections."""
        if self.redis_pool:
            await self.redis_pool.close()
            self.is_connected = False
            logger.info("Redis connection closed")
    
    async def get(self, key: Union[str, CacheKey]) -> Optional[Dict[str, Any]]:
        """
        Get cached data by key.
        
        Args:
            key: Cache key (string or CacheKey object)
            
        Returns:
            Cached data or None if not found
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            key_str = str(key)
            
            # Get data from Redis
            cached_data = await self.redis_pool.get(key_str)
            
            if cached_data is None:
                self.stats.miss_count += 1
                logger.debug("Cache miss", key=key_str)
                return None
            
            # Parse cached entry
            try:
                entry_data = json.loads(cached_data)
                cache_entry = CacheEntry(**entry_data)
                
                # Check expiration
                if cache_entry.is_expired():
                    await self.delete(key)
                    self.stats.miss_count += 1
                    logger.debug("Cache expired", key=key_str)
                    return None
                
                # Update access stats
                cache_entry.refresh_access()
                await self._update_entry_metadata(cache_entry)
                
                self.stats.hit_count += 1
                logger.debug("Cache hit", key=key_str, access_count=cache_entry.access_count)
                
                return cache_entry.data
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Invalid cache entry format", key=key_str, error=str(e))
                await self.delete(key)
                self.stats.miss_count += 1
                return None
                
        except Exception as e:
            self.stats.error_count += 1
            logger.error("Cache get failed", key=str(key), error=str(e))
            return None
    
    async def set(
        self,
        key: Union[str, CacheKey],
        data: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        Set cached data with TTL.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl_seconds: Time to live in seconds (uses category default if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected:
            await self.connect()
        
        try:
            key_str = str(key)
            
            # Determine TTL
            if ttl_seconds is None:
                ttl_seconds = self._get_ttl_for_key(key_str)
            
            # Create cache entry
            cache_entry = CacheEntry.create(key_str, data, ttl_seconds)
            
            # Serialize entry
            entry_json = cache_entry.json()
            
            # Store in Redis with TTL
            await self.redis_pool.setex(
                key_str,
                ttl_seconds,
                entry_json
            )
            
            # Update stats
            self.stats.total_keys += 1
            self.stats.total_size_bytes += cache_entry.size_bytes
            
            logger.debug(
                "Cache set",
                key=key_str,
                ttl=ttl_seconds,
                size_bytes=cache_entry.size_bytes
            )
            
            return True
            
        except Exception as e:
            self.stats.error_count += 1
            logger.error("Cache set failed", key=str(key), error=str(e))
            return False
    
    async def delete(self, key: Union[str, CacheKey]) -> bool:
        """
        Delete cached data.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if deleted, False otherwise
        """
        if not self.is_connected:
            return False
        
        try:
            key_str = str(key)
            result = await self.redis_pool.delete(key_str)
            
            if result:
                self.stats.eviction_count += 1
                logger.debug("Cache delete", key=key_str)
            
            return bool(result)
            
        except Exception as e:
            self.stats.error_count += 1
            logger.error("Cache delete failed", key=str(key), error=str(e))
            return False
    
    async def exists(self, key: Union[str, CacheKey]) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if exists, False otherwise
        """
        if not self.is_connected:
            return False
        
        try:
            result = await self.redis_pool.exists(str(key))
            return bool(result)
        except Exception as e:
            logger.error("Cache exists check failed", key=str(key), error=str(e))
            return False
    
    async def get_many(self, keys: List[Union[str, CacheKey]]) -> Dict[str, Dict[str, Any]]:
        """
        Get multiple cached items efficiently.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dict mapping keys to cached data
        """
        if not self.is_connected:
            await self.connect()
        
        results = {}
        key_strings = [str(key) for key in keys]
        
        try:
            # Use Redis pipeline for efficient batch operation
            pipe = self.redis_pool.pipeline()
            for key_str in key_strings:
                pipe.get(key_str)
            
            cached_values = await pipe.execute()
            
            for i, (key, cached_data) in enumerate(zip(keys, cached_values)):
                key_str = str(key)
                
                if cached_data is not None:
                    try:
                        entry_data = json.loads(cached_data)
                        cache_entry = CacheEntry(**entry_data)
                        
                        if not cache_entry.is_expired():
                            results[key_str] = cache_entry.data
                            self.stats.hit_count += 1
                        else:
                            await self.delete(key)
                            self.stats.miss_count += 1
                    except (json.JSONDecodeError, ValueError):
                        await self.delete(key)
                        self.stats.miss_count += 1
                else:
                    self.stats.miss_count += 1
            
            logger.debug("Cache get_many", keys_requested=len(keys), keys_found=len(results))
            
        except Exception as e:
            self.stats.error_count += 1
            logger.error("Cache get_many failed", error=str(e))
        
        return results
    
    async def set_many(
        self,
        items: Dict[Union[str, CacheKey], Dict[str, Any]],
        ttl_seconds: Optional[int] = None
    ) -> int:
        """
        Set multiple cached items efficiently.
        
        Args:
            items: Dict mapping keys to data
            ttl_seconds: Time to live for all items
            
        Returns:
            Number of items successfully cached
        """
        if not self.is_connected:
            await self.connect()
        
        success_count = 0
        
        try:
            pipe = self.redis_pool.pipeline()
            
            for key, data in items.items():
                key_str = str(key)
                ttl = ttl_seconds or self._get_ttl_for_key(key_str)
                
                cache_entry = CacheEntry.create(key_str, data, ttl)
                entry_json = cache_entry.json()
                
                pipe.setex(key_str, ttl, entry_json)
                self.stats.total_size_bytes += cache_entry.size_bytes
            
            results = await pipe.execute()
            success_count = sum(1 for result in results if result)
            
            self.stats.total_keys += success_count
            
            logger.debug("Cache set_many", items_set=success_count, items_requested=len(items))
            
        except Exception as e:
            self.stats.error_count += 1
            logger.error("Cache set_many failed", error=str(e))
        
        return success_count
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear cache entries matching pattern.
        
        Args:
            pattern: Redis pattern (e.g., "market_analysis:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.is_connected:
            return 0
        
        try:
            keys = []
            async for key in self.redis_pool.scan_iter(match=pattern, count=100):
                keys.append(key)
            
            if keys:
                deleted = await self.redis_pool.delete(*keys)
                self.stats.eviction_count += deleted
                
                logger.info("Cache pattern cleared", pattern=pattern, keys_deleted=deleted)
                return deleted
            
        except Exception as e:
            self.stats.error_count += 1
            logger.error("Cache pattern clear failed", pattern=pattern, error=str(e))
        
        return 0
    
    async def warm_cache(self, warm_functions: List[callable]) -> Dict[str, int]:
        """
        Warm cache with commonly accessed data.
        
        Args:
            warm_functions: List of async functions that populate cache
            
        Returns:
            Dict with warming results
        """
        results = {}
        
        for func in warm_functions:
            try:
                func_name = func.__name__
                start_time = time.time()
                
                items_cached = await func()
                duration = time.time() - start_time
                
                results[func_name] = {
                    "items_cached": items_cached,
                    "duration_seconds": duration,
                    "success": True
                }
                
                logger.info(
                    "Cache warmed",
                    function=func_name,
                    items=items_cached,
                    duration=duration
                )
                
            except Exception as e:
                results[func_name] = {
                    "error": str(e),
                    "success": False
                }
                
                logger.error("Cache warming failed", function=func_name, error=str(e))
        
        return results
    
    def _get_ttl_for_key(self, key: str) -> int:
        """Get appropriate TTL for key based on prefix."""
        for prefix, ttl in self.ttl_policies.items():
            if key.startswith(prefix):
                return ttl
        return self.default_ttl
    
    async def _update_entry_metadata(self, cache_entry: CacheEntry):
        """Update cache entry metadata (access count, last accessed)."""
        try:
            # Only update metadata periodically to avoid overhead
            if cache_entry.access_count % 10 == 0:  # Update every 10 accesses
                entry_json = cache_entry.json()
                remaining_ttl = await self.redis_pool.ttl(cache_entry.key)
                
                if remaining_ttl > 0:
                    await self.redis_pool.setex(
                        cache_entry.key,
                        remaining_ttl,
                        entry_json
                    )
        except Exception as e:
            logger.debug("Failed to update cache entry metadata", error=str(e))
    
    async def health_check(self) -> bool:
        """
        Check Redis connection health.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            if not self.is_connected:
                await self.connect()
            
            await self.redis_pool.ping()
            return True
            
        except Exception as e:
            logger.error("Cache health check failed", error=str(e))
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring.
        
        Returns:
            Dict containing cache statistics
        """
        return {
            "connected": self.is_connected,
            "hit_rate": self.stats.hit_rate,
            "total_keys": self.stats.total_keys,
            "total_size_mb": self.stats.total_size_bytes / (1024 * 1024),
            "hit_count": self.stats.hit_count,
            "miss_count": self.stats.miss_count,
            "error_count": self.stats.error_count,
            "eviction_count": self.stats.eviction_count,
            "average_size_kb": self.stats.average_size_kb,
            "ttl_policies": self.ttl_policies
        }


# Utility functions for common cache operations

def generate_hash(data: Union[Dict, List, str]) -> str:
    """Generate consistent hash for cache key generation."""
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, sort_keys=True, default=str)
    else:
        data_str = str(data)
    
    return hashlib.md5(data_str.encode()).hexdigest()[:16]


async def cache_with_fallback(
    cache_manager: CacheManager,
    key: Union[str, CacheKey],
    fallback_func: callable,
    ttl_seconds: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Get data from cache or fallback to function if not found.
    
    Args:
        cache_manager: Cache manager instance
        key: Cache key
        fallback_func: Async function to call if cache miss
        ttl_seconds: Cache TTL
        **kwargs: Arguments for fallback function
        
    Returns:
        Data from cache or fallback function
    """
    # Try cache first
    cached_data = await cache_manager.get(key)
    if cached_data is not None:
        return cached_data
    
    # Fallback to function
    try:
        data = await fallback_func(**kwargs)
        
        # Cache the result
        await cache_manager.set(key, data, ttl_seconds)
        
        return data
        
    except Exception as e:
        logger.error("Cache fallback failed", key=str(key), error=str(e))
        raise