use cache::{
    CacheConfig, CacheManager, CacheProvider,
    distributed::RedisCache,
    local::MemoryCache,
    layered::{LayeredCache, LayeredStrategy},
    metrics::{MetricsCache, MetricsConfig},
};
use redis::Client;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::time::sleep;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct TestData {
    id: u32,
    name: String,
    value: f64,
}

impl TestData {
    fn new(id: u32, name: &str, value: f64) -> Self {
        Self {
            id,
            name: name.to_string(),
            value,
        }
    }
}

/// Test configuration for Redis (skipped if Redis not available)
fn redis_available() -> bool {
    std::env::var("REDIS_URL").is_ok() || 
    std::process::Command::new("redis-cli")
        .arg("ping")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn create_test_config() -> CacheConfig {
    CacheConfig {
        redis_url: std::env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string()),
        max_connections: 5,
        connection_timeout: Duration::from_secs(1),
        local_cache_size: 100,
        local_cache_ttl: Duration::from_secs(10),
        enable_compression: true,
        compression_threshold: 100,
        key_prefix: Some("test".to_string()),
    }
}

#[tokio::test]
async fn test_memory_cache_integration() {
    let config = create_test_config();
    let cache = MemoryCache::new(config);
    
    let test_data = TestData::new(1, "test", 123.45);
    let key = "memory:test:1";
    
    // Test basic operations
    assert!(cache.set(key, &test_data, None).await.is_ok());
    
    let result: Option<TestData> = cache.get(key).await.unwrap();
    assert_eq!(result, Some(test_data.clone()));
    
    assert!(cache.exists(key).await.unwrap());
    assert!(cache.delete(key).await.unwrap());
    
    let result: Option<TestData> = cache.get(key).await.unwrap();
    assert_eq!(result, None);
}

#[tokio::test]
#[ignore] // Requires Redis
async fn test_redis_cache_integration() {
    if !redis_available() {
        println!("Skipping Redis test - Redis not available");
        return;
    }
    
    let config = create_test_config();
    let client = Client::open(config.redis_url.as_str()).unwrap();
    let cache = RedisCache::new(client, config).await.unwrap();
    
    let test_data = TestData::new(2, "redis_test", 67.89);
    let key = "redis:test:2";
    
    // Clean up first
    let _ = cache.delete(key).await;
    
    // Test basic operations
    assert!(cache.set(key, &test_data, Some(Duration::from_secs(30))).await.is_ok());
    
    let result: Option<TestData> = cache.get(key).await.unwrap();
    assert_eq!(result, Some(test_data.clone()));
    
    assert!(cache.exists(key).await.unwrap());
    
    // Test multi operations
    let mut items = HashMap::new();
    items.insert("redis:multi:1", TestData::new(3, "multi1", 1.0));
    items.insert("redis:multi:2", TestData::new(4, "multi2", 2.0));
    items.insert("redis:multi:3", TestData::new(5, "multi3", 3.0));
    
    // Clean up multi keys
    for key in items.keys() {
        let _ = cache.delete(key).await;
    }
    
    assert!(cache.set_multi(&items, Some(Duration::from_secs(30))).await.is_ok());
    
    let keys: Vec<&str> = items.keys().copied().collect();
    let results: HashMap<String, TestData> = cache.get_multi(&keys).await.unwrap();
    assert_eq!(results.len(), 3);
    
    // Clean up
    cache.delete(key).await.unwrap();
    for key in items.keys() {
        let _ = cache.delete(key).await;
    }
}

#[tokio::test]
async fn test_layered_cache_integration() {
    let local_config = create_test_config();
    let cache = LayeredCache::new_single_tier(local_config, LayeredStrategy::WriteThrough);
    
    let test_data = TestData::new(6, "layered_test", 98.76);
    let key = "layered:test:6";
    
    // Test layered operations
    assert!(cache.set(key, &test_data, Some(Duration::from_secs(5))).await.is_ok());
    
    let result: Option<TestData> = cache.get(key).await.unwrap();
    assert_eq!(result, Some(test_data.clone()));
    
    // Test multi operations
    let mut items = HashMap::new();
    items.insert("layered:multi:1", TestData::new(7, "layer1", 10.0));
    items.insert("layered:multi:2", TestData::new(8, "layer2", 20.0));
    
    assert!(cache.set_multi(&items, None).await.is_ok());
    
    let keys: Vec<&str> = items.keys().copied().collect();
    let results: HashMap<String, TestData> = cache.get_multi(&keys).await.unwrap();
    assert_eq!(results.len(), 2);
    
    // Test delete
    assert!(cache.delete(key).await.unwrap());
    let result: Option<TestData> = cache.get(key).await.unwrap();
    assert_eq!(result, None);
}

#[tokio::test]
#[ignore] // Requires Redis
async fn test_two_tier_layered_cache() {
    if !redis_available() {
        println!("Skipping two-tier test - Redis not available");
        return;
    }
    
    let local_config = create_test_config();
    let redis_config = create_test_config();
    
    let cache = LayeredCache::new_two_tier(
        local_config,
        redis_config,
        LayeredStrategy::WriteThrough,
    ).await.unwrap();
    
    let test_data = TestData::new(9, "two_tier", 54.32);
    let key = "twotier:test:9";
    
    // Clean up first
    let _ = cache.delete(key).await;
    
    // Set value (should go to both L1 and L2)
    assert!(cache.set(key, &test_data, Some(Duration::from_secs(30))).await.is_ok());
    
    // Get value (should hit L1)
    let result: Option<TestData> = cache.get(key).await.unwrap();
    assert_eq!(result, Some(test_data.clone()));
    
    // Clean up
    cache.delete(key).await.unwrap();
}

#[tokio::test]
async fn test_metrics_cache_integration() {
    let config = create_test_config();
    let memory_cache = cache::CacheProviderType::Memory(MemoryCache::new(config));
    
    let metrics_config = MetricsConfig {
        track_per_key_metrics: true,
        track_operation_timing: true,
        track_memory_usage: true,
        ..Default::default()
    };
    
    let cache = MetricsCache::wrap(memory_cache, metrics_config);
    
    let test_data = TestData::new(10, "metrics_test", 11.11);
    let key = "metrics:test:10";
    
    // Perform operations to generate metrics
    assert!(cache.set(key, &test_data, None).await.is_ok());
    
    let result: Option<TestData> = cache.get(key).await.unwrap(); // hit
    assert_eq!(result, Some(test_data.clone()));
    
    let _: Option<TestData> = cache.get("nonexistent").await.unwrap(); // miss
    
    assert!(cache.delete(key).await.unwrap());
    
    // Check metrics
    let stats = cache.get_detailed_stats().await;
    assert_eq!(stats.basic.hits, 1);
    assert_eq!(stats.basic.misses, 1);
    assert_eq!(stats.sets, 1);
    assert_eq!(stats.deletes, 1);
    assert_eq!(stats.basic.hit_rate, 50.0);
    
    // Check per-key metrics
    let top_keys = cache.get_top_keys(5).await;
    assert!(!top_keys.is_empty());
    
    // Check operation latencies
    let latencies = cache.get_operation_latencies().await;
    assert!(latencies.contains_key("get"));
    assert!(latencies.contains_key("set"));
    
    // Check efficiency metrics
    let efficiency = cache.get_efficiency_metrics().await;
    assert_eq!(efficiency.hit_rate, 50.0);
    assert!(efficiency.average_get_latency_micros > 0.0);
}

#[tokio::test]
async fn test_cache_manager_integration() {
    let config = create_test_config();
    let cache_manager = CacheManager::new_local_only(config);
    
    let test_data = TestData::new(11, "manager_test", 22.22);
    let key = "manager:test:11";
    
    // Test CacheManager operations
    assert!(cache_manager.set(key, &test_data, Some(Duration::from_secs(10))).await.is_ok());
    
    let result: Option<TestData> = cache_manager.get(key).await.unwrap();
    assert_eq!(result, Some(test_data.clone()));
    
    assert!(cache_manager.delete(key).await.unwrap());
    
    let result: Option<TestData> = cache_manager.get(key).await.unwrap();
    assert_eq!(result, None);
    
    // Test cache statistics
    let stats = cache_manager.get_stats().await;
    assert!(stats.total_operations > 0);
    assert!(stats.hit_rate >= 0.0);
}

#[tokio::test]
#[ignore] // Requires Redis
async fn test_cache_manager_with_redis() {
    if !redis_available() {
        println!("Skipping CacheManager Redis test - Redis not available");
        return;
    }
    
    let config = create_test_config();
    let cache_manager = CacheManager::new_with_redis(config).await.unwrap();
    
    let test_data = TestData::new(12, "redis_manager", 33.33);
    let key = "redis_manager:test:12";
    
    // Clean up first
    let _ = cache_manager.delete(key).await;
    
    // Test with both local and Redis
    assert!(cache_manager.set(key, &test_data, Some(Duration::from_secs(30))).await.is_ok());
    
    let result: Option<TestData> = cache_manager.get(key).await.unwrap();
    assert_eq!(result, Some(test_data.clone()));
    
    // Clean up
    cache_manager.delete(key).await.unwrap();
    
    // Test statistics
    let stats = cache_manager.get_stats().await;
    assert!(stats.total_operations > 0);
}

#[tokio::test]
async fn test_ttl_expiration_integration() {
    let config = create_test_config();
    let cache = MemoryCache::new(config);
    
    let test_data = TestData::new(13, "ttl_test", 44.44);
    let key = "ttl:test:13";
    
    // Set with very short TTL
    assert!(cache.set(key, &test_data, Some(Duration::from_millis(100))).await.is_ok());
    
    // Should exist immediately
    assert!(cache.exists(key).await.unwrap());
    
    // Wait for expiration
    sleep(Duration::from_millis(200)).await;
    
    // Should be expired
    let result: Option<TestData> = cache.get(key).await.unwrap();
    assert_eq!(result, None);
}

#[tokio::test]
async fn test_compression_integration() {
    let mut config = create_test_config();
    config.enable_compression = true;
    config.compression_threshold = 50; // Low threshold to ensure compression
    
    if redis_available() {
        let client = Client::open(config.redis_url.as_str()).unwrap();
        let cache = RedisCache::new(client, config).await.unwrap();
        
        // Create large data that should be compressed
        let large_data = TestData::new(14, &"x".repeat(200), 55.55);
        let key = "compression:test:14";
        
        // Clean up first
        let _ = cache.delete(key).await;
        
        assert!(cache.set(key, &large_data, Some(Duration::from_secs(30))).await.is_ok());
        
        let result: Option<TestData> = cache.get(key).await.unwrap();
        assert_eq!(result, Some(large_data));
        
        // Clean up
        cache.delete(key).await.unwrap();
    }
}

#[tokio::test]
async fn test_error_handling_integration() {
    // Test with invalid Redis URL to trigger errors
    let mut config = create_test_config();
    config.redis_url = "redis://invalid_host:9999".to_string();
    config.connection_timeout = Duration::from_millis(100);
    
    let client = Client::open(config.redis_url.as_str()).unwrap();
    
    // This should fail due to invalid host
    let result = RedisCache::new(client, config).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_concurrent_operations() {
    let config = create_test_config();
    let cache = Arc::new(MemoryCache::new(config));
    
    let mut handles = Vec::new();
    
    // Spawn multiple concurrent operations
    for i in 0..10 {
        let cache_clone = Arc::clone(&cache);
        let handle = tokio::spawn(async move {
            let test_data = TestData::new(i, &format!("concurrent_{i}"), i as f64);
            let key = format!("concurrent:test:{i}");
            
            // Set and get concurrently
            cache_clone.set(&key, &test_data, None).await.unwrap();
            let result: Option<TestData> = cache_clone.get(&key).await.unwrap();
            assert_eq!(result, Some(test_data));
            
            cache_clone.delete(&key).await.unwrap();
        });
        handles.push(handle);
    }
    
    // Wait for all operations to complete
    for handle in handles {
        handle.await.unwrap();
    }
}

#[tokio::test]
async fn test_key_prefix_integration() {
    let mut config = create_test_config();
    config.key_prefix = Some("integration_test".to_string());
    
    let cache_manager = CacheManager::new_local_only(config);
    
    let test_data = TestData::new(15, "prefix_test", 66.66);
    let key = "prefix:test:15"; // Will become "integration_test:prefix:test:15"
    
    assert!(cache_manager.set(key, &test_data, None).await.is_ok());
    
    let result: Option<TestData> = cache_manager.get(key).await.unwrap();
    assert_eq!(result, Some(test_data));
    
    cache_manager.delete(key).await.unwrap();
}

#[tokio::test]
async fn test_memory_pressure_simulation() {
    let mut config = create_test_config();
    config.local_cache_size = 5; // Very small cache to trigger evictions
    
    let cache = MemoryCache::new(config);
    
    // Fill cache beyond capacity
    for i in 0..10 {
        let test_data = TestData::new(i, &format!("pressure_{i}"), i as f64);
        let key = format!("pressure:test:{i}");
        
        cache.set(&key, &test_data, None).await.unwrap();
    }
    
    // Give time for evictions to occur
    sleep(Duration::from_millis(100)).await;
    
    // Some keys should have been evicted
    let mut found_keys = 0;
    for i in 0..10 {
        let key = format!("pressure:test:{i}");
        if cache.exists(&key).await.unwrap() {
            found_keys += 1;
        }
    }
    
    // Should have fewer keys than we inserted due to capacity limits
    assert!(found_keys <= 5);
}

#[tokio::test]
async fn test_serialization_error_handling() {
    let config = create_test_config();
    let cache = MemoryCache::new(config);
    
    let test_data = TestData::new(16, "serialization_test", 77.77);
    let key = "serialization:test:16";
    
    // Set valid data
    cache.set(key, &test_data, None).await.unwrap();
    
    // Try to get as wrong type - should handle gracefully
    let result: Result<Option<String>, _> = cache.get(key).await;
    
    // Should either return None or error, but not panic
    match result {
        Ok(None) => {
            // Key was removed due to serialization error - acceptable behavior
        }
        Err(_) => {
            // Serialization error - also acceptable
        }
        Ok(Some(_)) => {
            panic!("Unexpected successful deserialization of wrong type");
        }
    }
}

#[tokio::test]
async fn test_cache_flush_integration() {
    let config = create_test_config();
    let cache = MemoryCache::new(config);
    
    // Set multiple keys
    for i in 0..5 {
        let test_data = TestData::new(i + 20, &format!("flush_{i}"), (i + 20) as f64);
        let key = format!("flush:test:{i}");
        cache.set(&key, &test_data, None).await.unwrap();
    }
    
    // Verify keys exist
    for i in 0..5 {
        let key = format!("flush:test:{i}");
        assert!(cache.exists(&key).await.unwrap());
    }
    
    // Flush cache
    cache.flush().await.unwrap();
    
    // Verify all keys are gone
    for i in 0..5 {
        let key = format!("flush:test:{i}");
        assert!(!cache.exists(&key).await.unwrap());
    }
}