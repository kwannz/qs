#![allow(unused_imports, unused_variables, dead_code, unused_mut, deprecated)]


//! 服务中间件
//! 
//! 提供认证、授权、限流、追踪、监控等横切关注点的中间件

use crate::services::{ServiceResult, ServiceError, RequestMetadata, ResponseMetadata};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug, instrument};

/// 中间件trait
#[async_trait]
pub trait Middleware: Send + Sync + std::fmt::Debug {
    /// 中间件名称
    fn name(&self) -> &str;
    
    /// 处理请求（前置处理）
    async fn before_request(
        &self,
        metadata: &mut RequestMetadata,
    ) -> ServiceResult<()>;
    
    /// 处理响应（后置处理）
    async fn after_response(
        &self,
        metadata: &RequestMetadata,
        response_metadata: &mut ResponseMetadata,
        success: bool,
    ) -> ServiceResult<()>;
    
    /// 中间件优先级（数值越小优先级越高）
    fn priority(&self) -> u8 {
        50
    }
}

/// 中间件链
#[derive(Debug)]
pub struct MiddlewareChain {
    middlewares: Vec<Box<dyn Middleware>>,
}

impl MiddlewareChain {
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
        }
    }
    
    /// 添加中间件
    pub fn add_middleware(&mut self, middleware: Box<dyn Middleware>) {
        self.middlewares.push(middleware);
        // 按优先级排序
        self.middlewares.sort_by(|a, b| a.priority().cmp(&b.priority()));
    }
    
    /// 执行前置处理
    pub async fn before_request(&self, metadata: &mut RequestMetadata) -> ServiceResult<()> {
        for middleware in &self.middlewares {
            middleware.before_request(metadata).await?;
        }
        Ok(())
    }
    
    /// 执行后置处理
    pub async fn after_response(
        &self,
        metadata: &RequestMetadata,
        response_metadata: &mut ResponseMetadata,
        success: bool,
    ) -> ServiceResult<()> {
        // 后置处理按相反顺序执行
        for middleware in self.middlewares.iter().rev() {
            middleware.after_response(metadata, response_metadata, success).await?;
        }
        Ok(())
    }
}

impl Default for MiddlewareChain {
    fn default() -> Self {
        Self::new()
    }
}

/// 认证中间件
#[derive(Debug)]
pub struct AuthenticationMiddleware {
    auth_service: Arc<dyn AuthenticationService>,
    skip_paths: Vec<String>,
}

/// 认证服务trait
#[async_trait]
pub trait AuthenticationService: Send + Sync {
    async fn validate_token(&self, token: &str) -> ServiceResult<UserContext>;
    async fn get_user_permissions(&self, user_id: &str) -> ServiceResult<Vec<Permission>>;
}

/// 用户上下文
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserContext {
    pub user_id: String,
    pub username: String,
    pub roles: Vec<String>,
    pub permissions: Vec<Permission>,
    pub session_id: Option<String>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
}

/// 权限
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Permission {
    pub resource: String,
    pub action: String,
    pub scope: Option<String>,
}

impl AuthenticationMiddleware {
    pub fn new(auth_service: Arc<dyn AuthenticationService>) -> Self {
        Self {
            auth_service,
            skip_paths: vec!["/health".to_string(), "/metrics".to_string()],
        }
    }
    
    pub fn with_skip_paths(mut self, paths: Vec<String>) -> Self {
        self.skip_paths = paths;
        self
    }
}

#[async_trait]
impl Middleware for AuthenticationMiddleware {
    fn name(&self) -> &str {
        "authentication"
    }
    
    fn priority(&self) -> u8 {
        10 // 高优先级
    }
    
    async fn before_request(&self, metadata: &mut RequestMetadata) -> ServiceResult<()> {
        // 检查是否需要跳过认证
        let path = metadata.labels.get("path").unwrap_or(&String::new());
        if self.skip_paths.iter().any(|skip_path| path.starts_with(skip_path)) {
            return Ok(());
        }
        
        // 获取认证令牌
        let token = metadata.labels.get("authorization")
            .ok_or_else(|| ServiceError::PermissionDenied {
                reason: "Missing authorization token".to_string(),
            })?;
        
        // 验证令牌
        let user_context = self.auth_service.validate_token(token).await?;
        
        // 将用户上下文添加到元数据中
        metadata.labels.insert(
            "user_id".to_string(), 
            user_context.user_id.clone()
        );
        metadata.labels.insert(
            "username".to_string(), 
            user_context.username.clone()
        );
        
        debug!("用户认证成功: {}", user_context.username);
        Ok(())
    }
    
    async fn after_response(
        &self,
        _metadata: &RequestMetadata,
        _response_metadata: &mut ResponseMetadata,
        _success: bool,
    ) -> ServiceResult<()> {
        // 认证中间件通常不需要后置处理
        Ok(())
    }
}

/// 授权中间件
#[derive(Debug)]
pub struct AuthorizationMiddleware {
    permissions_required: HashMap<String, Vec<Permission>>,
}

impl AuthorizationMiddleware {
    pub fn new() -> Self {
        Self {
            permissions_required: HashMap::new(),
        }
    }
    
    pub fn require_permission(&mut self, path: String, permission: Permission) {
        self.permissions_required
            .entry(path)
            .or_insert_with(Vec::new)
            .push(permission);
    }
}

#[async_trait]
impl Middleware for AuthorizationMiddleware {
    fn name(&self) -> &str {
        "authorization"
    }
    
    fn priority(&self) -> u8 {
        15 // 在认证之后
    }
    
    async fn before_request(&self, metadata: &mut RequestMetadata) -> ServiceResult<()> {
        let path = metadata.labels.get("path").unwrap_or(&String::new());
        
        // 检查是否需要特定权限
        if let Some(required_permissions) = self.permissions_required.get(path) {
            let user_id = metadata.labels.get("user_id")
                .ok_or_else(|| ServiceError::PermissionDenied {
                    reason: "User not authenticated".to_string(),
                })?;
            
            // 这里简化处理，实际应该从用户上下文中获取权限
            // 或者调用权限服务
            for required_permission in required_permissions {
                debug!("检查权限: {} on {}", required_permission.action, required_permission.resource);
                // 实际的权限检查逻辑
            }
        }
        
        Ok(())
    }
    
    async fn after_response(
        &self,
        _metadata: &RequestMetadata,
        _response_metadata: &mut ResponseMetadata,
        _success: bool,
    ) -> ServiceResult<()> {
        Ok(())
    }
}

impl Default for AuthorizationMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

/// 限流中间件
#[derive(Debug)]
pub struct RateLimitingMiddleware {
    global_limiter: Arc<crate::services::common::RateLimiter>,
    user_limiters: Arc<RwLock<HashMap<String, Arc<crate::services::common::RateLimiter>>>>,
    per_user_rate: u32,
    per_user_capacity: u32,
}

impl RateLimitingMiddleware {
    pub fn new(global_rate: u32, global_capacity: u32, per_user_rate: u32, per_user_capacity: u32) -> Self {
        Self {
            global_limiter: Arc::new(crate::services::common::RateLimiter::new(global_capacity, global_rate)),
            user_limiters: Arc::new(RwLock::new(HashMap::new())),
            per_user_rate,
            per_user_capacity,
        }
    }
    
    async fn get_user_limiter(&self, user_id: &str) -> Arc<crate::services::common::RateLimiter> {
        let mut limiters = self.user_limiters.write().await;
        limiters.entry(user_id.to_string())
            .or_insert_with(|| Arc::new(crate::services::common::RateLimiter::new(
                self.per_user_capacity, 
                self.per_user_rate
            )))
            .clone()
    }
}

#[async_trait]
impl Middleware for RateLimitingMiddleware {
    fn name(&self) -> &str {
        "rate_limiting"
    }
    
    fn priority(&self) -> u8 {
        20
    }
    
    async fn before_request(&self, metadata: &mut RequestMetadata) -> ServiceResult<()> {
        // 全局限流检查
        if !self.global_limiter.try_acquire(1).await {
            return Err(ServiceError::ResourceExhausted {
                resource: "global_rate_limit".to_string(),
            });
        }
        
        // 用户级限流检查
        if let Some(user_id) = metadata.labels.get("user_id") {
            let user_limiter = self.get_user_limiter(user_id).await;
            if !user_limiter.try_acquire(1).await {
                return Err(ServiceError::ResourceExhausted {
                    resource: format!("user_rate_limit:{}", user_id),
                });
            }
        }
        
        Ok(())
    }
    
    async fn after_response(
        &self,
        _metadata: &RequestMetadata,
        _response_metadata: &mut ResponseMetadata,
        _success: bool,
    ) -> ServiceResult<()> {
        Ok(())
    }
}

/// 追踪中间件
#[derive(Debug)]
pub struct TracingMiddleware {
    service_name: String,
}

impl TracingMiddleware {
    pub fn new(service_name: String) -> Self {
        Self { service_name }
    }
}

#[async_trait]
impl Middleware for TracingMiddleware {
    fn name(&self) -> &str {
        "tracing"
    }
    
    fn priority(&self) -> u8 {
        5 // 最高优先级
    }
    
    #[instrument(skip(self, metadata), fields(request_id = %metadata.request_id))]
    async fn before_request(&self, metadata: &mut RequestMetadata) -> ServiceResult<()> {
        info!(
            service = %self.service_name,
            request_id = %metadata.request_id,
            correlation_id = %metadata.correlation_id,
            "开始处理请求"
        );
        
        // 添加追踪标签
        metadata.labels.insert("trace_id".to_string(), metadata.correlation_id.clone());
        metadata.labels.insert("service".to_string(), self.service_name.clone());
        
        Ok(())
    }
    
    async fn after_response(
        &self,
        metadata: &RequestMetadata,
        response_metadata: &mut ResponseMetadata,
        success: bool,
    ) -> ServiceResult<()> {
        let status = if success { "success" } else { "error" };
        
        info!(
            service = %self.service_name,
            request_id = %metadata.request_id,
            status = %status,
            duration_us = %response_metadata.processing_time_us,
            "请求处理完成"
        );
        
        Ok(())
    }
}

/// 监控中间件
#[derive(Debug)]
pub struct MonitoringMiddleware {
    metrics: Arc<RwLock<crate::services::common::ServiceMetrics>>,
}

impl MonitoringMiddleware {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(crate::services::common::ServiceMetrics::new())),
        }
    }
    
    pub async fn get_metrics(&self) -> crate::services::common::ServiceMetrics {
        self.metrics.read().await.clone()
    }
}

#[async_trait]
impl Middleware for MonitoringMiddleware {
    fn name(&self) -> &str {
        "monitoring"
    }
    
    fn priority(&self) -> u8 {
        100 // 最低优先级，最后执行
    }
    
    async fn before_request(&self, metadata: &mut RequestMetadata) -> ServiceResult<()> {
        // 记录请求开始时间
        metadata.labels.insert(
            "start_time".to_string(),
            Instant::now().elapsed().as_micros().to_string(),
        );
        Ok(())
    }
    
    async fn after_response(
        &self,
        metadata: &RequestMetadata,
        response_metadata: &mut ResponseMetadata,
        success: bool,
    ) -> ServiceResult<()> {
        // 记录请求指标
        let mut metrics = self.metrics.write().await;
        metrics.record_request(response_metadata.processing_time_us, success);
        
        // 记录详细指标
        if let Some(path) = metadata.labels.get("path") {
            debug!(
                path = %path,
                duration_us = %response_metadata.processing_time_us,
                success = %success,
                "记录请求指标"
            );
        }
        
        Ok(())
    }
}

impl Default for MonitoringMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

/// 错误处理中间件
#[derive(Debug)]
pub struct ErrorHandlingMiddleware {
    enable_detailed_errors: bool,
}

impl ErrorHandlingMiddleware {
    pub fn new(enable_detailed_errors: bool) -> Self {
        Self { enable_detailed_errors }
    }
}

#[async_trait]
impl Middleware for ErrorHandlingMiddleware {
    fn name(&self) -> &str {
        "error_handling"
    }
    
    fn priority(&self) -> u8 {
        95 // 接近最后执行
    }
    
    async fn before_request(&self, _metadata: &mut RequestMetadata) -> ServiceResult<()> {
        Ok(())
    }
    
    async fn after_response(
        &self,
        metadata: &RequestMetadata,
        response_metadata: &mut ResponseMetadata,
        success: bool,
    ) -> ServiceResult<()> {
        if !success {
            error!(
                request_id = %metadata.request_id,
                "请求处理失败"
            );
            
            // 在生产环境中可能需要隐藏详细错误信息
            if !self.enable_detailed_errors {
                // 清理敏感信息
            }
        }
        
        Ok(())
    }
}

/// 缓存中间件
#[derive(Debug)]
pub struct CachingMiddleware {
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    default_ttl_secs: u64,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    data: Vec<u8>,
    expires_at: chrono::DateTime<chrono::Utc>,
}

impl CachingMiddleware {
    pub fn new(default_ttl_secs: u64) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            default_ttl_secs,
        }
    }
    
    pub async fn get(&self, key: &str) -> Option<Vec<u8>> {
        let cache = self.cache.read().await;
        if let Some(entry) = cache.get(key) {
            if chrono::Utc::now() < entry.expires_at {
                return Some(entry.data.clone());
            }
        }
        None
    }
    
    pub async fn set(&self, key: String, data: Vec<u8>, ttl_secs: Option<u64>) {
        let ttl = ttl_secs.unwrap_or(self.default_ttl_secs);
        let expires_at = chrono::Utc::now() + chrono::Duration::seconds(ttl as i64);
        
        let entry = CacheEntry { data, expires_at };
        
        let mut cache = self.cache.write().await;
        cache.insert(key, entry);
    }
    
    pub async fn cleanup_expired(&self) {
        let now = chrono::Utc::now();
        let mut cache = self.cache.write().await;
        cache.retain(|_, entry| entry.expires_at > now);
    }
}

#[async_trait]
impl Middleware for CachingMiddleware {
    fn name(&self) -> &str {
        "caching"
    }
    
    fn priority(&self) -> u8 {
        30
    }
    
    async fn before_request(&self, metadata: &mut RequestMetadata) -> ServiceResult<()> {
        // 生成缓存键
        let cache_key = format!(
            "{}:{}:{}",
            metadata.labels.get("method").unwrap_or(&"GET".to_string()),
            metadata.labels.get("path").unwrap_or(&"unknown".to_string()),
            metadata.labels.get("query").unwrap_or(&"".to_string())
        );
        
        // 检查缓存
        if let Some(_cached_data) = self.get(&cache_key).await {
            metadata.labels.insert("cache_key".to_string(), cache_key);
            metadata.labels.insert("cache_hit".to_string(), "true".to_string());
            debug!("缓存命中: {}", cache_key);
            // 这里可以直接返回缓存的响应，但需要请求处理器支持
        } else {
            metadata.labels.insert("cache_key".to_string(), cache_key);
            metadata.labels.insert("cache_hit".to_string(), "false".to_string());
        }
        
        Ok(())
    }
    
    async fn after_response(
        &self,
        metadata: &RequestMetadata,
        response_metadata: &mut ResponseMetadata,
        success: bool,
    ) -> ServiceResult<()> {
        // 更新缓存命中标记
        response_metadata.cache_hit = metadata.labels.get("cache_hit")
            .map(|s| s == "true")
            .unwrap_or(false);
        
        // 如果请求成功且不是缓存命中，存储到缓存
        if success && !response_metadata.cache_hit {
            if let Some(cache_key) = metadata.labels.get("cache_key") {
                // 这里需要响应数据，实际实现时需要与请求处理器协调
                debug!("存储到缓存: {}", cache_key);
            }
        }
        
        Ok(())
    }
}

/// 压缩中间件
#[derive(Debug)]
pub struct CompressionMiddleware {
    min_size: usize,
    compression_level: u32,
}

impl CompressionMiddleware {
    pub fn new(min_size: usize, compression_level: u32) -> Self {
        Self {
            min_size,
            compression_level,
        }
    }
}

#[async_trait]
impl Middleware for CompressionMiddleware {
    fn name(&self) -> &str {
        "compression"
    }
    
    fn priority(&self) -> u8 {
        90
    }
    
    async fn before_request(&self, metadata: &mut RequestMetadata) -> ServiceResult<()> {
        // 检查客户端是否支持压缩
        if let Some(accept_encoding) = metadata.labels.get("accept-encoding") {
            if accept_encoding.contains("gzip") {
                metadata.labels.insert("compression_supported".to_string(), "gzip".to_string());
            }
        }
        Ok(())
    }
    
    async fn after_response(
        &self,
        metadata: &RequestMetadata,
        _response_metadata: &mut ResponseMetadata,
        success: bool,
    ) -> ServiceResult<()> {
        if success {
            if let Some(compression) = metadata.labels.get("compression_supported") {
                debug!("响应将使用压缩: {}", compression);
                // 实际的压缩逻辑需要在响应处理器中实现
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct MockAuthService {
        should_succeed: AtomicBool,
    }

    impl MockAuthService {
        fn new(should_succeed: bool) -> Self {
            Self {
                should_succeed: AtomicBool::new(should_succeed),
            }
        }
    }

    #[async_trait]
    impl AuthenticationService for MockAuthService {
        async fn validate_token(&self, _token: &str) -> ServiceResult<UserContext> {
            if self.should_succeed.load(Ordering::SeqCst) {
                Ok(UserContext {
                    user_id: "test_user".to_string(),
                    username: "testuser".to_string(),
                    roles: vec!["user".to_string()],
                    permissions: vec![],
                    session_id: None,
                    expires_at: chrono::Utc::now() + chrono::Duration::hours(1),
                })
            } else {
                Err(ServiceError::PermissionDenied {
                    reason: "Invalid token".to_string(),
                })
            }
        }

        async fn get_user_permissions(&self, _user_id: &str) -> ServiceResult<Vec<Permission>> {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn test_middleware_chain() {
        let mut chain = MiddlewareChain::new();
        chain.add_middleware(Box::new(TracingMiddleware::new("test".to_string())));
        chain.add_middleware(Box::new(MonitoringMiddleware::new()));
        
        let mut metadata = RequestMetadata::default();
        let result = chain.before_request(&mut metadata).await;
        assert!(result.is_ok());
        
        let mut response_metadata = ResponseMetadata {
            request_id: metadata.request_id.clone(),
            timestamp: chrono::Utc::now(),
            processing_time_us: 1000,
            service_version: "1.0.0".to_string(),
            service_instance_id: "test".to_string(),
            retry_count: 0,
            cache_hit: false,
        };
        
        let result = chain.after_response(&metadata, &mut response_metadata, true).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_authentication_middleware() {
        let auth_service = Arc::new(MockAuthService::new(true));
        let middleware = AuthenticationMiddleware::new(auth_service);
        
        let mut metadata = RequestMetadata::default();
        metadata.labels.insert("path".to_string(), "/api/test".to_string());
        metadata.labels.insert("authorization".to_string(), "Bearer token123".to_string());
        
        let result = middleware.before_request(&mut metadata).await;
        assert!(result.is_ok());
        assert!(metadata.labels.contains_key("user_id"));
    }

    #[tokio::test]
    async fn test_rate_limiting_middleware() {
        let middleware = RateLimitingMiddleware::new(10, 10, 5, 5);
        let mut metadata = RequestMetadata::default();
        metadata.labels.insert("user_id".to_string(), "test_user".to_string());
        
        // 前几个请求应该成功
        for _ in 0..5 {
            let result = middleware.before_request(&mut metadata).await;
            assert!(result.is_ok());
        }
        
        // 超出限制后应该失败
        let result = middleware.before_request(&mut metadata).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_caching_middleware() {
        let middleware = CachingMiddleware::new(300);
        
        // 测试缓存存储
        middleware.set(
            "test_key".to_string(),
            b"test_data".to_vec(),
            Some(60),
        ).await;
        
        // 测试缓存获取
        let cached_data = middleware.get("test_key").await;
        assert!(cached_data.is_some());
        assert_eq!(cached_data.unwrap(), b"test_data");
        
        // 测试不存在的键
        let missing_data = middleware.get("missing_key").await;
        assert!(missing_data.is_none());
    }
}