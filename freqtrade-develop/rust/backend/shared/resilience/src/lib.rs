pub mod retry;
pub mod timeout;
pub mod rate_limiter;
pub mod health_check;
pub mod redis_cluster;
pub mod outbox; // Sprint 1: Outbox pattern
pub mod health_aggregator; // Sprint 5: Health aggregation

#[allow(ambiguous_glob_reexports)]
pub use retry::*;
#[allow(ambiguous_glob_reexports)]
pub use timeout::*;
#[allow(ambiguous_glob_reexports)]
pub use rate_limiter::*;
#[allow(ambiguous_glob_reexports)]
pub use health_check::*;
#[allow(ambiguous_glob_reexports)]
pub use redis_cluster::*;
pub use outbox::*; // Sprint 1: Outbox pattern
pub use health_aggregator::*; // Sprint 5: Health aggregation

use anyhow::Result;

/// Initialize the resilience module
pub async fn init_resilience() -> Result<()> {
    tracing::info!("Initializing platform resilience module");
    
    // Initialize any global resilience patterns here
    
    tracing::info!("Platform resilience module initialized successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_resilience_init() {
        let result = init_resilience().await;
        assert!(result.is_ok());
    }
}