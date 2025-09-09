pub mod crypto;
pub mod certificates;
pub mod audit;

pub use crypto::*;
pub use certificates::*;
pub use audit::*;

use anyhow::Result;

/// Initialize the security module
pub async fn init_security() -> Result<()> {
    tracing::info!("Initializing platform security module");
    
    // Initialize crypto subsystem
    crypto::init_crypto()?;
    
    // Initialize certificate manager
    certificates::init_certificate_manager().await?;
    
    tracing::info!("Platform security module initialized successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_security_init() {
        let result = init_security().await;
        assert!(result.is_ok());
    }
}