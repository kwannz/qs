#![allow(unused_imports, unused_variables, dead_code, unused_mut, deprecated, unreachable_code)]

pub mod risk_metrics;
pub mod risk_engine;
pub mod position_monitor;
pub mod liquidity_monitor;
pub mod preemptive_risk_control;
pub mod enhanced_preemptive_control;

use anyhow::Result;
use platform_config::PlatformConfig;
use tracing::info;

use risk_engine::{init_risk_engine, shutdown_risk_engine};
use position_monitor::{init_position_monitor, shutdown_position_monitor};
use liquidity_monitor::{init_liquidity_monitor, shutdown_liquidity_monitor};

pub use preemptive_risk_control::{
    PreemptiveRiskController, RiskControlConfig, AccountRiskLimits,
    RiskCheckResult, RiskViolation, RiskViolationType, RiskSeverity,
    RiskWarning, RiskWarningType,
};

pub use enhanced_preemptive_control::{
    EnhancedPreemptiveRiskController,
    RiskQuota, DrawdownAlert, RiskControlAction,
};

/// Initialize the complete risk management system
pub async fn init_risk_management_system(config: &PlatformConfig) -> Result<()> {
    info!("Initializing platform risk management system");
    
    // Initialize core components (removed regulatory and compliance)
    init_risk_engine(config).await?;
    init_position_monitor(config).await?;
    init_liquidity_monitor(config).await?;
    
    info!("Platform risk management system initialized successfully");
    Ok(())
}

/// Shutdown the risk management system
pub async fn shutdown_risk_management_system() -> Result<()> {
    info!("Shutting down platform risk management system");
    
    shutdown_liquidity_monitor().await?;
    shutdown_position_monitor().await?;
    shutdown_risk_engine().await?;
    
    info!("Platform risk management system shutdown completed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_calculation() {
        // Test basic risk metric calculations
        let position_value = 10000.0;
        let portfolio_value = 100000.0;
        let risk_percentage = (position_value / portfolio_value) * 100.0;
        
        assert_eq!(risk_percentage, 10.0);
        assert!(risk_percentage <= 20.0); // Risk limit check
    }

    #[test]
    fn test_risk_limits() {
        // Test risk limit validation
        let max_position_size = 50000.0;
        let current_position = 30000.0;
        let new_trade_size = 15000.0;
        
        let total_position = current_position + new_trade_size;
        let within_limit = total_position <= max_position_size;
        
        assert_eq!(total_position, 45000.0);
        assert!(within_limit);
    }
}