//! Comprehensive factor analysis library
//!
//! This module provides a complete factor management system including:
//! - Factor data models and structures
//! - Database operations through repository pattern
//! - Factor calculation algorithms and engines
//! - Unified service API for factor operations
//! - Integration with existing batch processing and dynamic selection
//!
//! # Examples
//!
//! ## Creating and using a factor
//!
//! ```rust
//! use shared_analytics::factors::{Factor, FactorCategory, FactorService, FactorServiceConfig};
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create a factor service
//!     let config = FactorServiceConfig::default();
//!     let data_provider = Arc::new(MockMarketDataProvider);
//!     let service = FactorService::new(config, data_provider).await?;
//!
//!     // Create a simple moving average factor
//!     let sma_factor = Factor::simple_moving_average(20);
//!     let factor_id = service.create_factor(sma_factor).await?;
//!
//!     // Calculate factor for a symbol
//!     let result = service.calculate_factor_point(
//!         &factor_id,
//!         "AAPL",
//!         chrono::Utc::now(),
//!     ).await?;
//!
//!     println!("Factor value: {}", result.value);
//!     Ok(())
//! }
//! ```
//!
//! ## Working with factor universes
//!
//! ```rust
//! use shared_analytics::factors::{FactorUniverse, FactorScreeningCriteria};
//!
//! // Create a factor universe
//! let mut universe = FactorUniverse::new("Technical Factors".to_string());
//! universe.add_factor(Factor::simple_moving_average(20));
//! universe.add_factor(Factor::rsi(14));
//!
//! // Screen factors based on criteria
//! let criteria = FactorScreeningCriteria {
//!     categories: Some(vec![FactorCategory::Technical]),
//!     min_sharpe: Some(0.5),
//!     ..Default::default()
//! };
//!
//! let filtered_factors = universe.screen_factors(&criteria);
//! ```

pub mod models;
pub mod repository;
pub mod calculator;
pub mod service;

// Re-export commonly used types for convenience
pub use models::{
    // Core factor types
    Factor, FactorCategory, FactorParameters, FactorParameterValue, FactorMetadata,
    ComputationCost, DataField, DataRequirements, UpdateFrequency,
    
    // Factor results and time series
    FactorResult, FactorResultMetadata, FactorTimeSeries, FactorDataPoint, 
    TimeSeriesStatistics,
    
    // Performance metrics
    FactorPerformanceMetrics,
    
    // Factor universe and correlation
    FactorUniverse, FactorCorrelationMatrix, FactorScreeningCriteria,
    
    // Batch processing
    FactorBatchRequest, FactorBatchStatus, BatchStatusType, BatchPriority,
    
    // Computation context and errors
    FactorComputationContext, FactorError,
};

pub use repository::{
    FactorRepository, InMemoryFactorRepository, FactorRepositoryFactory,
};

pub use calculator::{
    FactorCalculator, FactorCalculationEngine, MarketDataProvider, MathUtils, OhlcData,
};

pub use service::{
    FactorService, FactorServiceConfig, ServiceHealth,
};

// Integration with existing factor processing modules
use crate::factor_batch_processing::{
    FactorBatchProcessor, BatchProcessorConfig, BatchProcessingJob, JobPriority, JobStatus,
};
use crate::dynamic_factor_selection::{
    DynamicFactorSelector, DynamicFactorConfig, FactorAnalyzer, WeightOptimizer,
    FactorMetrics, WeightAllocation, RegimeState,
};

/// Factory for creating integrated factor services that leverage existing modules
pub struct IntegratedFactorServiceFactory;

impl IntegratedFactorServiceFactory {
    /// Create a factor service that integrates with existing batch processor
    pub async fn create_with_batch_processor(
        service_config: FactorServiceConfig,
        batch_config: BatchProcessorConfig,
        data_provider: Arc<dyn MarketDataProvider>,
    ) -> anyhow::Result<(FactorService, FactorBatchProcessor)> {
        let factor_service = FactorService::new(service_config, data_provider).await?;
        let batch_processor = FactorBatchProcessor::new(batch_config)?;
        
        Ok((factor_service, batch_processor))
    }

    /// Create a factor service with dynamic factor selection capabilities
    pub async fn create_with_dynamic_selection(
        service_config: FactorServiceConfig,
        dynamic_config: DynamicFactorConfig,
        data_provider: Arc<dyn MarketDataProvider>,
        risk_model: Arc<dyn crate::dynamic_factor_selection::RiskModel>,
        transaction_cost_model: Arc<dyn crate::dynamic_factor_selection::TransactionCostModel>,
    ) -> anyhow::Result<(FactorService, DynamicFactorSelector)> {
        let factor_service = FactorService::new(service_config, data_provider).await?;
        let dynamic_selector = DynamicFactorSelector::new(
            dynamic_config,
            risk_model,
            transaction_cost_model,
        );
        
        Ok((factor_service, dynamic_selector))
    }

    /// Create a fully integrated factor service with all capabilities
    pub async fn create_full_integrated(
        service_config: FactorServiceConfig,
        batch_config: BatchProcessorConfig,
        dynamic_config: DynamicFactorConfig,
        data_provider: Arc<dyn MarketDataProvider>,
        risk_model: Arc<dyn crate::dynamic_factor_selection::RiskModel>,
        transaction_cost_model: Arc<dyn crate::dynamic_factor_selection::TransactionCostModel>,
    ) -> anyhow::Result<IntegratedFactorService> {
        let factor_service = FactorService::new(service_config, data_provider).await?;
        let batch_processor = FactorBatchProcessor::new(batch_config)?;
        let dynamic_selector = DynamicFactorSelector::new(
            dynamic_config,
            risk_model,
            transaction_cost_model,
        );
        
        Ok(IntegratedFactorService {
            core_service: factor_service,
            batch_processor,
            dynamic_selector,
        })
    }
}

/// Integrated factor service that combines all factor-related functionality
pub struct IntegratedFactorService {
    pub core_service: FactorService,
    pub batch_processor: FactorBatchProcessor,
    pub dynamic_selector: DynamicFactorSelector,
}

impl IntegratedFactorService {
    /// Run a comprehensive factor analysis workflow
    pub async fn run_factor_analysis_workflow(
        &self,
        universe_id: &str,
        symbols: &[String],
        start_date: chrono::DateTime<chrono::Utc>,
        end_date: chrono::DateTime<chrono::Utc>,
    ) -> anyhow::Result<FactorAnalysisReport> {
        // 1. Get factor universe
        let universe = self.core_service.get_factor_universe(universe_id).await?
            .ok_or_else(|| FactorError::UnknownFactor(format!("Universe not found: {}", universe_id)))?;

        // 2. Calculate factors for all symbols using batch processing
        let factor_ids: Vec<String> = universe.factors.iter().map(|f| f.id.clone()).collect();
        let batch_id = self.core_service.calculate_factors_batch(
            &factor_ids,
            symbols,
            start_date,
            end_date,
            BatchPriority::High,
        ).await?;

        // 3. Wait for batch completion (in real implementation, you'd poll or use callbacks)
        // For now, we'll simulate completion
        
        // 4. Gather factor metrics for dynamic selection
        let mut factor_metrics = std::collections::HashMap::new();
        for factor in &universe.factors {
            // In real implementation, calculate actual metrics from results
            let metrics = FactorMetrics {
                factor_id: factor.id.clone(),
                ic: 0.05, // Dummy values
                rank_ic: 0.04,
                ic_ir: 1.2,
                t_stat: 2.1,
                p_value: 0.03,
                turnover: 0.8,
                decay_rate: 0.02,
                sharpe_ratio: 0.6,
                max_drawdown: -0.12,
                correlation_to_benchmark: 0.1,
                regime_stability: 0.7,
                last_updated: chrono::Utc::now(),
            };
            factor_metrics.insert(factor.id.clone(), metrics);
        }

        // 5. Run dynamic factor selection
        let selected_factors = self.dynamic_selector.select_factors(
            &factor_ids,
            &factor_metrics,
        ).await?;

        // 6. Calculate correlation matrix for selected factors
        let correlation_matrix = self.core_service.calculate_correlation_matrix(
            &selected_factors,
            symbols,
            start_date,
            end_date,
        ).await?;

        // 7. Generate report
        Ok(FactorAnalysisReport {
            universe_id: universe_id.to_string(),
            symbols: symbols.to_vec(),
            analysis_period: (start_date, end_date),
            total_factors: universe.factors.len(),
            selected_factors,
            correlation_matrix: Some(correlation_matrix),
            batch_id: Some(batch_id),
            generated_at: chrono::Utc::now(),
        })
    }

    /// Get comprehensive service health including all components
    pub async fn get_comprehensive_health(&self) -> anyhow::Result<ComprehensiveServiceHealth> {
        let core_health = self.core_service.get_service_health().await?;
        let batch_stats = self.batch_processor.get_statistics();
        
        Ok(ComprehensiveServiceHealth {
            core_service: core_health,
            batch_processor_stats: BatchProcessorHealth {
                total_jobs_processed: batch_stats.total_jobs_processed,
                successful_jobs: batch_stats.successful_jobs,
                failed_jobs: batch_stats.failed_jobs,
                average_job_duration_ms: batch_stats.average_job_duration_ms,
                cache_hit_rate: batch_stats.cache_hit_rate,
            },
            dynamic_selector_active: true, // Could add actual health check
        })
    }
}

/// Report generated from comprehensive factor analysis
#[derive(Debug)]
pub struct FactorAnalysisReport {
    pub universe_id: String,
    pub symbols: Vec<String>,
    pub analysis_period: (chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>),
    pub total_factors: usize,
    pub selected_factors: Vec<String>,
    pub correlation_matrix: Option<FactorCorrelationMatrix>,
    pub batch_id: Option<String>,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

/// Comprehensive health status for all factor services
#[derive(Debug)]
pub struct ComprehensiveServiceHealth {
    pub core_service: ServiceHealth,
    pub batch_processor_stats: BatchProcessorHealth,
    pub dynamic_selector_active: bool,
}

#[derive(Debug)]
pub struct BatchProcessorHealth {
    pub total_jobs_processed: u64,
    pub successful_jobs: u64,
    pub failed_jobs: u64,
    pub average_job_duration_ms: f64,
    pub cache_hit_rate: f64,
}

// Utility functions for common factor operations

/// Create a standard technical analysis factor universe
pub fn create_technical_universe() -> FactorUniverse {
    let mut universe = FactorUniverse::new("Technical Analysis Factors".to_string());
    universe.description = Some("Standard technical analysis factors for equity trading".to_string());
    
    // Moving averages
    universe.add_factor(Factor::simple_moving_average(5));
    universe.add_factor(Factor::simple_moving_average(10));
    universe.add_factor(Factor::simple_moving_average(20));
    universe.add_factor(Factor::simple_moving_average(50));
    universe.add_factor(Factor::simple_moving_average(200));
    
    // Momentum indicators
    universe.add_factor(Factor::rsi(14));
    universe.add_factor(Factor::rsi(21));
    
    // Volatility indicators
    universe.add_factor(Factor::bollinger_bands(20, 2.0));
    
    universe
}

/// Create a momentum-focused factor universe
pub fn create_momentum_universe() -> FactorUniverse {
    let mut universe = FactorUniverse::new("Momentum Factors".to_string());
    universe.description = Some("Momentum-based factors for trend following strategies".to_string());
    
    // Various RSI periods
    for period in [9, 14, 21, 30] {
        universe.add_factor(Factor::rsi(period));
    }
    
    universe
}

/// Validate factor configuration before creation
pub fn validate_factor_config(factor: &Factor) -> Result<(), FactorError> {
    // Check required fields
    if factor.name.is_empty() {
        return Err(FactorError::InvalidParameters("Factor name cannot be empty".to_string()));
    }
    
    // Validate window size if specified
    if let Some(window_size) = factor.parameters.window_size {
        if window_size == 0 {
            return Err(FactorError::InvalidParameters("Window size must be greater than 0".to_string()));
        }
        if window_size > 1000 {
            return Err(FactorError::InvalidParameters("Window size too large (max 1000)".to_string()));
        }
    }
    
    // Validate custom parameters
    for (key, value) in &factor.parameters.custom_params {
        match value {
            FactorParameterValue::Number(n) => {
                if n.is_nan() || n.is_infinite() {
                    return Err(FactorError::InvalidParameters(
                        format!("Parameter '{}' has invalid numeric value", key)
                    ));
                }
            }
            FactorParameterValue::Array(arr) => {
                if arr.is_empty() {
                    return Err(FactorError::InvalidParameters(
                        format!("Parameter '{}' array cannot be empty", key)
                    ));
                }
                for (i, &val) in arr.iter().enumerate() {
                    if val.is_nan() || val.is_infinite() {
                        return Err(FactorError::InvalidParameters(
                            format!("Parameter '{}' array element {} has invalid value", key, i)
                        ));
                    }
                }
            }
            _ => {} // Other types are generally safe
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_technical_universe_creation() {
        let universe = create_technical_universe();
        assert_eq!(universe.name, "Technical Analysis Factors");
        assert!(!universe.factors.is_empty());
        
        let sma_factors = universe.get_factors_by_category(&FactorCategory::Technical);
        assert!(!sma_factors.is_empty());
    }

    #[test]
    fn test_factor_validation() {
        // Valid factor
        let valid_factor = Factor::simple_moving_average(20);
        assert!(validate_factor_config(&valid_factor).is_ok());
        
        // Invalid factor - empty name
        let mut invalid_factor = Factor::simple_moving_average(20);
        invalid_factor.name = String::new();
        assert!(validate_factor_config(&invalid_factor).is_err());
        
        // Invalid factor - zero window
        let mut invalid_factor = Factor::simple_moving_average(0);
        assert!(validate_factor_config(&invalid_factor).is_err());
    }

    #[tokio::test]
    async fn test_integrated_service_creation() {
        use crate::dynamic_factor_selection::{SimpleRiskModel, SimpleTransactionCostModel};
        
        let service_config = FactorServiceConfig::default();
        let batch_config = crate::factor_batch_processing::BatchProcessorConfig::default();
        let dynamic_config = DynamicFactorConfig::default();
        
        struct MockProvider;
        
        #[async_trait::async_trait]
        impl MarketDataProvider for MockProvider {
            async fn get_price_data(&self, _: &str, _: chrono::DateTime<chrono::Utc>, _: chrono::DateTime<chrono::Utc>) -> anyhow::Result<Vec<(chrono::DateTime<chrono::Utc>, f64)>> {
                Ok(vec![(chrono::Utc::now(), 100.0)])
            }
            async fn get_ohlc_data(&self, _: &str, _: chrono::DateTime<chrono::Utc>, _: chrono::DateTime<chrono::Utc>) -> anyhow::Result<Vec<OhlcData>> {
                Ok(vec![OhlcData {
                    timestamp: chrono::Utc::now(),
                    open: 100.0, high: 105.0, low: 95.0, close: 102.0, volume: 10000.0,
                }])
            }
            async fn get_volume_data(&self, _: &str, _: chrono::DateTime<chrono::Utc>, _: chrono::DateTime<chrono::Utc>) -> anyhow::Result<Vec<(chrono::DateTime<chrono::Utc>, f64)>> {
                Ok(vec![(chrono::Utc::now(), 10000.0)])
            }
        }
        
        let data_provider = Arc::new(MockProvider);
        let risk_model = Arc::new(SimpleRiskModel::new(252));
        let tc_model = Arc::new(SimpleTransactionCostModel::new(0.001, 0.01));
        
        let integrated_service = IntegratedFactorServiceFactory::create_full_integrated(
            service_config,
            batch_config,
            dynamic_config,
            data_provider,
            risk_model,
            tc_model,
        ).await.unwrap();
        
        // Test that all components are available
        assert_eq!(integrated_service.core_service.get_service_health().await.unwrap().total_factors, 0);
    }
}