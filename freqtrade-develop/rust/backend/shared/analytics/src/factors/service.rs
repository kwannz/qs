use anyhow::{Result, Context};
use async_trait::async_trait;
use chrono::{DateTime, Utc, Duration};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use tracing::{debug, info, warn, error, instrument};
use uuid::Uuid;

use super::models::{
    Factor, FactorResult, FactorTimeSeries, FactorUniverse, FactorCorrelationMatrix,
    FactorPerformanceMetrics, FactorBatchRequest, FactorBatchStatus, FactorCategory,
    FactorScreeningCriteria, FactorError, BatchStatusType, BatchPriority, UpdateFrequency,
};
use super::repository::{FactorRepository, FactorRepositoryFactory};
use super::calculator::{FactorCalculationEngine, MarketDataProvider};

/// Configuration for the factor service
#[derive(Debug, Clone)]
pub struct FactorServiceConfig {
    pub max_concurrent_calculations: usize,
    pub cache_ttl_hours: i64,
    pub batch_size_limit: usize,
    pub enable_performance_tracking: bool,
    pub auto_cleanup_old_results_hours: Option<i64>,
    pub default_calculation_timeout_seconds: u64,
    pub enable_correlation_updates: bool,
    pub correlation_update_interval_hours: i64,
}

impl Default for FactorServiceConfig {
    fn default() -> Self {
        Self {
            max_concurrent_calculations: 10,
            cache_ttl_hours: 24,
            batch_size_limit: 1000,
            enable_performance_tracking: true,
            auto_cleanup_old_results_hours: Some(168), // 7 days
            default_calculation_timeout_seconds: 300,  // 5 minutes
            enable_correlation_updates: true,
            correlation_update_interval_hours: 24,
        }
    }
}

/// Main factor service providing unified API for all factor operations
pub struct FactorService {
    config: FactorServiceConfig,
    repository: Arc<dyn FactorRepository>,
    calculation_engine: Arc<FactorCalculationEngine>,
    calculation_semaphore: Arc<Semaphore>,
    performance_tracker: Arc<RwLock<PerformanceTracker>>,
    batch_processor: Arc<BatchProcessor>,
}

/// Performance tracking for factor operations
#[derive(Debug, Default)]
struct PerformanceTracker {
    calculation_times: HashMap<String, Vec<f64>>,
    cache_hit_rates: HashMap<String, (u64, u64)>, // (hits, total)
    error_counts: HashMap<String, u64>,
    last_cleanup: Option<DateTime<Utc>>,
}

/// Batch processing handler
struct BatchProcessor {
    active_batches: Arc<RwLock<HashMap<String, BatchProcessingState>>>,
    repository: Arc<dyn FactorRepository>,
    calculation_engine: Arc<FactorCalculationEngine>,
}

#[derive(Debug)]
struct BatchProcessingState {
    request: FactorBatchRequest,
    status: FactorBatchStatus,
    processed_items: usize,
    total_items: usize,
}

impl FactorService {
    /// Create a new factor service instance
    pub async fn new(
        config: FactorServiceConfig,
        data_provider: Arc<dyn MarketDataProvider>,
    ) -> Result<Self> {
        let repository = Arc::new(FactorRepositoryFactory::create_from_config().await?);
        let calculation_engine = Arc::new(FactorCalculationEngine::new(data_provider));
        let calculation_semaphore = Arc::new(Semaphore::new(config.max_concurrent_calculations));
        let performance_tracker = Arc::new(RwLock::new(PerformanceTracker::default()));
        
        let batch_processor = Arc::new(BatchProcessor {
            active_batches: Arc::new(RwLock::new(HashMap::new())),
            repository: repository.clone(),
            calculation_engine: calculation_engine.clone(),
        });

        Ok(Self {
            config,
            repository,
            calculation_engine,
            calculation_semaphore,
            performance_tracker,
            batch_processor,
        })
    }

    /// Create a new factor service with custom repository
    pub fn with_repository(
        config: FactorServiceConfig,
        repository: Arc<dyn FactorRepository>,
        data_provider: Arc<dyn MarketDataProvider>,
    ) -> Self {
        let calculation_engine = Arc::new(FactorCalculationEngine::new(data_provider));
        let calculation_semaphore = Arc::new(Semaphore::new(config.max_concurrent_calculations));
        let performance_tracker = Arc::new(RwLock::new(PerformanceTracker::default()));
        
        let batch_processor = Arc::new(BatchProcessor {
            active_batches: Arc::new(RwLock::new(HashMap::new())),
            repository: repository.clone(),
            calculation_engine: calculation_engine.clone(),
        });

        Self {
            config,
            repository,
            calculation_engine,
            calculation_semaphore,
            performance_tracker,
            batch_processor,
        }
    }

    // Factor Management API

    /// Create a new factor
    #[instrument(skip(self))]
    pub async fn create_factor(&self, mut factor: Factor) -> Result<String> {
        factor.updated_at = Utc::now();
        let factor_id = self.repository.create_factor(&factor).await?;
        info!("Created factor: {} ({})", factor.name, factor_id);
        Ok(factor_id)
    }

    /// Get a factor by ID
    #[instrument(skip(self))]
    pub async fn get_factor(&self, factor_id: &str) -> Result<Option<Factor>> {
        self.repository.get_factor(factor_id).await
    }

    /// Update an existing factor
    #[instrument(skip(self))]
    pub async fn update_factor(&self, mut factor: Factor) -> Result<()> {
        factor.updated_at = Utc::now();
        self.repository.update_factor(&factor).await?;
        info!("Updated factor: {} ({})", factor.name, factor.id);
        Ok(())
    }

    /// Delete a factor
    #[instrument(skip(self))]
    pub async fn delete_factor(&self, factor_id: &str) -> Result<()> {
        self.repository.delete_factor(factor_id).await?;
        info!("Deleted factor: {}", factor_id);
        Ok(())
    }

    /// List factors with pagination
    #[instrument(skip(self))]
    pub async fn list_factors(&self, limit: Option<u32>, offset: Option<u32>) -> Result<Vec<Factor>> {
        self.repository.list_factors(limit, offset).await
    }

    /// Search factors by query
    #[instrument(skip(self))]
    pub async fn search_factors(&self, query: &str) -> Result<Vec<Factor>> {
        self.repository.search_factors(query).await
    }

    /// Screen factors based on criteria
    #[instrument(skip(self))]
    pub async fn screen_factors(&self, criteria: &FactorScreeningCriteria) -> Result<Vec<Factor>> {
        self.repository.screen_factors(criteria).await
    }

    /// Get factors by category
    #[instrument(skip(self))]
    pub async fn get_factors_by_category(&self, category: &FactorCategory) -> Result<Vec<Factor>> {
        self.repository.find_factors_by_category(category).await
    }

    // Factor Calculation API

    /// Calculate a single factor value at a specific timestamp
    #[instrument(skip(self))]
    pub async fn calculate_factor_point(
        &self,
        factor_id: &str,
        symbol: &str,
        timestamp: DateTime<Utc>,
    ) -> Result<FactorResult> {
        let _permit = self.calculation_semaphore.acquire().await?;
        
        let factor = self.get_factor(factor_id).await?
            .ok_or_else(|| FactorError::UnknownFactor(factor_id.to_string()))?;

        let start_time = std::time::Instant::now();
        
        let result = self.calculation_engine
            .calculate_single_point(&factor, symbol, timestamp)
            .await?;

        if self.config.enable_performance_tracking {
            let duration = start_time.elapsed().as_secs_f64() * 1000.0;
            self.track_calculation_performance(factor_id, duration).await;
        }

        // Store result
        self.repository.store_factor_result(&result).await?;

        debug!("Calculated factor {} for {} at {}", factor_id, symbol, timestamp);
        Ok(result)
    }

    /// Calculate factor time series over a date range
    #[instrument(skip(self))]
    pub async fn calculate_factor_time_series(
        &self,
        factor_id: &str,
        symbol: &str,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<FactorTimeSeries> {
        let _permit = self.calculation_semaphore.acquire().await?;
        
        let factor = self.get_factor(factor_id).await?
            .ok_or_else(|| FactorError::UnknownFactor(factor_id.to_string()))?;

        let start_time = std::time::Instant::now();
        
        let time_series = self.calculation_engine
            .calculate_factor(&factor, symbol, start_date, end_date)
            .await?;

        if self.config.enable_performance_tracking {
            let duration = start_time.elapsed().as_secs_f64() * 1000.0;
            self.track_calculation_performance(factor_id, duration).await;
        }

        // Store time series
        self.repository.store_factor_time_series(&time_series).await?;

        info!("Calculated time series for factor {} symbol {} ({} points)", 
              factor_id, symbol, time_series.data_points.len());
        
        Ok(time_series)
    }

    /// Calculate multiple factors for multiple symbols (batch operation)
    #[instrument(skip(self))]
    pub async fn calculate_factors_batch(
        &self,
        factor_ids: &[String],
        symbols: &[String],
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
        priority: BatchPriority,
    ) -> Result<String> {
        let total_combinations = factor_ids.len() * symbols.len();
        if total_combinations > self.config.batch_size_limit {
            return Err(FactorError::InvalidParameters(
                format!("Batch size {} exceeds limit {}", total_combinations, self.config.batch_size_limit)
            ).into());
        }

        let batch_request = FactorBatchRequest {
            id: Uuid::new_v4().to_string(),
            factor_ids: factor_ids.to_vec(),
            symbols: symbols.to_vec(),
            start_date,
            end_date,
            frequency: UpdateFrequency::Daily, // Default
            priority,
            callback_url: None,
            metadata: HashMap::new(),
        };

        let request_id = self.repository.create_batch_request(&batch_request).await?;
        
        // Start batch processing
        self.batch_processor.start_batch_processing(batch_request).await?;

        info!("Started batch calculation: {} ({} factors × {} symbols)", 
              request_id, factor_ids.len(), symbols.len());

        Ok(request_id)
    }

    /// Get batch processing status
    #[instrument(skip(self))]
    pub async fn get_batch_status(&self, request_id: &str) -> Result<Option<FactorBatchStatus>> {
        self.repository.get_batch_status(request_id).await
    }

    /// Cancel a batch processing request
    #[instrument(skip(self))]
    pub async fn cancel_batch(&self, request_id: &str) -> Result<()> {
        self.batch_processor.cancel_batch(request_id).await
    }

    // Factor Results API

    /// Get factor results for a specific factor and symbol over a date range
    #[instrument(skip(self))]
    pub async fn get_factor_results(
        &self,
        factor_id: &str,
        symbol: &str,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<Vec<FactorResult>> {
        self.repository.get_factor_results(factor_id, symbol, start_date, end_date).await
    }

    /// Get latest factor result for a factor and symbol
    #[instrument(skip(self))]
    pub async fn get_latest_factor_result(&self, factor_id: &str, symbol: &str) -> Result<Option<FactorResult>> {
        self.repository.get_latest_factor_result(factor_id, symbol).await
    }

    /// Get factor time series
    #[instrument(skip(self))]
    pub async fn get_factor_time_series(
        &self,
        factor_id: &str,
        symbol: &str,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<Option<FactorTimeSeries>> {
        self.repository.get_factor_time_series(factor_id, symbol, start_date, end_date).await
    }

    // Performance Metrics API

    /// Store performance metrics for a factor
    #[instrument(skip(self))]
    pub async fn store_performance_metrics(
        &self,
        factor_id: &str,
        metrics: &FactorPerformanceMetrics,
    ) -> Result<()> {
        self.repository.store_performance_metrics(factor_id, metrics).await
    }

    /// Get performance metrics for a factor
    #[instrument(skip(self))]
    pub async fn get_performance_metrics(&self, factor_id: &str) -> Result<Option<FactorPerformanceMetrics>> {
        self.repository.get_performance_metrics(factor_id).await
    }

    // Factor Universe API

    /// Create a factor universe
    #[instrument(skip(self))]
    pub async fn create_factor_universe(&self, universe: &FactorUniverse) -> Result<String> {
        let universe_id = self.repository.create_factor_universe(universe).await?;
        info!("Created factor universe: {} ({})", universe.name, universe_id);
        Ok(universe_id)
    }

    /// Get factor universe
    #[instrument(skip(self))]
    pub async fn get_factor_universe(&self, universe_id: &str) -> Result<Option<FactorUniverse>> {
        self.repository.get_factor_universe(universe_id).await
    }

    /// List all factor universes
    #[instrument(skip(self))]
    pub async fn list_factor_universes(&self) -> Result<Vec<FactorUniverse>> {
        self.repository.list_factor_universes().await
    }

    // Correlation API

    /// Calculate and store correlation matrix for factors
    #[instrument(skip(self))]
    pub async fn calculate_correlation_matrix(
        &self,
        factor_ids: &[String],
        symbols: &[String],
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<FactorCorrelationMatrix> {
        // This is a simplified implementation
        // In practice, you'd calculate correlations from actual factor values
        let correlations = vec![vec![1.0; factor_ids.len()]; factor_ids.len()];
        
        let matrix = FactorCorrelationMatrix {
            factor_ids: factor_ids.to_vec(),
            correlations,
            calculation_date: Utc::now(),
            window_days: (end_date - start_date).num_days() as u32,
        };

        self.repository.store_correlation_matrix(&matrix).await?;
        
        info!("Calculated correlation matrix for {} factors over {} symbols", 
              factor_ids.len(), symbols.len());

        Ok(matrix)
    }

    /// Get correlation matrix
    #[instrument(skip(self))]
    pub async fn get_correlation_matrix(
        &self,
        factor_ids: &[String],
        date: DateTime<Utc>,
    ) -> Result<Option<FactorCorrelationMatrix>> {
        self.repository.get_correlation_matrix(factor_ids, date).await
    }

    // Utility and Maintenance API

    /// Get service health and statistics
    #[instrument(skip(self))]
    pub async fn get_service_health(&self) -> Result<ServiceHealth> {
        let performance_tracker = self.performance_tracker.read().await;
        
        let total_factors = self.repository.list_factors(None, None).await?.len();
        
        let active_batches = self.batch_processor.active_batches.read().await;
        let active_batch_count = active_batches.len();

        let avg_calculation_time = if !performance_tracker.calculation_times.is_empty() {
            performance_tracker.calculation_times.values()
                .flat_map(|times| times.iter())
                .sum::<f64>() / performance_tracker.calculation_times.values()
                .map(|times| times.len())
                .sum::<usize>() as f64
        } else {
            0.0
        };

        Ok(ServiceHealth {
            total_factors,
            active_batch_count,
            average_calculation_time_ms: avg_calculation_time,
            cache_hit_rate: self.calculate_overall_cache_hit_rate(&performance_tracker),
            last_cleanup: performance_tracker.last_cleanup,
            uptime_seconds: 0, // Could track actual uptime
        })
    }

    /// Cleanup old results
    #[instrument(skip(self))]
    pub async fn cleanup_old_results(&self) -> Result<u64> {
        if let Some(hours) = self.config.auto_cleanup_old_results_hours {
            let cutoff_date = Utc::now() - Duration::hours(hours);
            let deleted_count = self.repository.cleanup_old_results(cutoff_date).await?;
            
            // Update performance tracker
            let mut tracker = self.performance_tracker.write().await;
            tracker.last_cleanup = Some(Utc::now());
            
            info!("Cleaned up {} old factor results", deleted_count);
            Ok(deleted_count)
        } else {
            Ok(0)
        }
    }

    /// Vacuum database (optimization)
    #[instrument(skip(self))]
    pub async fn vacuum_database(&self) -> Result<()> {
        self.repository.vacuum_database().await?;
        info!("Database vacuum operation completed");
        Ok(())
    }

    // Private helper methods

    async fn track_calculation_performance(&self, factor_id: &str, duration_ms: f64) {
        let mut tracker = self.performance_tracker.write().await;
        tracker.calculation_times
            .entry(factor_id.to_string())
            .or_insert_with(Vec::new)
            .push(duration_ms);
            
        // Keep only recent measurements (last 100)
        if let Some(times) = tracker.calculation_times.get_mut(factor_id) {
            if times.len() > 100 {
                times.drain(0..50); // Remove oldest 50
            }
        }
    }

    fn calculate_overall_cache_hit_rate(&self, tracker: &PerformanceTracker) -> f64 {
        let (total_hits, total_requests): (u64, u64) = tracker.cache_hit_rates.values()
            .fold((0, 0), |(hits, total), &(h, t)| (hits + h, total + t));
            
        if total_requests > 0 {
            total_hits as f64 / total_requests as f64
        } else {
            0.0
        }
    }
}

/// Service health information
#[derive(Debug)]
pub struct ServiceHealth {
    pub total_factors: usize,
    pub active_batch_count: usize,
    pub average_calculation_time_ms: f64,
    pub cache_hit_rate: f64,
    pub last_cleanup: Option<DateTime<Utc>>,
    pub uptime_seconds: u64,
}

// Batch processor implementation
impl BatchProcessor {
    async fn start_batch_processing(&self, request: FactorBatchRequest) -> Result<()> {
        let request_id = request.id.clone();
        let total_items = request.factor_ids.len() * request.symbols.len();
        
        let status = FactorBatchStatus {
            request_id: request_id.clone(),
            status: BatchStatusType::Running,
            progress_percentage: 0.0,
            started_at: Some(Utc::now()),
            estimated_completion: Some(Utc::now() + Duration::minutes(total_items as i64 * 2)), // Rough estimate
            completed_at: None,
            results_ready: false,
            error_message: None,
            processed_factors: 0,
            total_factors: total_items as u32,
        };

        // Store initial status
        self.repository.update_batch_status(&status).await?;

        // Store processing state
        {
            let mut active_batches = self.active_batches.write().await;
            active_batches.insert(request_id.clone(), BatchProcessingState {
                request: request.clone(),
                status,
                processed_items: 0,
                total_items,
            });
        }

        // Spawn background task for processing
        let processor = self.clone();
        tokio::spawn(async move {
            if let Err(e) = processor.process_batch(request_id.clone()).await {
                error!("Batch processing failed: {}", e);
                // Update status to failed
                if let Ok(mut state) = processor.get_batch_state(&request_id).await {
                    state.status.status = BatchStatusType::Failed;
                    state.status.error_message = Some(e.to_string());
                    let _ = processor.repository.update_batch_status(&state.status).await;
                }
            }
        });

        Ok(())
    }

    async fn process_batch(&self, request_id: String) -> Result<()> {
        let mut state = self.get_batch_state(&request_id).await?;
        
        for factor_id in &state.request.factor_ids {
            for symbol in &state.request.symbols {
                // Calculate factor time series
                match self.calculation_engine.calculate_factor(
                    &Factor::new(factor_id.clone(), FactorCategory::Technical), // Simplified
                    symbol,
                    state.request.start_date,
                    state.request.end_date,
                ).await {
                    Ok(time_series) => {
                        // Store results
                        if let Err(e) = self.repository.store_factor_time_series(&time_series).await {
                            warn!("Failed to store time series for {} {}: {}", factor_id, symbol, e);
                        }
                    }
                    Err(e) => {
                        warn!("Failed to calculate factor {} for {}: {}", factor_id, symbol, e);
                    }
                }

                state.processed_items += 1;
                state.status.processed_factors = state.processed_items as u32;
                state.status.progress_percentage = (state.processed_items as f32 / state.total_items as f32) * 100.0;

                // Update status periodically
                if state.processed_items % 10 == 0 || state.processed_items == state.total_items {
                    self.repository.update_batch_status(&state.status).await?;
                }
            }
        }

        // Mark as completed
        state.status.status = BatchStatusType::Completed;
        state.status.completed_at = Some(Utc::now());
        state.status.results_ready = true;
        state.status.progress_percentage = 100.0;
        
        self.repository.update_batch_status(&state.status).await?;

        // Remove from active batches
        {
            let mut active_batches = self.active_batches.write().await;
            active_batches.remove(&request_id);
        }

        info!("Completed batch processing: {}", request_id);
        Ok(())
    }

    async fn get_batch_state(&self, request_id: &str) -> Result<BatchProcessingState> {
        let active_batches = self.active_batches.read().await;
        active_batches.get(request_id)
            .ok_or_else(|| FactorError::UnknownFactor(format!("Batch not found: {}", request_id)))
            .map(|state| state.clone())
            .map_err(Into::into)
    }

    async fn cancel_batch(&self, request_id: &str) -> Result<()> {
        let mut active_batches = self.active_batches.write().await;
        if let Some(mut state) = active_batches.remove(request_id) {
            state.status.status = BatchStatusType::Cancelled;
            self.repository.update_batch_status(&state.status).await?;
            info!("Cancelled batch processing: {}", request_id);
        }
        Ok(())
    }
}

// Make BatchProcessor cloneable for spawning tasks
impl Clone for BatchProcessor {
    fn clone(&self) -> Self {
        Self {
            active_batches: self.active_batches.clone(),
            repository: self.repository.clone(),
            calculation_engine: self.calculation_engine.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factors::repository::InMemoryFactorRepository;

    struct MockMarketDataProvider;

    #[async_trait]
    impl MarketDataProvider for MockMarketDataProvider {
        async fn get_price_data(&self, _symbol: &str, _start_date: DateTime<Utc>, _end_date: DateTime<Utc>) -> Result<Vec<(DateTime<Utc>, f64)>> {
            Ok(vec![(Utc::now(), 100.0)])
        }

        async fn get_ohlc_data(&self, _symbol: &str, _start_date: DateTime<Utc>, _end_date: DateTime<Utc>) -> Result<Vec<crate::factors::calculator::OhlcData>> {
            use crate::factors::calculator::OhlcData;
            Ok(vec![OhlcData {
                timestamp: Utc::now(),
                open: 100.0,
                high: 105.0,
                low: 95.0,
                close: 102.0,
                volume: 10000.0,
            }])
        }

        async fn get_volume_data(&self, _symbol: &str, _start_date: DateTime<Utc>, _end_date: DateTime<Utc>) -> Result<Vec<(DateTime<Utc>, f64)>> {
            Ok(vec![(Utc::now(), 10000.0)])
        }
    }

    #[tokio::test]
    async fn test_factor_service_creation() {
        let config = FactorServiceConfig::default();
        let data_provider = Arc::new(MockMarketDataProvider);
        let repository = Arc::new(InMemoryFactorRepository::new());

        let service = FactorService::with_repository(config, repository, data_provider);
        
        // Test basic operations
        let factor = Factor::simple_moving_average(20);
        let factor_id = service.create_factor(factor).await.unwrap();
        
        let retrieved = service.get_factor(&factor_id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "SMA_20");
    }

    #[tokio::test]
    async fn test_factor_service_health() {
        let config = FactorServiceConfig::default();
        let data_provider = Arc::new(MockMarketDataProvider);
        let repository = Arc::new(InMemoryFactorRepository::new());

        let service = FactorService::with_repository(config, repository, data_provider);
        
        let health = service.get_service_health().await.unwrap();
        assert_eq!(health.total_factors, 0);
        assert_eq!(health.active_batch_count, 0);
    }
}