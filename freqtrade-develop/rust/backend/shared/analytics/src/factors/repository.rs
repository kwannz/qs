use anyhow::{Result, Context};
use async_trait::async_trait;
use chrono::{DateTime, Utc, Duration};
use serde_json;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error};

use super::models::{
    Factor, FactorResult, FactorTimeSeries, FactorUniverse, FactorCorrelationMatrix,
    FactorPerformanceMetrics, FactorBatchRequest, FactorBatchStatus, FactorCategory,
    FactorScreeningCriteria, FactorError, BatchStatusType, BatchPriority,
};

/// Trait defining factor repository operations
#[async_trait]
pub trait FactorRepository: Send + Sync {
    // Factor CRUD operations
    async fn create_factor(&self, factor: &Factor) -> Result<String>;
    async fn get_factor(&self, factor_id: &str) -> Result<Option<Factor>>;
    async fn update_factor(&self, factor: &Factor) -> Result<()>;
    async fn delete_factor(&self, factor_id: &str) -> Result<()>;
    async fn list_factors(&self, limit: Option<u32>, offset: Option<u32>) -> Result<Vec<Factor>>;
    
    // Factor search and filtering
    async fn find_factors_by_category(&self, category: &FactorCategory) -> Result<Vec<Factor>>;
    async fn search_factors(&self, query: &str) -> Result<Vec<Factor>>;
    async fn screen_factors(&self, criteria: &FactorScreeningCriteria) -> Result<Vec<Factor>>;
    
    // Factor results operations
    async fn store_factor_result(&self, result: &FactorResult) -> Result<()>;
    async fn store_factor_results_batch(&self, results: &[FactorResult]) -> Result<()>;
    async fn get_factor_results(
        &self,
        factor_id: &str,
        symbol: &str,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<Vec<FactorResult>>;
    async fn get_latest_factor_result(&self, factor_id: &str, symbol: &str) -> Result<Option<FactorResult>>;
    
    // Time series operations
    async fn store_factor_time_series(&self, time_series: &FactorTimeSeries) -> Result<()>;
    async fn get_factor_time_series(
        &self,
        factor_id: &str,
        symbol: &str,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<Option<FactorTimeSeries>>;
    
    // Performance metrics operations
    async fn store_performance_metrics(
        &self,
        factor_id: &str,
        metrics: &FactorPerformanceMetrics,
    ) -> Result<()>;
    async fn get_performance_metrics(&self, factor_id: &str) -> Result<Option<FactorPerformanceMetrics>>;
    
    // Factor universe operations
    async fn create_factor_universe(&self, universe: &FactorUniverse) -> Result<String>;
    async fn get_factor_universe(&self, universe_id: &str) -> Result<Option<FactorUniverse>>;
    async fn update_factor_universe(&self, universe: &FactorUniverse) -> Result<()>;
    async fn list_factor_universes(&self) -> Result<Vec<FactorUniverse>>;
    
    // Correlation matrix operations
    async fn store_correlation_matrix(&self, matrix: &FactorCorrelationMatrix) -> Result<()>;
    async fn get_correlation_matrix(
        &self,
        factor_ids: &[String],
        date: DateTime<Utc>,
    ) -> Result<Option<FactorCorrelationMatrix>>;
    
    // Batch processing operations
    async fn create_batch_request(&self, request: &FactorBatchRequest) -> Result<String>;
    async fn get_batch_request(&self, request_id: &str) -> Result<Option<FactorBatchRequest>>;
    async fn update_batch_status(&self, status: &FactorBatchStatus) -> Result<()>;
    async fn get_batch_status(&self, request_id: &str) -> Result<Option<FactorBatchStatus>>;
    async fn list_batch_requests(&self, status_filter: Option<BatchStatusType>) -> Result<Vec<FactorBatchRequest>>;
    
    // Cleanup operations
    async fn cleanup_old_results(&self, older_than: DateTime<Utc>) -> Result<u64>;
    async fn vacuum_database(&self) -> Result<()>;
}

/// In-memory implementation of factor repository (for testing and development)
#[derive(Debug)]
pub struct InMemoryFactorRepository {
    factors: Arc<RwLock<HashMap<String, Factor>>>,
    results: Arc<RwLock<Vec<FactorResult>>>,
    time_series: Arc<RwLock<HashMap<String, FactorTimeSeries>>>,
    performance_metrics: Arc<RwLock<HashMap<String, FactorPerformanceMetrics>>>,
    universes: Arc<RwLock<HashMap<String, FactorUniverse>>>,
    correlation_matrices: Arc<RwLock<Vec<FactorCorrelationMatrix>>>,
    batch_requests: Arc<RwLock<HashMap<String, FactorBatchRequest>>>,
    batch_statuses: Arc<RwLock<HashMap<String, FactorBatchStatus>>>,
}

impl InMemoryFactorRepository {
    pub fn new() -> Self {
        Self {
            factors: Arc::new(RwLock::new(HashMap::new())),
            results: Arc::new(RwLock::new(Vec::new())),
            time_series: Arc::new(RwLock::new(HashMap::new())),
            performance_metrics: Arc::new(RwLock::new(HashMap::new())),
            universes: Arc::new(RwLock::new(HashMap::new())),
            correlation_matrices: Arc::new(RwLock::new(Vec::new())),
            batch_requests: Arc::new(RwLock::new(HashMap::new())),
            batch_statuses: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn generate_time_series_key(&self, factor_id: &str, symbol: &str) -> String {
        format!("{}:{}", factor_id, symbol)
    }
}

#[async_trait]
impl FactorRepository for InMemoryFactorRepository {
    async fn create_factor(&self, factor: &Factor) -> Result<String> {
        let mut factors = self.factors.write().await;
        let factor_id = factor.id.clone();
        factors.insert(factor_id.clone(), factor.clone());
        debug!("Created factor: {} ({})", factor.name, factor_id);
        Ok(factor_id)
    }

    async fn get_factor(&self, factor_id: &str) -> Result<Option<Factor>> {
        let factors = self.factors.read().await;
        Ok(factors.get(factor_id).cloned())
    }

    async fn update_factor(&self, factor: &Factor) -> Result<()> {
        let mut factors = self.factors.write().await;
        factors.insert(factor.id.clone(), factor.clone());
        debug!("Updated factor: {} ({})", factor.name, factor.id);
        Ok(())
    }

    async fn delete_factor(&self, factor_id: &str) -> Result<()> {
        let mut factors = self.factors.write().await;
        if factors.remove(factor_id).is_some() {
            debug!("Deleted factor: {}", factor_id);
            Ok(())
        } else {
            Err(FactorError::UnknownFactor(factor_id.to_string()).into())
        }
    }

    async fn list_factors(&self, limit: Option<u32>, offset: Option<u32>) -> Result<Vec<Factor>> {
        let factors = self.factors.read().await;
        let mut factor_list: Vec<Factor> = factors.values().cloned().collect();
        factor_list.sort_by(|a, b| a.name.cmp(&b.name));

        let start = offset.unwrap_or(0) as usize;
        let end = if let Some(limit) = limit {
            std::cmp::min(start + limit as usize, factor_list.len())
        } else {
            factor_list.len()
        };

        Ok(factor_list.get(start..end).unwrap_or(&[]).to_vec())
    }

    async fn find_factors_by_category(&self, category: &FactorCategory) -> Result<Vec<Factor>> {
        let factors = self.factors.read().await;
        let filtered: Vec<Factor> = factors.values()
            .filter(|factor| &factor.category == category)
            .cloned()
            .collect();
        Ok(filtered)
    }

    async fn search_factors(&self, query: &str) -> Result<Vec<Factor>> {
        let factors = self.factors.read().await;
        let query_lower = query.to_lowercase();
        
        let filtered: Vec<Factor> = factors.values()
            .filter(|factor| {
                factor.name.to_lowercase().contains(&query_lower) ||
                factor.description.as_ref()
                    .map(|d| d.to_lowercase().contains(&query_lower))
                    .unwrap_or(false) ||
                factor.metadata.tags.iter()
                    .any(|tag| tag.to_lowercase().contains(&query_lower))
            })
            .cloned()
            .collect();
        
        Ok(filtered)
    }

    async fn screen_factors(&self, criteria: &FactorScreeningCriteria) -> Result<Vec<Factor>> {
        let factors = self.factors.read().await;
        let universe = FactorUniverse {
            id: "temp".to_string(),
            name: "temp".to_string(),
            description: None,
            factors: factors.values().cloned().collect(),
            correlation_matrix: None,
            created_at: Utc::now(),
            last_updated: Utc::now(),
        };
        
        Ok(universe.screen_factors(criteria).into_iter().cloned().collect())
    }

    async fn store_factor_result(&self, result: &FactorResult) -> Result<()> {
        let mut results = self.results.write().await;
        results.push(result.clone());
        debug!("Stored factor result: {} for {}", result.factor_id, result.symbol);
        Ok(())
    }

    async fn store_factor_results_batch(&self, results: &[FactorResult]) -> Result<()> {
        let mut result_store = self.results.write().await;
        result_store.extend_from_slice(results);
        debug!("Stored {} factor results in batch", results.len());
        Ok(())
    }

    async fn get_factor_results(
        &self,
        factor_id: &str,
        symbol: &str,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<Vec<FactorResult>> {
        let results = self.results.read().await;
        let filtered: Vec<FactorResult> = results.iter()
            .filter(|result| {
                result.factor_id == factor_id &&
                result.symbol == symbol &&
                result.timestamp >= start_date &&
                result.timestamp <= end_date
            })
            .cloned()
            .collect();
        Ok(filtered)
    }

    async fn get_latest_factor_result(&self, factor_id: &str, symbol: &str) -> Result<Option<FactorResult>> {
        let results = self.results.read().await;
        let mut matching_results: Vec<&FactorResult> = results.iter()
            .filter(|result| result.factor_id == factor_id && result.symbol == symbol)
            .collect();
        
        matching_results.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(matching_results.first().map(|&result| result.clone()))
    }

    async fn store_factor_time_series(&self, time_series: &FactorTimeSeries) -> Result<()> {
        let mut ts_store = self.time_series.write().await;
        let key = self.generate_time_series_key(&time_series.factor_id, &time_series.symbol);
        ts_store.insert(key, time_series.clone());
        debug!("Stored time series for factor {} symbol {}", time_series.factor_id, time_series.symbol);
        Ok(())
    }

    async fn get_factor_time_series(
        &self,
        factor_id: &str,
        symbol: &str,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
    ) -> Result<Option<FactorTimeSeries>> {
        let ts_store = self.time_series.read().await;
        let key = self.generate_time_series_key(factor_id, symbol);
        
        if let Some(ts) = ts_store.get(&key) {
            // Filter data points by date range
            let filtered_points = ts.data_points.iter()
                .filter(|point| point.timestamp >= start_date && point.timestamp <= end_date)
                .cloned()
                .collect();
            
            let mut filtered_ts = ts.clone();
            filtered_ts.data_points = filtered_points;
            Ok(Some(filtered_ts))
        } else {
            Ok(None)
        }
    }

    async fn store_performance_metrics(
        &self,
        factor_id: &str,
        metrics: &FactorPerformanceMetrics,
    ) -> Result<()> {
        let mut metrics_store = self.performance_metrics.write().await;
        metrics_store.insert(factor_id.to_string(), metrics.clone());
        debug!("Stored performance metrics for factor {}", factor_id);
        Ok(())
    }

    async fn get_performance_metrics(&self, factor_id: &str) -> Result<Option<FactorPerformanceMetrics>> {
        let metrics_store = self.performance_metrics.read().await;
        Ok(metrics_store.get(factor_id).cloned())
    }

    async fn create_factor_universe(&self, universe: &FactorUniverse) -> Result<String> {
        let mut universes = self.universes.write().await;
        let universe_id = universe.id.clone();
        universes.insert(universe_id.clone(), universe.clone());
        debug!("Created factor universe: {} ({})", universe.name, universe_id);
        Ok(universe_id)
    }

    async fn get_factor_universe(&self, universe_id: &str) -> Result<Option<FactorUniverse>> {
        let universes = self.universes.read().await;
        Ok(universes.get(universe_id).cloned())
    }

    async fn update_factor_universe(&self, universe: &FactorUniverse) -> Result<()> {
        let mut universes = self.universes.write().await;
        universes.insert(universe.id.clone(), universe.clone());
        debug!("Updated factor universe: {} ({})", universe.name, universe.id);
        Ok(())
    }

    async fn list_factor_universes(&self) -> Result<Vec<FactorUniverse>> {
        let universes = self.universes.read().await;
        Ok(universes.values().cloned().collect())
    }

    async fn store_correlation_matrix(&self, matrix: &FactorCorrelationMatrix) -> Result<()> {
        let mut matrices = self.correlation_matrices.write().await;
        matrices.push(matrix.clone());
        debug!("Stored correlation matrix for {} factors", matrix.factor_ids.len());
        Ok(())
    }

    async fn get_correlation_matrix(
        &self,
        factor_ids: &[String],
        date: DateTime<Utc>,
    ) -> Result<Option<FactorCorrelationMatrix>> {
        let matrices = self.correlation_matrices.read().await;
        
        // Find the most recent matrix that includes all requested factors
        let mut best_match: Option<&FactorCorrelationMatrix> = None;
        let mut best_date_diff = Duration::max_value();
        
        for matrix in matrices.iter() {
            // Check if matrix includes all requested factors
            let all_included = factor_ids.iter()
                .all(|id| matrix.factor_ids.contains(id));
            
            if all_included {
                let date_diff = (matrix.calculation_date - date).abs();
                if date_diff < best_date_diff {
                    best_match = Some(matrix);
                    best_date_diff = date_diff;
                }
            }
        }
        
        Ok(best_match.cloned())
    }

    async fn create_batch_request(&self, request: &FactorBatchRequest) -> Result<String> {
        let mut requests = self.batch_requests.write().await;
        let request_id = request.id.clone();
        requests.insert(request_id.clone(), request.clone());
        debug!("Created batch request: {}", request_id);
        Ok(request_id)
    }

    async fn get_batch_request(&self, request_id: &str) -> Result<Option<FactorBatchRequest>> {
        let requests = self.batch_requests.read().await;
        Ok(requests.get(request_id).cloned())
    }

    async fn update_batch_status(&self, status: &FactorBatchStatus) -> Result<()> {
        let mut statuses = self.batch_statuses.write().await;
        statuses.insert(status.request_id.clone(), status.clone());
        debug!("Updated batch status for request: {}", status.request_id);
        Ok(())
    }

    async fn get_batch_status(&self, request_id: &str) -> Result<Option<FactorBatchStatus>> {
        let statuses = self.batch_statuses.read().await;
        Ok(statuses.get(request_id).cloned())
    }

    async fn list_batch_requests(&self, status_filter: Option<BatchStatusType>) -> Result<Vec<FactorBatchRequest>> {
        let requests = self.batch_requests.read().await;
        let statuses = self.batch_statuses.read().await;
        
        let filtered: Vec<FactorBatchRequest> = requests.values()
            .filter(|request| {
                if let Some(ref filter_status) = status_filter {
                    if let Some(status) = statuses.get(&request.id) {
                        &status.status == filter_status
                    } else {
                        // If no status found, consider it as queued
                        filter_status == &BatchStatusType::Queued
                    }
                } else {
                    true
                }
            })
            .cloned()
            .collect();
        
        Ok(filtered)
    }

    async fn cleanup_old_results(&self, older_than: DateTime<Utc>) -> Result<u64> {
        let mut results = self.results.write().await;
        let initial_count = results.len();
        results.retain(|result| result.timestamp > older_than);
        let deleted_count = initial_count - results.len();
        info!("Cleaned up {} old factor results", deleted_count);
        Ok(deleted_count as u64)
    }

    async fn vacuum_database(&self) -> Result<()> {
        // For in-memory implementation, this is a no-op
        debug!("Vacuum operation completed (no-op for in-memory store)");
        Ok(())
    }
}

/// PostgreSQL implementation of factor repository
#[cfg(feature = "postgres")]
pub struct PostgresFactorRepository {
    pool: sqlx::PgPool,
}

#[cfg(feature = "postgres")]
impl PostgresFactorRepository {
    pub fn new(pool: sqlx::PgPool) -> Self {
        Self { pool }
    }

    pub async fn run_migrations(&self) -> Result<()> {
        sqlx::migrate!("./migrations").run(&self.pool).await?;
        Ok(())
    }
}

#[cfg(feature = "postgres")]
#[async_trait]
impl FactorRepository for PostgresFactorRepository {
    async fn create_factor(&self, factor: &Factor) -> Result<String> {
        let factor_json = serde_json::to_value(factor)?;
        
        let row = sqlx::query!(
            r#"
            INSERT INTO factors (id, name, category, description, formula, parameters, metadata, created_at, updated_at, data)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            RETURNING id
            "#,
            factor.id,
            factor.name,
            factor.category.to_string(),
            factor.description,
            factor.formula,
            serde_json::to_string(&factor.parameters)?,
            serde_json::to_string(&factor.metadata)?,
            factor.created_at,
            factor.updated_at,
            factor_json
        )
        .fetch_one(&self.pool)
        .await?;

        Ok(row.id)
    }

    async fn get_factor(&self, factor_id: &str) -> Result<Option<Factor>> {
        let row = sqlx::query!(
            "SELECT data FROM factors WHERE id = $1",
            factor_id
        )
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => {
                let factor: Factor = serde_json::from_value(row.data)?;
                Ok(Some(factor))
            }
            None => Ok(None),
        }
    }

    async fn update_factor(&self, factor: &Factor) -> Result<()> {
        let factor_json = serde_json::to_value(factor)?;
        
        sqlx::query!(
            r#"
            UPDATE factors 
            SET name = $2, category = $3, description = $4, formula = $5, 
                parameters = $6, metadata = $7, updated_at = $8, data = $9
            WHERE id = $1
            "#,
            factor.id,
            factor.name,
            factor.category.to_string(),
            factor.description,
            factor.formula,
            serde_json::to_string(&factor.parameters)?,
            serde_json::to_string(&factor.metadata)?,
            factor.updated_at,
            factor_json
        )
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn delete_factor(&self, factor_id: &str) -> Result<()> {
        let result = sqlx::query!(
            "DELETE FROM factors WHERE id = $1",
            factor_id
        )
        .execute(&self.pool)
        .await?;

        if result.rows_affected() == 0 {
            return Err(FactorError::UnknownFactor(factor_id.to_string()).into());
        }

        Ok(())
    }

    async fn list_factors(&self, limit: Option<u32>, offset: Option<u32>) -> Result<Vec<Factor>> {
        let rows = sqlx::query!(
            "SELECT data FROM factors ORDER BY name LIMIT $1 OFFSET $2",
            limit.map(|l| l as i64),
            offset.unwrap_or(0) as i64
        )
        .fetch_all(&self.pool)
        .await?;

        let mut factors = Vec::new();
        for row in rows {
            let factor: Factor = serde_json::from_value(row.data)?;
            factors.push(factor);
        }

        Ok(factors)
    }

    async fn find_factors_by_category(&self, category: &FactorCategory) -> Result<Vec<Factor>> {
        let rows = sqlx::query!(
            "SELECT data FROM factors WHERE category = $1",
            category.to_string()
        )
        .fetch_all(&self.pool)
        .await?;

        let mut factors = Vec::new();
        for row in rows {
            let factor: Factor = serde_json::from_value(row.data)?;
            factors.push(factor);
        }

        Ok(factors)
    }

    async fn search_factors(&self, query: &str) -> Result<Vec<Factor>> {
        let search_pattern = format!("%{}%", query.to_lowercase());
        
        let rows = sqlx::query!(
            r#"
            SELECT data FROM factors 
            WHERE LOWER(name) LIKE $1 
               OR LOWER(description) LIKE $1 
               OR data::text ILIKE $1
            "#,
            search_pattern
        )
        .fetch_all(&self.pool)
        .await?;

        let mut factors = Vec::new();
        for row in rows {
            let factor: Factor = serde_json::from_value(row.data)?;
            factors.push(factor);
        }

        Ok(factors)
    }

    // Implement remaining methods...
    // This is a substantial implementation that would require full PostgreSQL schema
    // For brevity, I'm showing the pattern. The remaining methods would follow similar patterns.

    async fn screen_factors(&self, _criteria: &FactorScreeningCriteria) -> Result<Vec<Factor>> {
        // Complex query building based on criteria
        // This would involve dynamic SQL generation based on the screening criteria
        todo!("Implement complex screening query")
    }

    // ... implement remaining methods following similar patterns
    async fn store_factor_result(&self, _result: &FactorResult) -> Result<()> { todo!() }
    async fn store_factor_results_batch(&self, _results: &[FactorResult]) -> Result<()> { todo!() }
    async fn get_factor_results(&self, _factor_id: &str, _symbol: &str, _start_date: DateTime<Utc>, _end_date: DateTime<Utc>) -> Result<Vec<FactorResult>> { todo!() }
    async fn get_latest_factor_result(&self, _factor_id: &str, _symbol: &str) -> Result<Option<FactorResult>> { todo!() }
    async fn store_factor_time_series(&self, _time_series: &FactorTimeSeries) -> Result<()> { todo!() }
    async fn get_factor_time_series(&self, _factor_id: &str, _symbol: &str, _start_date: DateTime<Utc>, _end_date: DateTime<Utc>) -> Result<Option<FactorTimeSeries>> { todo!() }
    async fn store_performance_metrics(&self, _factor_id: &str, _metrics: &FactorPerformanceMetrics) -> Result<()> { todo!() }
    async fn get_performance_metrics(&self, _factor_id: &str) -> Result<Option<FactorPerformanceMetrics>> { todo!() }
    async fn create_factor_universe(&self, _universe: &FactorUniverse) -> Result<String> { todo!() }
    async fn get_factor_universe(&self, _universe_id: &str) -> Result<Option<FactorUniverse>> { todo!() }
    async fn update_factor_universe(&self, _universe: &FactorUniverse) -> Result<()> { todo!() }
    async fn list_factor_universes(&self) -> Result<Vec<FactorUniverse>> { todo!() }
    async fn store_correlation_matrix(&self, _matrix: &FactorCorrelationMatrix) -> Result<()> { todo!() }
    async fn get_correlation_matrix(&self, _factor_ids: &[String], _date: DateTime<Utc>) -> Result<Option<FactorCorrelationMatrix>> { todo!() }
    async fn create_batch_request(&self, _request: &FactorBatchRequest) -> Result<String> { todo!() }
    async fn get_batch_request(&self, _request_id: &str) -> Result<Option<FactorBatchRequest>> { todo!() }
    async fn update_batch_status(&self, _status: &FactorBatchStatus) -> Result<()> { todo!() }
    async fn get_batch_status(&self, _request_id: &str) -> Result<Option<FactorBatchStatus>> { todo!() }
    async fn list_batch_requests(&self, _status_filter: Option<BatchStatusType>) -> Result<Vec<FactorBatchRequest>> { todo!() }
    async fn cleanup_old_results(&self, _older_than: DateTime<Utc>) -> Result<u64> { todo!() }
    async fn vacuum_database(&self) -> Result<()> { 
        sqlx::query("VACUUM ANALYZE").execute(&self.pool).await?;
        Ok(())
    }
}

/// Factory for creating factor repository instances
pub struct FactorRepositoryFactory;

impl FactorRepositoryFactory {
    /// Create an in-memory factor repository (for testing/development)
    pub fn create_in_memory() -> Box<dyn FactorRepository> {
        Box::new(InMemoryFactorRepository::new())
    }

    /// Create a PostgreSQL factor repository
    #[cfg(feature = "postgres")]
    pub fn create_postgres(pool: sqlx::PgPool) -> Box<dyn FactorRepository> {
        Box::new(PostgresFactorRepository::new(pool))
    }

    /// Create repository based on environment configuration
    pub async fn create_from_config() -> Result<Box<dyn FactorRepository>> {
        // In a real implementation, this would read from config/environment
        // For now, default to in-memory
        Ok(Self::create_in_memory())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factors::models::*;

    #[tokio::test]
    async fn test_in_memory_repository() {
        let repo = InMemoryFactorRepository::new();
        
        // Test factor CRUD
        let factor = Factor::simple_moving_average(20);
        let factor_id = repo.create_factor(&factor).await.unwrap();
        assert_eq!(factor_id, factor.id);
        
        let retrieved = repo.get_factor(&factor_id).await.unwrap();
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "SMA_20");
        
        // Test listing
        let factors = repo.list_factors(None, None).await.unwrap();
        assert_eq!(factors.len(), 1);
        
        // Test category search
        let technical_factors = repo.find_factors_by_category(&FactorCategory::Technical).await.unwrap();
        assert_eq!(technical_factors.len(), 1);
    }

    #[tokio::test]
    async fn test_factor_results_storage() {
        let repo = InMemoryFactorRepository::new();
        
        let result = FactorResult {
            factor_id: "test_factor".to_string(),
            symbol: "AAPL".to_string(),
            timestamp: Utc::now(),
            value: 0.05,
            percentile_rank: Some(0.8),
            z_score: Some(1.2),
            confidence: 0.95,
            metadata: FactorResultMetadata {
                computation_time_ms: 10.0,
                data_points_used: 100,
                data_quality_score: 0.98,
                cache_hit: false,
                warnings: vec![],
                debug_info: None,
            },
        };
        
        repo.store_factor_result(&result).await.unwrap();
        
        let latest = repo.get_latest_factor_result("test_factor", "AAPL").await.unwrap();
        assert!(latest.is_some());
        assert_eq!(latest.unwrap().value, 0.05);
    }

    #[tokio::test]
    async fn test_batch_operations() {
        let repo = InMemoryFactorRepository::new();
        
        let request = FactorBatchRequest {
            id: "batch_001".to_string(),
            factor_ids: vec!["RSI_14".to_string(), "SMA_20".to_string()],
            symbols: vec!["AAPL".to_string(), "GOOGL".to_string()],
            start_date: Utc::now() - Duration::days(30),
            end_date: Utc::now(),
            frequency: UpdateFrequency::Daily,
            priority: BatchPriority::Normal,
            callback_url: None,
            metadata: HashMap::new(),
        };
        
        let request_id = repo.create_batch_request(&request).await.unwrap();
        assert_eq!(request_id, "batch_001");
        
        let status = FactorBatchStatus {
            request_id: request_id.clone(),
            status: BatchStatusType::Running,
            progress_percentage: 50.0,
            started_at: Some(Utc::now()),
            estimated_completion: Some(Utc::now() + Duration::minutes(10)),
            completed_at: None,
            results_ready: false,
            error_message: None,
            processed_factors: 2,
            total_factors: 4,
        };
        
        repo.update_batch_status(&status).await.unwrap();
        
        let retrieved_status = repo.get_batch_status(&request_id).await.unwrap();
        assert!(retrieved_status.is_some());
        assert_eq!(retrieved_status.unwrap().progress_percentage, 50.0);
    }
}