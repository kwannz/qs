use anyhow::{Result, Context};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, BTreeMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use chrono::{DateTime, Utc, Duration};
use ndarray::{Array1, Array2};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicFactorConfig {
    pub selection_window: usize,
    pub rebalance_frequency: Duration,
    pub min_factor_count: usize,
    pub max_factor_count: usize,
    pub significance_threshold: f64,
    pub correlation_threshold: f64,
    pub decay_factor: f64,
    pub weight_bounds: (f64, f64),
    pub turnover_penalty: f64,
    pub regime_sensitivity: f64,
}

impl Default for DynamicFactorConfig {
    fn default() -> Self {
        Self {
            selection_window: 252,
            rebalance_frequency: Duration::days(7),
            min_factor_count: 5,
            max_factor_count: 20,
            significance_threshold: 0.05,
            correlation_threshold: 0.8,
            decay_factor: 0.94,
            weight_bounds: (-0.1, 0.1),
            turnover_penalty: 0.01,
            regime_sensitivity: 0.3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorMetrics {
    pub factor_id: String,
    pub ic: f64,
    pub rank_ic: f64,
    pub ic_ir: f64,
    pub t_stat: f64,
    pub p_value: f64,
    pub turnover: f64,
    pub decay_rate: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
    pub correlation_to_benchmark: f64,
    pub regime_stability: f64,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightAllocation {
    pub factor_id: String,
    pub weight: f64,
    pub confidence: f64,
    pub expected_return: f64,
    pub risk_contribution: f64,
    pub regime_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeState {
    pub regime_id: String,
    pub probability: f64,
    pub volatility: f64,
    pub trend_strength: f64,
    pub correlation_regime: String,
    pub factor_preferences: HashMap<String, f64>,
}

pub struct FactorAnalyzer {
    config: DynamicFactorConfig,
    factor_history: Arc<RwLock<HashMap<String, VecDeque<FactorMetrics>>>>,
    correlation_matrix: Arc<RwLock<Array2<f64>>>,
    factor_names: Arc<RwLock<Vec<String>>>,
}

impl FactorAnalyzer {
    pub fn new(config: DynamicFactorConfig) -> Self {
        Self {
            config,
            factor_history: Arc::new(RwLock::new(HashMap::new())),
            correlation_matrix: Arc::new(RwLock::new(Array2::zeros((0, 0)))),
            factor_names: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn update_factor_metrics(&self, metrics: FactorMetrics) -> Result<()> {
        let mut history = self.factor_history.write().await;
        let factor_history = history.entry(metrics.factor_id.clone())
            .or_insert_with(VecDeque::new);
        
        factor_history.push_back(metrics);
        
        // Keep only recent history
        while factor_history.len() > self.config.selection_window {
            factor_history.pop_front();
        }
        
        Ok(())
    }

    pub async fn calculate_factor_ic(&self, factor_data: &Array1<f64>, returns: &Array1<f64>) -> Result<f64> {
        if factor_data.len() != returns.len() || factor_data.len() < 2 {
            return Ok(0.0);
        }

        let n = factor_data.len() as f64;
        let factor_mean = factor_data.mean().unwrap_or(0.0);
        let returns_mean = returns.mean().unwrap_or(0.0);

        let mut numerator = 0.0;
        let mut factor_var = 0.0;
        let mut returns_var = 0.0;

        for i in 0..factor_data.len() {
            let factor_dev = factor_data[i] - factor_mean;
            let returns_dev = returns[i] - returns_mean;
            
            numerator += factor_dev * returns_dev;
            factor_var += factor_dev * factor_dev;
            returns_var += returns_dev * returns_dev;
        }

        if factor_var == 0.0 || returns_var == 0.0 {
            return Ok(0.0);
        }

        let correlation = numerator / (factor_var * returns_var).sqrt();
        Ok(correlation)
    }

    pub async fn calculate_rank_ic(&self, factor_data: &Array1<f64>, returns: &Array1<f64>) -> Result<f64> {
        if factor_data.len() != returns.len() || factor_data.len() < 2 {
            return Ok(0.0);
        }

        // Convert to ranks
        let mut factor_ranks = self.to_ranks(factor_data);
        let mut returns_ranks = self.to_ranks(returns);

        self.calculate_factor_ic(&factor_ranks, &returns_ranks).await
    }

    fn to_ranks(&self, data: &Array1<f64>) -> Array1<f64> {
        let mut indexed_data: Vec<(usize, f64)> = data.iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();
        
        indexed_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut ranks = vec![0.0; data.len()];
        for (rank, (original_index, _)) in indexed_data.iter().enumerate() {
            ranks[*original_index] = rank as f64;
        }
        
        Array1::from_vec(ranks)
    }

    pub async fn update_correlation_matrix(&self, factor_data: &HashMap<String, Array1<f64>>) -> Result<()> {
        let factor_names: Vec<String> = factor_data.keys().cloned().collect();
        let n_factors = factor_names.len();
        
        if n_factors == 0 {
            return Ok(());
        }

        let mut correlation_matrix = Array2::zeros((n_factors, n_factors));
        
        for i in 0..n_factors {
            for j in 0..n_factors {
                if i == j {
                    correlation_matrix[[i, j]] = 1.0;
                } else {
                    let corr = self.calculate_factor_ic(
                        &factor_data[&factor_names[i]], 
                        &factor_data[&factor_names[j]]
                    ).await?;
                    correlation_matrix[[i, j]] = corr;
                }
            }
        }

        *self.correlation_matrix.write().await = correlation_matrix;
        *self.factor_names.write().await = factor_names;
        
        Ok(())
    }

    pub async fn get_factor_significance(&self, factor_id: &str) -> Result<f64> {
        let history = self.factor_history.read().await;
        if let Some(factor_history) = history.get(factor_id) {
            if let Some(latest) = factor_history.back() {
                return Ok(latest.p_value);
            }
        }
        Ok(1.0) // Not significant if no data
    }

    pub async fn detect_multicollinearity(&self, threshold: f64) -> Result<Vec<String>> {
        let correlation_matrix = self.correlation_matrix.read().await;
        let factor_names = self.factor_names.read().await;
        let mut to_remove = Vec::new();

        if correlation_matrix.nrows() == 0 {
            return Ok(to_remove);
        }

        for i in 0..correlation_matrix.nrows() {
            for j in (i + 1)..correlation_matrix.ncols() {
                if correlation_matrix[[i, j]].abs() > threshold {
                    // Remove factor with lower IC
                    let factor_i_ic = self.get_latest_ic(&factor_names[i]).await?;
                    let factor_j_ic = self.get_latest_ic(&factor_names[j]).await?;
                    
                    if factor_i_ic.abs() < factor_j_ic.abs() {
                        to_remove.push(factor_names[i].clone());
                    } else {
                        to_remove.push(factor_names[j].clone());
                    }
                }
            }
        }

        Ok(to_remove)
    }

    async fn get_latest_ic(&self, factor_id: &str) -> Result<f64> {
        let history = self.factor_history.read().await;
        if let Some(factor_history) = history.get(factor_id) {
            if let Some(latest) = factor_history.back() {
                return Ok(latest.ic);
            }
        }
        Ok(0.0)
    }
}

pub struct WeightOptimizer {
    config: DynamicFactorConfig,
    risk_model: Arc<dyn RiskModel>,
    transaction_cost_model: Arc<dyn TransactionCostModel>,
}

impl WeightOptimizer {
    pub fn new(
        config: DynamicFactorConfig,
        risk_model: Arc<dyn RiskModel>,
        transaction_cost_model: Arc<dyn TransactionCostModel>,
    ) -> Self {
        Self {
            config,
            risk_model,
            transaction_cost_model,
        }
    }

    pub async fn optimize_weights(
        &self,
        selected_factors: &[String],
        factor_metrics: &HashMap<String, FactorMetrics>,
        current_weights: &HashMap<String, f64>,
        regime_state: &RegimeState,
    ) -> Result<HashMap<String, WeightAllocation>> {
        let mut allocations = HashMap::new();

        if selected_factors.is_empty() {
            return Ok(allocations);
        }

        // Extract expected returns and covariance matrix
        let expected_returns = self.extract_expected_returns(selected_factors, factor_metrics)?;
        let covariance_matrix = self.risk_model.get_factor_covariance(selected_factors).await?;

        // Apply regime adjustments
        let regime_adjusted_returns = self.apply_regime_adjustments(
            &expected_returns,
            selected_factors,
            regime_state,
        )?;

        // Optimize using mean-variance with transaction costs
        let optimal_weights = self.solve_optimization(
            &regime_adjusted_returns,
            &covariance_matrix,
            current_weights,
            selected_factors,
        ).await?;

        // Create weight allocations
        for (i, factor_id) in selected_factors.iter().enumerate() {
            let weight = optimal_weights[i];
            let expected_return = regime_adjusted_returns[i];
            let risk_contribution = self.calculate_risk_contribution(
                weight, i, &covariance_matrix
            )?;

            let confidence = self.calculate_confidence(factor_id, factor_metrics)?;
            let regime_weight = regime_state.factor_preferences
                .get(factor_id)
                .copied()
                .unwrap_or(1.0);

            allocations.insert(factor_id.clone(), WeightAllocation {
                factor_id: factor_id.clone(),
                weight,
                confidence,
                expected_return,
                risk_contribution,
                regime_weight,
            });
        }

        Ok(allocations)
    }

    fn extract_expected_returns(
        &self,
        factors: &[String],
        metrics: &HashMap<String, FactorMetrics>,
    ) -> Result<Array1<f64>> {
        let mut returns = Vec::with_capacity(factors.len());
        
        for factor_id in factors {
            let expected_return = if let Some(metric) = metrics.get(factor_id) {
                metric.ic * metric.ic_ir * 0.1 // Simple expected return model
            } else {
                0.0
            };
            returns.push(expected_return);
        }
        
        Ok(Array1::from_vec(returns))
    }

    fn apply_regime_adjustments(
        &self,
        returns: &Array1<f64>,
        factors: &[String],
        regime_state: &RegimeState,
    ) -> Result<Array1<f64>> {
        let mut adjusted_returns = returns.clone();
        
        for (i, factor_id) in factors.iter().enumerate() {
            let regime_preference = regime_state.factor_preferences
                .get(factor_id)
                .copied()
                .unwrap_or(1.0);
            
            adjusted_returns[i] *= regime_preference * regime_state.probability;
        }
        
        Ok(adjusted_returns)
    }

    async fn solve_optimization(
        &self,
        expected_returns: &Array1<f64>,
        covariance_matrix: &Array2<f64>,
        current_weights: &HashMap<String, f64>,
        factors: &[String],
    ) -> Result<Array1<f64>> {
        let n = factors.len();
        let mut weights = Array1::zeros(n);

        if n == 0 {
            return Ok(weights);
        }

        // Simple mean-variance optimization with turnover penalty
        let lambda = 2.0; // Risk aversion parameter
        
        for i in 0..n {
            let current_weight = current_weights.get(&factors[i]).copied().unwrap_or(0.0);
            
            // Calculate optimal weight considering expected return, risk, and turnover
            let expected_return = expected_returns[i];
            let variance = covariance_matrix[[i, i]];
            let turnover_cost = self.config.turnover_penalty * current_weight.abs();
            
            let optimal_weight = if variance > 0.0 {
                (expected_return - turnover_cost) / (lambda * variance)
            } else {
                0.0
            };
            
            // Apply bounds
            weights[i] = optimal_weight.max(self.config.weight_bounds.0)
                                     .min(self.config.weight_bounds.1);
        }

        // Normalize weights
        let weight_sum: f64 = weights.iter().sum();
        if weight_sum.abs() > 1e-8 {
            weights = weights / weight_sum;
        }

        Ok(weights)
    }

    fn calculate_risk_contribution(
        &self,
        weight: f64,
        index: usize,
        covariance_matrix: &Array2<f64>,
    ) -> Result<f64> {
        if covariance_matrix.nrows() <= index || covariance_matrix.ncols() <= index {
            return Ok(0.0);
        }

        let variance_contribution = weight * weight * covariance_matrix[[index, index]];
        Ok(variance_contribution)
    }

    fn calculate_confidence(
        &self,
        factor_id: &str,
        metrics: &HashMap<String, FactorMetrics>,
    ) -> Result<f64> {
        if let Some(metric) = metrics.get(factor_id) {
            // Confidence based on IC information ratio and significance
            let ic_confidence = (1.0 - metric.p_value) * metric.ic_ir.abs();
            Ok(ic_confidence.min(1.0).max(0.0))
        } else {
            Ok(0.0)
        }
    }
}

#[async_trait]
pub trait RiskModel: Send + Sync {
    async fn get_factor_covariance(&self, factors: &[String]) -> Result<Array2<f64>>;
    async fn get_factor_volatility(&self, factor_id: &str) -> Result<f64>;
    async fn update_risk_model(&self, factor_data: &HashMap<String, Array1<f64>>) -> Result<()>;
}

pub struct SimpleRiskModel {
    factor_covariances: Arc<RwLock<HashMap<String, HashMap<String, f64>>>>,
    lookback_window: usize,
}

impl SimpleRiskModel {
    pub fn new(lookback_window: usize) -> Self {
        Self {
            factor_covariances: Arc::new(RwLock::new(HashMap::new())),
            lookback_window,
        }
    }
}

#[async_trait]
impl RiskModel for SimpleRiskModel {
    async fn get_factor_covariance(&self, factors: &[String]) -> Result<Array2<f64>> {
        let n = factors.len();
        let mut covariance_matrix = Array2::zeros((n, n));
        let covariances = self.factor_covariances.read().await;

        for i in 0..n {
            for j in 0..n {
                let cov = if i == j {
                    // Diagonal elements (variance)
                    covariances.get(&factors[i])
                        .and_then(|row| row.get(&factors[j]))
                        .copied()
                        .unwrap_or(0.01) // Default variance
                } else {
                    // Off-diagonal elements (covariance)
                    covariances.get(&factors[i])
                        .and_then(|row| row.get(&factors[j]))
                        .copied()
                        .unwrap_or(0.0)
                };
                covariance_matrix[[i, j]] = cov;
            }
        }

        Ok(covariance_matrix)
    }

    async fn get_factor_volatility(&self, factor_id: &str) -> Result<f64> {
        let covariances = self.factor_covariances.read().await;
        let volatility = covariances.get(factor_id)
            .and_then(|row| row.get(factor_id))
            .map(|var| var.sqrt())
            .unwrap_or(0.1); // Default volatility
        Ok(volatility)
    }

    async fn update_risk_model(&self, factor_data: &HashMap<String, Array1<f64>>) -> Result<()> {
        let mut covariances = self.factor_covariances.write().await;
        
        for (factor1, data1) in factor_data {
            let row = covariances.entry(factor1.clone()).or_insert_with(HashMap::new);
            
            for (factor2, data2) in factor_data {
                if data1.len() == data2.len() && data1.len() > 1 {
                    let cov = self.calculate_covariance(data1, data2)?;
                    row.insert(factor2.clone(), cov);
                }
            }
        }

        Ok(())
    }
}

impl SimpleRiskModel {
    fn calculate_covariance(&self, data1: &Array1<f64>, data2: &Array1<f64>) -> Result<f64> {
        let n = data1.len() as f64;
        if n < 2.0 {
            return Ok(0.0);
        }

        let mean1 = data1.mean().unwrap_or(0.0);
        let mean2 = data2.mean().unwrap_or(0.0);

        let mut covariance = 0.0;
        for i in 0..data1.len() {
            covariance += (data1[i] - mean1) * (data2[i] - mean2);
        }

        Ok(covariance / (n - 1.0))
    }
}

#[async_trait]
pub trait TransactionCostModel: Send + Sync {
    async fn estimate_cost(&self, factor_id: &str, turnover: f64) -> Result<f64>;
    async fn get_total_cost(&self, turnovers: &HashMap<String, f64>) -> Result<f64>;
}

pub struct SimpleTransactionCostModel {
    base_cost: f64,
    market_impact_coefficient: f64,
}

impl SimpleTransactionCostModel {
    pub fn new(base_cost: f64, market_impact_coefficient: f64) -> Self {
        Self {
            base_cost,
            market_impact_coefficient,
        }
    }
}

#[async_trait]
impl TransactionCostModel for SimpleTransactionCostModel {
    async fn estimate_cost(&self, _factor_id: &str, turnover: f64) -> Result<f64> {
        let cost = self.base_cost + self.market_impact_coefficient * turnover.abs().sqrt();
        Ok(cost)
    }

    async fn get_total_cost(&self, turnovers: &HashMap<String, f64>) -> Result<f64> {
        let mut total_cost = 0.0;
        for (factor_id, &turnover) in turnovers {
            total_cost += self.estimate_cost(factor_id, turnover).await?;
        }
        Ok(total_cost)
    }
}

pub struct DynamicFactorSelector {
    config: DynamicFactorConfig,
    analyzer: Arc<FactorAnalyzer>,
    optimizer: Arc<WeightOptimizer>,
    regime_detector: Arc<RwLock<Option<RegimeState>>>,
    last_rebalance: Arc<RwLock<DateTime<Utc>>>,
}

impl DynamicFactorSelector {
    pub fn new(
        config: DynamicFactorConfig,
        risk_model: Arc<dyn RiskModel>,
        transaction_cost_model: Arc<dyn TransactionCostModel>,
    ) -> Self {
        let analyzer = Arc::new(FactorAnalyzer::new(config.clone()));
        let optimizer = Arc::new(WeightOptimizer::new(
            config.clone(),
            risk_model,
            transaction_cost_model,
        ));

        Self {
            config,
            analyzer,
            optimizer,
            regime_detector: Arc::new(RwLock::new(None)),
            last_rebalance: Arc::new(RwLock::new(Utc::now())),
        }
    }

    pub async fn select_factors(
        &self,
        available_factors: &[String],
        factor_metrics: &HashMap<String, FactorMetrics>,
    ) -> Result<Vec<String>> {
        let mut selected_factors = Vec::new();

        // Filter by significance
        for factor_id in available_factors {
            if let Some(metrics) = factor_metrics.get(factor_id) {
                if metrics.p_value <= self.config.significance_threshold {
                    selected_factors.push(factor_id.clone());
                }
            }
        }

        // Remove multicollinear factors
        let factor_data: HashMap<String, Array1<f64>> = HashMap::new(); // Would be populated with actual data
        self.analyzer.update_correlation_matrix(&factor_data).await?;
        
        let to_remove = self.analyzer.detect_multicollinearity(
            self.config.correlation_threshold
        ).await?;

        selected_factors.retain(|f| !to_remove.contains(f));

        // Ensure within bounds
        selected_factors.truncate(self.config.max_factor_count);
        if selected_factors.len() < self.config.min_factor_count {
            // Add back some factors if needed
            for factor_id in available_factors {
                if selected_factors.len() >= self.config.min_factor_count {
                    break;
                }
                if !selected_factors.contains(factor_id) {
                    selected_factors.push(factor_id.clone());
                }
            }
        }

        Ok(selected_factors)
    }

    pub async fn should_rebalance(&self) -> Result<bool> {
        let last_rebalance = *self.last_rebalance.read().await;
        let now = Utc::now();
        Ok(now - last_rebalance >= self.config.rebalance_frequency)
    }

    pub async fn rebalance(
        &self,
        available_factors: &[String],
        factor_metrics: &HashMap<String, FactorMetrics>,
        current_weights: &HashMap<String, f64>,
    ) -> Result<HashMap<String, WeightAllocation>> {
        let selected_factors = self.select_factors(available_factors, factor_metrics).await?;
        
        let regime_state = self.regime_detector.read().await
            .as_ref()
            .cloned()
            .unwrap_or_else(|| RegimeState {
                regime_id: "neutral".to_string(),
                probability: 1.0,
                volatility: 0.2,
                trend_strength: 0.0,
                correlation_regime: "normal".to_string(),
                factor_preferences: HashMap::new(),
            });

        let allocations = self.optimizer.optimize_weights(
            &selected_factors,
            factor_metrics,
            current_weights,
            &regime_state,
        ).await?;

        *self.last_rebalance.write().await = Utc::now();

        Ok(allocations)
    }

    pub async fn update_regime_state(&self, regime_state: RegimeState) -> Result<()> {
        *self.regime_detector.write().await = Some(regime_state);
        Ok(())
    }

    pub async fn get_factor_attribution(
        &self,
        allocations: &HashMap<String, WeightAllocation>,
        returns: &HashMap<String, f64>,
    ) -> Result<HashMap<String, f64>> {
        let mut attribution = HashMap::new();

        for (factor_id, allocation) in allocations {
            if let Some(&factor_return) = returns.get(factor_id) {
                let factor_contribution = allocation.weight * factor_return;
                attribution.insert(factor_id.clone(), factor_contribution);
            }
        }

        Ok(attribution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_factor_analyzer() {
        let config = DynamicFactorConfig::default();
        let analyzer = FactorAnalyzer::new(config);

        let metrics = FactorMetrics {
            factor_id: "test_factor".to_string(),
            ic: 0.05,
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
            last_updated: Utc::now(),
        };

        analyzer.update_factor_metrics(metrics).await.unwrap();
        
        let significance = analyzer.get_factor_significance("test_factor").await.unwrap();
        assert!(significance < 0.05);
    }

    #[tokio::test]
    async fn test_dynamic_factor_selector() {
        let config = DynamicFactorConfig::default();
        let risk_model = Arc::new(SimpleRiskModel::new(252));
        let tc_model = Arc::new(SimpleTransactionCostModel::new(0.001, 0.01));
        
        let selector = DynamicFactorSelector::new(config, risk_model, tc_model);

        let factors = vec!["factor1".to_string(), "factor2".to_string()];
        let mut metrics = HashMap::new();
        
        metrics.insert("factor1".to_string(), FactorMetrics {
            factor_id: "factor1".to_string(),
            ic: 0.08,
            rank_ic: 0.06,
            ic_ir: 1.5,
            t_stat: 3.2,
            p_value: 0.001,
            turnover: 0.6,
            decay_rate: 0.01,
            sharpe_ratio: 0.8,
            max_drawdown: -0.08,
            correlation_to_benchmark: 0.05,
            regime_stability: 0.8,
            last_updated: Utc::now(),
        });

        let selected = selector.select_factors(&factors, &metrics).await.unwrap();
        assert!(selected.contains(&"factor1".to_string()));
    }
}