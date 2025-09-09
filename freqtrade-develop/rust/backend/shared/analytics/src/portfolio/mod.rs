pub mod vol_targeting;
pub mod risk_parity;
// pub mod optimization;
// pub mod rebalancing;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

// 临时结构体（替代缺失模块）
struct SmartRebalancer {
    threshold: f64,
}

struct CostModel {
    cost_penalty: f64,
}

impl SmartRebalancer {
    fn new(threshold: f64) -> Self {
        Self { threshold }
    }
}

impl CostModel {
    fn new(cost_penalty: f64) -> Self {
        Self { cost_penalty }
    }
    
    fn optimize_with_costs(
        &self,
        _positions: &[Position],
        weights: &HashMap<String, f64>,
        _params: &OptimizationParams,
    ) -> Result<HashMap<String, f64>> {
        Ok(weights.clone())
    }
    
    fn estimate_transaction_costs(
        &self,
        current_positions: &[Position],
        target_weights: &HashMap<String, f64>,
    ) -> Result<f64> {
        let current_weights: HashMap<String, f64> = current_positions.iter()
            .map(|pos| (pos.symbol.clone(), pos.weight))
            .collect();
        
        let mut turnover = 0.0;
        let all_symbols: std::collections::HashSet<String> = current_weights.keys()
            .chain(target_weights.keys())
            .cloned()
            .collect();
        
        for symbol in all_symbols {
            let current_weight = current_weights.get(&symbol).unwrap_or(&0.0);
            let target_weight = target_weights.get(&symbol).unwrap_or(&0.0);
            turnover += (target_weight - current_weight).abs();
        }
        
        Ok(turnover * self.cost_penalty)
    }
}

/// AG3组合优化引擎
pub struct PortfolioOptimizer {
    vol_targeter: vol_targeting::VolTargeter,
    risk_pariter: risk_parity::RiskParityOptimizer,
    rebalancer: SmartRebalancer,
    cost_model: CostModel,
}

/// 组合持仓结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: String,
    pub size: f64,
    pub market_value: f64,
    pub weight: f64,
    pub target_weight: f64,
    pub last_update: DateTime<Utc>,
}

/// 组合优化参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationParams {
    pub target_vol: f64,              // 目标波动率
    pub max_position_weight: f64,     // 最大单个持仓权重
    pub min_position_weight: f64,     // 最小单个持仓权重
    pub rebalance_threshold: f64,     // 再平衡阈值
    pub cost_penalty: f64,            // 成本惩罚参数
    pub lookback_period: usize,       // 历史回望期
    pub risk_aversion: f64,           // 风险厌恶系数
    pub use_shrinkage: bool,          // 是否使用收缩估计
    pub alpha_confidence: f64,        // Alpha置信度
}

/// 优化结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub target_weights: HashMap<String, f64>,
    pub expected_return: f64,
    pub expected_vol: f64,
    pub sharpe_ratio: f64,
    pub turnover: f64,
    pub cost_estimate: f64,
    pub risk_contribution: HashMap<String, f64>,
    pub optimization_method: String,
    pub confidence_score: f64,
    pub timestamp: DateTime<Utc>,
}

/// 风险指标
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMetrics {
    pub portfolio_vol: f64,
    pub var_95: f64,
    pub cvar_95: f64,
    pub max_drawdown: f64,
    pub correlation_risk: f64,
    pub concentration_risk: f64,
    pub tail_risk: f64,
}

impl PortfolioOptimizer {
    pub fn new(params: OptimizationParams) -> Self {
        Self {
            vol_targeter: vol_targeting::VolTargeter::new(params.target_vol),
            risk_pariter: risk_parity::RiskParityOptimizer::new(),
            rebalancer: SmartRebalancer::new(params.rebalance_threshold),
            cost_model: CostModel::new(params.cost_penalty),
        }
    }

    /// 执行组合优化
    pub fn optimize(
        &mut self,
        positions: &[Position],
        returns_matrix: &[Vec<f64>],  // [时间][资产]收益率矩阵
        alphas: &HashMap<String, f64>,
        params: &OptimizationParams,
    ) -> Result<OptimizationResult> {
        
        // 1. 风险矩阵估计
        let risk_matrix = self.estimate_risk_matrix(returns_matrix, params)?;
        
        // 2. 根据不同方法计算目标权重
        let vol_weights = self.vol_targeter.optimize(positions, &risk_matrix, params)?;
        let risk_parity_weights = self.risk_pariter.optimize(positions, &risk_matrix, params)?;
        
        // 3. Alpha整合权重调整
        let alpha_adjusted_weights = self.integrate_alpha_signals(&vol_weights, alphas, params)?;
        
        // 4. 成本感知优化
        let cost_optimal_weights = self.cost_model.optimize_with_costs(
            positions,
            &alpha_adjusted_weights,
            params,
        )?;
        
        // 5. 计算组合指标
        let expected_return = self.calculate_expected_return(&cost_optimal_weights, alphas)?;
        let expected_vol = self.calculate_portfolio_vol(&cost_optimal_weights, &risk_matrix)?;
        let sharpe_ratio = if expected_vol > 0.0 { expected_return / expected_vol } else { 0.0 };
        
        // 6. 计算换手率和成本
        let turnover = self.calculate_turnover(positions, &cost_optimal_weights)?;
        let cost_estimate = self.cost_model.estimate_transaction_costs(positions, &cost_optimal_weights)?;
        
        // 7. 风险贡献分解
        let risk_contribution = self.calculate_risk_contributions(&cost_optimal_weights, &risk_matrix)?;
        
        // 8. 置信度评分
        let confidence_score = self.calculate_confidence_score(
            &cost_optimal_weights,
            alphas,
            &risk_matrix,
            params,
        )?;

        Ok(OptimizationResult {
            target_weights: cost_optimal_weights,
            expected_return,
            expected_vol,
            sharpe_ratio,
            turnover,
            cost_estimate,
            risk_contribution,
            optimization_method: "AG3_HYBRID".to_string(),
            confidence_score,
            timestamp: Utc::now(),
        })
    }

    /// 估计风险矩阵（协方差矩阵）
    fn estimate_risk_matrix(
        &self,
        returns_matrix: &[Vec<f64>],
        params: &OptimizationParams,
    ) -> Result<Vec<Vec<f64>>> {
        if returns_matrix.is_empty() || returns_matrix[0].is_empty() {
            return Err(anyhow::anyhow!("Empty returns matrix"));
        }

        let n_assets = returns_matrix[0].len();
        let n_periods = returns_matrix.len().min(params.lookback_period);
        
        // 计算收益率均值
        let mut means = vec![0.0; n_assets];
        for i in 0..n_assets {
            let returns: Vec<f64> = returns_matrix.iter()
                .rev()
                .take(n_periods)
                .map(|period| period[i])
                .collect();
            means[i] = returns.iter().sum::<f64>() / returns.len() as f64;
        }

        // 计算协方差矩阵
        let mut cov_matrix = vec![vec![0.0; n_assets]; n_assets];
        for i in 0..n_assets {
            for j in 0..n_assets {
                let mut covariance = 0.0;
                let count = n_periods.min(returns_matrix.len());
                
                for t in 0..count {
                    let idx = returns_matrix.len() - 1 - t;
                    let ret_i = returns_matrix[idx][i] - means[i];
                    let ret_j = returns_matrix[idx][j] - means[j];
                    covariance += ret_i * ret_j;
                }
                
                cov_matrix[i][j] = covariance / (count - 1).max(1) as f64;
            }
        }

        // 收缩估计（Ledoit-Wolf）
        if params.use_shrinkage {
            self.apply_shrinkage_estimation(&mut cov_matrix)?;
        }

        Ok(cov_matrix)
    }

    /// 应用收缩估计
    fn apply_shrinkage_estimation(&self, cov_matrix: &mut Vec<Vec<f64>>) -> Result<()> {
        let n = cov_matrix.len();
        let shrinkage_intensity = 0.2; // 收缩强度
        
        // 计算市场方差（平均对角元素）
        let market_var: f64 = (0..n).map(|i| cov_matrix[i][i]).sum::<f64>() / n as f64;
        
        // 计算平均相关系数
        let mut correlations = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if cov_matrix[i][i] > 0.0 && cov_matrix[j][j] > 0.0 {
                    let correlation = cov_matrix[i][j] / (cov_matrix[i][i] * cov_matrix[j][j]).sqrt();
                    correlations.push(correlation);
                }
            }
        }
        let avg_correlation = if correlations.is_empty() {
            0.0
        } else {
            correlations.iter().sum::<f64>() / correlations.len() as f64
        };

        // 应用收缩
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    // 对角元素：向市场方差收缩
                    cov_matrix[i][j] = (1.0 - shrinkage_intensity) * cov_matrix[i][j] + 
                                      shrinkage_intensity * market_var;
                } else {
                    // 非对角元素：向平均相关性收缩
                    let target_cov = avg_correlation * (cov_matrix[i][i] * cov_matrix[j][j]).sqrt();
                    cov_matrix[i][j] = (1.0 - shrinkage_intensity) * cov_matrix[i][j] + 
                                      shrinkage_intensity * target_cov;
                }
            }
        }

        Ok(())
    }

    /// 整合Alpha信号
    fn integrate_alpha_signals(
        &self,
        base_weights: &HashMap<String, f64>,
        alphas: &HashMap<String, f64>,
        params: &OptimizationParams,
    ) -> Result<HashMap<String, f64>> {
        let mut adjusted_weights = base_weights.clone();
        let alpha_impact = 0.3; // Alpha影响权重

        for (symbol, &base_weight) in base_weights {
            if let Some(&alpha) = alphas.get(symbol) {
                // Alpha调整：置信度加权
                let confidence_adjusted_alpha = alpha * params.alpha_confidence;
                let alpha_adjustment = confidence_adjusted_alpha * alpha_impact;
                
                // 应用调整
                let new_weight = base_weight * (1.0 + alpha_adjustment);
                adjusted_weights.insert(symbol.clone(), new_weight);
            }
        }

        // 重新标准化权重
        let total_weight: f64 = adjusted_weights.values().sum();
        if total_weight > 0.0 {
            for weight in adjusted_weights.values_mut() {
                *weight /= total_weight;
            }
        }

        Ok(adjusted_weights)
    }

    /// 计算预期收益
    fn calculate_expected_return(
        &self,
        weights: &HashMap<String, f64>,
        alphas: &HashMap<String, f64>,
    ) -> Result<f64> {
        let mut expected_return = 0.0;
        
        for (symbol, &weight) in weights {
            if let Some(&alpha) = alphas.get(symbol) {
                expected_return += weight * alpha;
            }
        }
        
        Ok(expected_return)
    }

    /// 计算组合波动率
    fn calculate_portfolio_vol(
        &self,
        weights: &HashMap<String, f64>,
        risk_matrix: &[Vec<f64>],
    ) -> Result<f64> {
        // 构建权重向量
        let symbols: Vec<String> = weights.keys().cloned().collect();
        let weight_vec: Vec<f64> = symbols.iter().map(|s| weights[s]).collect();
        
        // 计算 w^T * Σ * w
        let mut portfolio_variance = 0.0;
        for i in 0..weight_vec.len() {
            for j in 0..weight_vec.len() {
                if i < risk_matrix.len() && j < risk_matrix[0].len() {
                    portfolio_variance += weight_vec[i] * weight_vec[j] * risk_matrix[i][j];
                }
            }
        }
        
        Ok(portfolio_variance.sqrt())
    }

    /// 计算换手率
    fn calculate_turnover(
        &self,
        current_positions: &[Position],
        target_weights: &HashMap<String, f64>,
    ) -> Result<f64> {
        let mut turnover = 0.0;
        
        // 当前权重
        let current_weights: HashMap<String, f64> = current_positions.iter()
            .map(|pos| (pos.symbol.clone(), pos.weight))
            .collect();
        
        // 计算所有持仓的权重变化
        let all_symbols: std::collections::HashSet<String> = current_weights.keys()
            .chain(target_weights.keys())
            .cloned()
            .collect();
        
        for symbol in all_symbols {
            let current_weight = current_weights.get(&symbol).unwrap_or(&0.0);
            let target_weight = target_weights.get(&symbol).unwrap_or(&0.0);
            turnover += (target_weight - current_weight).abs();
        }
        
        Ok(turnover / 2.0) // 除以2因为买入和卖出重复计算
    }

    /// 计算风险贡献
    fn calculate_risk_contributions(
        &self,
        weights: &HashMap<String, f64>,
        risk_matrix: &[Vec<f64>],
    ) -> Result<HashMap<String, f64>> {
        let symbols: Vec<String> = weights.keys().cloned().collect();
        let weight_vec: Vec<f64> = symbols.iter().map(|s| weights[s]).collect();
        
        let portfolio_vol = self.calculate_portfolio_vol(weights, risk_matrix)?;
        let mut risk_contributions = HashMap::new();
        
        if portfolio_vol > 0.0 {
            for (i, symbol) in symbols.iter().enumerate() {
                let mut marginal_risk = 0.0;
                
                // 计算边际风险贡献
                for j in 0..weight_vec.len() {
                    if i < risk_matrix.len() && j < risk_matrix[0].len() {
                        marginal_risk += weight_vec[j] * risk_matrix[i][j];
                    }
                }
                
                // 风险贡献 = 权重 × 边际风险贡献 / 组合波动率
                let risk_contribution = weight_vec[i] * marginal_risk / (portfolio_vol * portfolio_vol);
                risk_contributions.insert(symbol.clone(), risk_contribution);
            }
        }
        
        Ok(risk_contributions)
    }

    /// 计算置信度评分
    fn calculate_confidence_score(
        &self,
        weights: &HashMap<String, f64>,
        alphas: &HashMap<String, f64>,
        risk_matrix: &[Vec<f64>],
        params: &OptimizationParams,
    ) -> Result<f64> {
        // 1. Alpha信号强度评分
        let alpha_strength: f64 = alphas.values().map(|&a| a.abs()).sum::<f64>() / alphas.len() as f64;
        
        // 2. 权重集中度评分（越分散越好）
        let herfindahl_index: f64 = weights.values().map(|&w| w.powi(2)).sum();
        let diversification_score = 1.0 / herfindahl_index.max(1e-6);
        
        // 3. 风险调整后收益评分
        let portfolio_vol = self.calculate_portfolio_vol(weights, risk_matrix)?;
        let risk_adjusted_score = if portfolio_vol > 0.0 && portfolio_vol <= params.target_vol * 1.2 {
            1.0
        } else {
            0.5
        };
        
        // 综合评分
        let confidence = (alpha_strength * 0.4 + diversification_score * 0.3 + risk_adjusted_score * 0.3)
            .min(1.0)
            .max(0.0);
            
        Ok(confidence)
    }

    /// 计算组合风险指标
    pub fn calculate_risk_metrics(
        &self,
        weights: &HashMap<String, f64>,
        returns_matrix: &[Vec<f64>],
        confidence_level: f64,
    ) -> Result<RiskMetrics> {
        let risk_matrix = self.estimate_risk_matrix(returns_matrix, &OptimizationParams::default())?;
        let portfolio_vol = self.calculate_portfolio_vol(weights, &risk_matrix)?;
        
        // 计算组合历史收益
        let mut portfolio_returns = Vec::new();
        let symbols: Vec<String> = weights.keys().cloned().collect();
        
        for period_returns in returns_matrix.iter().rev().take(252) { // 最近一年
            let mut portfolio_return = 0.0;
            for (i, symbol) in symbols.iter().enumerate() {
                if i < period_returns.len() {
                    portfolio_return += weights[symbol] * period_returns[i];
                }
            }
            portfolio_returns.push(portfolio_return);
        }
        
        // VaR和CVaR计算
        let mut sorted_returns = portfolio_returns.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let var_index = ((1.0 - confidence_level) * sorted_returns.len() as f64) as usize;
        let var_95 = if var_index < sorted_returns.len() {
            -sorted_returns[var_index]
        } else {
            0.0
        };
        
        let cvar_95 = if var_index > 0 {
            -sorted_returns[0..var_index].iter().sum::<f64>() / var_index as f64
        } else {
            var_95
        };
        
        // 最大回撤
        let max_drawdown = self.calculate_max_drawdown(&portfolio_returns)?;
        
        // 相关性风险（简化版）
        let correlation_risk = self.calculate_correlation_risk(&risk_matrix)?;
        
        // 集中度风险
        let concentration_risk = weights.values().map(|&w| w.powi(2)).sum::<f64>();
        
        // 尾部风险
        let tail_risk = self.calculate_tail_risk(&portfolio_returns)?;

        Ok(RiskMetrics {
            portfolio_vol,
            var_95,
            cvar_95,
            max_drawdown,
            correlation_risk,
            concentration_risk,
            tail_risk,
        })
    }

    // 辅助计算方法
    fn calculate_max_drawdown(&self, returns: &[f64]) -> Result<f64> {
        let mut max_dd = 0.0_f64;
        let mut peak = 1.0_f64;
        let mut current_value = 1.0_f64;
        
        for &ret in returns.iter().rev() {
            current_value *= 1.0 + ret;
            peak = peak.max(current_value);
            let drawdown = (peak - current_value) / peak;
            max_dd = max_dd.max(drawdown);
        }
        
        Ok(max_dd)
    }

    fn calculate_correlation_risk(&self, risk_matrix: &[Vec<f64>]) -> Result<f64> {
        let n = risk_matrix.len();
        if n <= 1 {
            return Ok(0.0);
        }
        
        let mut correlations = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if risk_matrix[i][i] > 0.0 && risk_matrix[j][j] > 0.0 {
                    let correlation = risk_matrix[i][j] / (risk_matrix[i][i] * risk_matrix[j][j]).sqrt();
                    correlations.push(correlation.abs());
                }
            }
        }
        
        if correlations.is_empty() {
            return Ok(0.0);
        }
        
        Ok(correlations.iter().sum::<f64>() / correlations.len() as f64)
    }

    fn calculate_tail_risk(&self, returns: &[f64]) -> Result<f64> {
        if returns.len() < 20 {
            return Ok(0.0);
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let mut negative_deviations = Vec::new();
        
        for &ret in returns {
            if ret < mean {
                negative_deviations.push((ret - mean).abs());
            }
        }
        
        if negative_deviations.is_empty() {
            return Ok(0.0);
        }
        
        // 计算下尾风险（负偏离的平均值）
        Ok(negative_deviations.iter().sum::<f64>() / negative_deviations.len() as f64)
    }
}

impl Default for OptimizationParams {
    fn default() -> Self {
        Self {
            target_vol: 0.15,              // 15%目标波动率
            max_position_weight: 0.20,     // 20%最大权重
            min_position_weight: 0.01,     // 1%最小权重
            rebalance_threshold: 0.05,     // 5%再平衡阈值
            cost_penalty: 0.001,           // 0.1%成本惩罚
            lookback_period: 252,          // 一年历史数据
            risk_aversion: 2.0,            // 风险厌恶系数
            use_shrinkage: true,           // 使用收缩估计
            alpha_confidence: 0.7,         // 70%Alpha置信度
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_optimizer_creation() {
        let params = OptimizationParams::default();
        let optimizer = PortfolioOptimizer::new(params);
        // 基本创建测试
    }

    #[test]
    fn test_risk_matrix_estimation() {
        let optimizer = PortfolioOptimizer::new(OptimizationParams::default());
        let returns_matrix = vec![
            vec![0.01, -0.005, 0.002],
            vec![-0.008, 0.012, -0.001],
            vec![0.003, -0.002, 0.008],
        ];
        let params = OptimizationParams::default();
        
        let risk_matrix = optimizer.estimate_risk_matrix(&returns_matrix, &params).unwrap();
        assert_eq!(risk_matrix.len(), 3);
        assert_eq!(risk_matrix[0].len(), 3);
    }
}