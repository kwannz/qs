use crate::portfolio::{Position, OptimizationParams};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Vol Targeting优化器 - AG3波动率目标化
pub struct VolTargeter {
    target_vol: f64,
    lookback_window: usize,
    rebalance_frequency: usize,
    vol_forecast_model: VolForecastModel,
}

/// 波动率预测模型
#[derive(Debug, Clone)]
pub enum VolForecastModel {
    EWMA(f64),          // 指数加权移动平均
    GARCH(f64, f64),    // GARCH(1,1)
    RealizedVol(usize), // 已实现波动率
}

/// Vol Targeting结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolTargetingResult {
    pub target_weights: HashMap<String, f64>,
    pub predicted_vol: f64,
    pub vol_scaling_factor: f64,
    pub individual_vols: HashMap<String, f64>,
    pub correlations: HashMap<(String, String), f64>,
    pub diversification_ratio: f64,
}

impl VolTargeter {
    pub fn new(target_vol: f64) -> Self {
        Self {
            target_vol,
            lookback_window: 60,
            rebalance_frequency: 5,
            vol_forecast_model: VolForecastModel::EWMA(0.94),
        }
    }

    pub fn with_model(target_vol: f64, model: VolForecastModel) -> Self {
        Self {
            target_vol,
            lookback_window: 60,
            rebalance_frequency: 5,
            vol_forecast_model: model,
        }
    }

    /// 执行Vol Targeting优化
    pub fn optimize(
        &mut self,
        positions: &[Position],
        risk_matrix: &[Vec<f64>],
        params: &OptimizationParams,
    ) -> Result<HashMap<String, f64>> {
        if positions.is_empty() {
            return Ok(HashMap::new());
        }

        // 1. 预测个体资产波动率
        let individual_vols = self.forecast_individual_volatilities(positions, risk_matrix)?;
        
        // 2. 计算基础权重（等权重或基于风险预算）
        let base_weights = self.calculate_base_weights(positions, &individual_vols, params)?;
        
        // 3. 预测组合波动率
        let portfolio_vol = self.predict_portfolio_volatility(&base_weights, risk_matrix, positions)?;
        
        // 4. 计算波动率缩放因子
        let vol_scaling_factor = if portfolio_vol > 0.0 {
            self.target_vol / portfolio_vol
        } else {
            1.0
        };
        
        // 5. 应用波动率目标化
        let mut target_weights = self.apply_vol_targeting(
            &base_weights,
            vol_scaling_factor,
            &individual_vols,
            params,
        )?;
        
        // 6. 应用约束条件
        self.apply_constraints(&mut target_weights, params)?;
        
        Ok(target_weights)
    }

    /// 预测个体资产波动率
    fn forecast_individual_volatilities(
        &self,
        positions: &[Position],
        risk_matrix: &[Vec<f64>],
    ) -> Result<HashMap<String, f64>> {
        let mut volatilities = HashMap::new();
        
        for (i, position) in positions.iter().enumerate() {
            if i < risk_matrix.len() && i < risk_matrix[i].len() {
                let variance = risk_matrix[i][i];
                let volatility = variance.sqrt();
                volatilities.insert(position.symbol.clone(), volatility);
            }
        }
        
        Ok(volatilities)
    }

    /// 计算基础权重分配
    fn calculate_base_weights(
        &self,
        positions: &[Position],
        individual_vols: &HashMap<String, f64>,
        params: &OptimizationParams,
    ) -> Result<HashMap<String, f64>> {
        let mut weights = HashMap::new();
        
        // 逆波动率加权（风险预算方法）
        let mut total_inv_vol = 0.0;
        let mut inv_vols = HashMap::new();
        
        for position in positions {
            if let Some(&vol) = individual_vols.get(&position.symbol) {
                if vol > 0.0 {
                    let inv_vol = 1.0 / vol;
                    inv_vols.insert(position.symbol.clone(), inv_vol);
                    total_inv_vol += inv_vol;
                }
            }
        }
        
        // 标准化权重
        for position in positions {
            if let Some(&inv_vol) = inv_vols.get(&position.symbol) {
                let weight = if total_inv_vol > 0.0 {
                    inv_vol / total_inv_vol
                } else {
                    1.0 / positions.len() as f64
                };
                
                // 应用权重限制
                let constrained_weight = weight
                    .max(params.min_position_weight)
                    .min(params.max_position_weight);
                
                weights.insert(position.symbol.clone(), constrained_weight);
            }
        }
        
        // 重新标准化
        let total_weight: f64 = weights.values().sum();
        if total_weight > 0.0 {
            for weight in weights.values_mut() {
                *weight /= total_weight;
            }
        }
        
        Ok(weights)
    }

    /// 预测组合波动率
    fn predict_portfolio_volatility(
        &self,
        weights: &HashMap<String, f64>,
        risk_matrix: &[Vec<f64>],
        positions: &[Position],
    ) -> Result<f64> {
        let symbols: Vec<String> = positions.iter().map(|p| p.symbol.clone()).collect();
        let weight_vec: Vec<f64> = symbols.iter().map(|s| weights.get(s).unwrap_or(&0.0)).cloned().collect();
        
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

    /// 应用波动率目标化
    fn apply_vol_targeting(
        &self,
        base_weights: &HashMap<String, f64>,
        vol_scaling_factor: f64,
        individual_vols: &HashMap<String, f64>,
        params: &OptimizationParams,
    ) -> Result<HashMap<String, f64>> {
        let mut target_weights = HashMap::new();
        
        // 基础情况：简单缩放
        if vol_scaling_factor > 0.0 && vol_scaling_factor.is_finite() {
            let effective_scaling = vol_scaling_factor.min(2.0).max(0.1); // 限制缩放范围
            
            for (symbol, &weight) in base_weights {
                target_weights.insert(symbol.clone(), weight * effective_scaling);
            }
        } else {
            target_weights = base_weights.clone();
        }
        
        // 高级Vol Targeting：考虑个体波动率变化
        if params.use_shrinkage {
            target_weights = self.apply_adaptive_vol_targeting(
                &target_weights,
                individual_vols,
                vol_scaling_factor,
            )?;
        }
        
        Ok(target_weights)
    }

    /// 自适应Vol Targeting
    fn apply_adaptive_vol_targeting(
        &self,
        weights: &HashMap<String, f64>,
        individual_vols: &HashMap<String, f64>,
        global_scaling: f64,
    ) -> Result<HashMap<String, f64>> {
        let mut adaptive_weights = HashMap::new();
        
        // 计算波动率分布统计
        let vols: Vec<f64> = individual_vols.values().cloned().collect();
        let avg_vol = vols.iter().sum::<f64>() / vols.len() as f64;
        let vol_std = {
            let variance = vols.iter().map(|v| (v - avg_vol).powi(2)).sum::<f64>() / vols.len() as f64;
            variance.sqrt()
        };
        
        for (symbol, &weight) in weights {
            if let Some(&vol) = individual_vols.get(symbol) {
                // 个体调整因子：高波动率资产降权重，低波动率资产升权重
                let vol_z_score = if vol_std > 0.0 { (vol - avg_vol) / vol_std } else { 0.0 };
                let individual_adjustment = 1.0 - (vol_z_score * 0.1); // 10%最大调整
                
                // 综合调整
                let total_adjustment = global_scaling * individual_adjustment;
                let adjusted_weight = weight * total_adjustment;
                
                adaptive_weights.insert(symbol.clone(), adjusted_weight);
            } else {
                adaptive_weights.insert(symbol.clone(), weight * global_scaling);
            }
        }
        
        Ok(adaptive_weights)
    }

    /// 应用约束条件
    fn apply_constraints(
        &self,
        weights: &mut HashMap<String, f64>,
        params: &OptimizationParams,
    ) -> Result<()> {
        // 1. 权重边界约束
        for weight in weights.values_mut() {
            *weight = weight.max(params.min_position_weight).min(params.max_position_weight);
        }
        
        // 2. 权重和标准化
        let total_weight: f64 = weights.values().sum();
        if total_weight > 0.0 {
            for weight in weights.values_mut() {
                *weight /= total_weight;
            }
        }
        
        // 3. 去除过小权重
        let min_threshold = params.min_position_weight * 0.5;
        weights.retain(|_, &mut weight| weight >= min_threshold);
        
        // 4. 再次标准化
        let final_total: f64 = weights.values().sum();
        if final_total > 0.0 {
            for weight in weights.values_mut() {
                *weight /= final_total;
            }
        }
        
        Ok(())
    }

    /// 计算分散化比率
    pub fn calculate_diversification_ratio(
        &self,
        weights: &HashMap<String, f64>,
        individual_vols: &HashMap<String, f64>,
        portfolio_vol: f64,
    ) -> Result<f64> {
        // 分散化比率 = 加权平均波动率 / 组合波动率
        let mut weighted_avg_vol = 0.0;
        let mut total_weight = 0.0;
        
        for (symbol, &weight) in weights {
            if let Some(&vol) = individual_vols.get(symbol) {
                weighted_avg_vol += weight * vol;
                total_weight += weight;
            }
        }
        
        if total_weight > 0.0 {
            weighted_avg_vol /= total_weight;
        }
        
        if portfolio_vol > 0.0 {
            Ok(weighted_avg_vol / portfolio_vol)
        } else {
            Ok(1.0)
        }
    }

    /// 更新波动率预测模型
    pub fn update_forecast_model(&mut self, returns_matrix: &[Vec<f64>]) -> Result<()> {
        match self.vol_forecast_model.clone() {
            VolForecastModel::EWMA(_lambda) => {
                // 可以根据历史数据优化lambda参数
                let optimized_lambda = self.optimize_ewma_lambda(returns_matrix)?;
                self.vol_forecast_model = VolForecastModel::EWMA(optimized_lambda);
            }
            VolForecastModel::GARCH(_alpha, _beta) => {
                // 简化的GARCH参数更新
                let (new_alpha, new_beta) = self.estimate_garch_params(returns_matrix)?;
                self.vol_forecast_model = VolForecastModel::GARCH(new_alpha, new_beta);
            }
            VolForecastModel::RealizedVol(_) => {
                // 已实现波动率模型不需要更新参数
            }
        }
        Ok(())
    }

    // 辅助方法
    fn optimize_ewma_lambda(&self, returns_matrix: &[Vec<f64>]) -> Result<f64> {
        // 简化的lambda优化：通过交叉验证选择最优lambda
        let lambdas = vec![0.90, 0.94, 0.97, 0.99];
        let mut best_lambda = 0.94;
        let mut best_score = f64::INFINITY;
        
        for &lambda in &lambdas {
            let score = self.evaluate_lambda_performance(returns_matrix, lambda)?;
            if score < best_score {
                best_score = score;
                best_lambda = lambda;
            }
        }
        
        Ok(best_lambda)
    }

    fn evaluate_lambda_performance(&self, returns_matrix: &[Vec<f64>], lambda: f64) -> Result<f64> {
        if returns_matrix.len() < 20 {
            return Ok(1.0);
        }
        
        let mut total_error = 0.0;
        let mut count = 0;
        
        // 对每个资产计算EWMA预测误差
        for asset_returns in returns_matrix.iter() {
            if asset_returns.len() < 20 {
                continue;
            }
            
            let mut ewma_var = asset_returns[0].powi(2);
            
            for i in 1..asset_returns.len() {
                let forecast_var = ewma_var;
                let realized_var = asset_returns[i].powi(2);
                
                // 预测误差
                let error = (forecast_var - realized_var).powi(2);
                total_error += error;
                count += 1;
                
                // 更新EWMA
                ewma_var = lambda * ewma_var + (1.0 - lambda) * realized_var;
            }
        }
        
        if count > 0 {
            Ok(total_error / count as f64)
        } else {
            Ok(1.0)
        }
    }

    fn estimate_garch_params(&self, returns_matrix: &[Vec<f64>]) -> Result<(f64, f64)> {
        // 简化的GARCH参数估计
        // 实际实现中应该使用最大似然估计
        Ok((0.1, 0.85)) // 典型的GARCH(1,1)参数
    }

    /// 生成Vol Targeting报告
    pub fn generate_report(
        &self,
        result: &VolTargetingResult,
        current_vol: f64,
    ) -> Result<String> {
        let mut report = String::new();
        
        report.push_str("=== Vol Targeting优化报告 ===\n");
        report.push_str(&format!("目标波动率: {:.2}%\n", self.target_vol * 100.0));
        report.push_str(&format!("预测波动率: {:.2}%\n", result.predicted_vol * 100.0));
        report.push_str(&format!("当前波动率: {:.2}%\n", current_vol * 100.0));
        report.push_str(&format!("缩放因子: {:.3}\n", result.vol_scaling_factor));
        report.push_str(&format!("分散化比率: {:.3}\n", result.diversification_ratio));
        
        report.push_str("\n--- 目标权重分配 ---\n");
        let mut weights: Vec<_> = result.target_weights.iter().collect();
        weights.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        
        for (symbol, weight) in weights.iter().take(10) {
            report.push_str(&format!("{}: {:.2}%\n", symbol, *weight * 100.0));
        }
        
        Ok(report)
    }
}

impl Default for VolTargeter {
    fn default() -> Self {
        Self::new(0.15) // 默认15%目标波动率
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_positions() -> Vec<Position> {
        vec![
            Position {
                symbol: "AAPL".to_string(),
                size: 100.0,
                market_value: 10000.0,
                weight: 0.4,
                target_weight: 0.0,
                last_update: Utc::now(),
            },
            Position {
                symbol: "GOOGL".to_string(),
                size: 50.0,
                market_value: 15000.0,
                weight: 0.6,
                target_weight: 0.0,
                last_update: Utc::now(),
            },
        ]
    }

    #[test]
    fn test_vol_targeter_creation() {
        let targeter = VolTargeter::new(0.12);
        assert_eq!(targeter.target_vol, 0.12);
    }

    #[test]
    fn test_individual_vol_forecasting() {
        let targeter = VolTargeter::new(0.15);
        let positions = create_test_positions();
        let risk_matrix = vec![
            vec![0.0004, 0.0001],
            vec![0.0001, 0.0009],
        ];
        
        let vols = targeter.forecast_individual_volatilities(&positions, &risk_matrix).unwrap();
        assert_eq!(vols.len(), 2);
        assert!(vols.contains_key("AAPL"));
        assert!(vols.contains_key("GOOGL"));
    }

    #[test]
    fn test_vol_targeting_optimization() {
        let mut targeter = VolTargeter::new(0.10);
        let positions = create_test_positions();
        let risk_matrix = vec![
            vec![0.0004, 0.0001],
            vec![0.0001, 0.0009],
        ];
        let params = OptimizationParams::default();
        
        let weights = targeter.optimize(&positions, &risk_matrix, &params).unwrap();
        
        // 验证权重和为1
        let total_weight: f64 = weights.values().sum();
        assert!((total_weight - 1.0).abs() < 1e-6);
        
        // 验证所有权重为正
        for weight in weights.values() {
            assert!(*weight >= 0.0);
        }
    }

    #[test]
    fn test_diversification_ratio() {
        let targeter = VolTargeter::new(0.15);
        let mut weights = HashMap::new();
        weights.insert("A".to_string(), 0.5);
        weights.insert("B".to_string(), 0.5);
        
        let mut vols = HashMap::new();
        vols.insert("A".to_string(), 0.20);
        vols.insert("B".to_string(), 0.30);
        
        let portfolio_vol = 0.22; // 假设由于分散化，组合波动率低于加权平均
        
        let div_ratio = targeter.calculate_diversification_ratio(&weights, &vols, portfolio_vol).unwrap();
        assert!(div_ratio > 1.0); // 分散化比率应该大于1
    }
}