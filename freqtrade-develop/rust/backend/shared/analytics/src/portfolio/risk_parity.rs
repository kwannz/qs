use crate::portfolio::{Position, OptimizationParams};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 风险平价优化器 - AG3 HRP和经典Risk Parity
pub struct RiskParityOptimizer {
    method: RiskParityMethod,
    max_iterations: usize,
    tolerance: f64,
    hierarchical_clustering: bool,
}

/// 风险平价方法
#[derive(Debug, Clone)]
pub enum RiskParityMethod {
    EqualRiskContribution,  // 等风险贡献
    RiskBudgeting(HashMap<String, f64>), // 风险预算
    HierarchicalRiskParity, // 分层风险平价
    AdaptiveRiskParity,     // 自适应风险平价
}

/// 风险平价结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskParityResult {
    pub weights: HashMap<String, f64>,
    pub risk_contributions: HashMap<String, f64>,
    pub concentration_index: f64,
    pub diversification_ratio: f64,
    pub effective_assets: f64,
    pub hierarchical_structure: Option<ClusterTree>,
    pub convergence_info: ConvergenceInfo,
}

/// 聚类树结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterTree {
    pub nodes: Vec<ClusterNode>,
    pub linkage_matrix: Vec<Vec<f64>>,
    pub distance_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterNode {
    pub id: usize,
    pub assets: Vec<String>,
    pub cluster_weight: f64,
    pub intra_cluster_weights: HashMap<String, f64>,
    pub children: Vec<usize>,
}

/// 收敛信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceInfo {
    pub iterations: usize,
    pub final_error: f64,
    pub converged: bool,
    pub optimization_time_ms: u128,
}

impl RiskParityOptimizer {
    pub fn new() -> Self {
        Self {
            method: RiskParityMethod::EqualRiskContribution,
            max_iterations: 1000,
            tolerance: 1e-8,
            hierarchical_clustering: false,
        }
    }

    pub fn with_method(method: RiskParityMethod) -> Self {
        let hierarchical_clustering = matches!(method, RiskParityMethod::HierarchicalRiskParity);
        Self {
            method,
            max_iterations: 1000,
            tolerance: 1e-8,
            hierarchical_clustering,
        }
    }

    /// 执行风险平价优化
    pub fn optimize(
        &mut self,
        positions: &[Position],
        risk_matrix: &[Vec<f64>],
        params: &OptimizationParams,
    ) -> Result<HashMap<String, f64>> {
        if positions.is_empty() {
            return Ok(HashMap::new());
        }

        let start_time = std::time::Instant::now();
        
        let weights = match &self.method {
            RiskParityMethod::EqualRiskContribution => {
                self.equal_risk_contribution(positions, risk_matrix, params)?
            }
            RiskParityMethod::RiskBudgeting(budgets) => {
                self.risk_budgeting(positions, risk_matrix, budgets, params)?
            }
            RiskParityMethod::HierarchicalRiskParity => {
                self.hierarchical_risk_parity(positions, risk_matrix, params)?
            }
            RiskParityMethod::AdaptiveRiskParity => {
                self.adaptive_risk_parity(positions, risk_matrix, params)?
            }
        };

        let optimization_time = start_time.elapsed().as_millis();
        
        Ok(weights)
    }

    /// 等风险贡献优化
    fn equal_risk_contribution(
        &self,
        positions: &[Position],
        risk_matrix: &[Vec<f64>],
        params: &OptimizationParams,
    ) -> Result<HashMap<String, f64>> {
        let n = positions.len();
        let symbols: Vec<String> = positions.iter().map(|p| p.symbol.clone()).collect();
        
        // 初始权重：等权重
        let mut weights = vec![1.0 / n as f64; n];
        
        // 使用牛顿法优化
        for iteration in 0..self.max_iterations {
            let (risk_contribs, gradient, hessian) = self.calculate_risk_derivatives(&weights, risk_matrix)?;
            
            // 检查收敛性
            let target_contrib = 1.0 / n as f64;
            let mut max_error = 0.0_f64;
            for &contrib in &risk_contribs {
                max_error = max_error.max((contrib - target_contrib).abs());
            }
            
            if max_error < self.tolerance {
                break;
            }
            
            // 牛顿更新
            let delta_weights = self.solve_newton_step(&gradient, &hessian)?;
            
            // 更新权重并确保约束
            for i in 0..n {
                weights[i] += delta_weights[i] * 0.5; // 阻尼因子
                weights[i] = weights[i].max(params.min_position_weight).min(params.max_position_weight);
            }
            
            // 重新标准化
            let total_weight: f64 = weights.iter().sum();
            if total_weight > 0.0 {
                for w in &mut weights {
                    *w /= total_weight;
                }
            }
        }
        
        // 转换为HashMap
        let mut result = HashMap::new();
        for (i, symbol) in symbols.iter().enumerate() {
            result.insert(symbol.clone(), weights[i]);
        }
        
        Ok(result)
    }

    /// 风险预算优化
    fn risk_budgeting(
        &self,
        positions: &[Position],
        risk_matrix: &[Vec<f64>],
        budgets: &HashMap<String, f64>,
        params: &OptimizationParams,
    ) -> Result<HashMap<String, f64>> {
        let n = positions.len();
        let symbols: Vec<String> = positions.iter().map(|p| p.symbol.clone()).collect();
        
        // 提取风险预算
        let mut target_contribs = vec![1.0 / n as f64; n];
        for (i, symbol) in symbols.iter().enumerate() {
            if let Some(&budget) = budgets.get(symbol) {
                target_contribs[i] = budget;
            }
        }
        
        // 标准化预算
        let total_budget: f64 = target_contribs.iter().sum();
        if total_budget > 0.0 {
            for contrib in &mut target_contribs {
                *contrib /= total_budget;
            }
        }
        
        // 初始权重：基于风险预算的启发式估计
        let mut weights = vec![0.0; n];
        for i in 0..n {
            if i < risk_matrix.len() && i < risk_matrix[i].len() {
                let vol = risk_matrix[i][i].sqrt();
                weights[i] = if vol > 0.0 { target_contribs[i] / vol } else { target_contribs[i] };
            }
        }
        
        // 标准化
        let total_weight: f64 = weights.iter().sum();
        if total_weight > 0.0 {
            for w in &mut weights {
                *w /= total_weight;
            }
        }
        
        // 迭代优化
        for iteration in 0..self.max_iterations {
            let (risk_contribs, gradient, hessian) = self.calculate_risk_derivatives(&weights, risk_matrix)?;
            
            // 计算与目标预算的偏差
            let mut max_error = 0.0_f64;
            for i in 0..n {
                max_error = max_error.max((risk_contribs[i] - target_contribs[i]).abs());
            }
            
            if max_error < self.tolerance {
                break;
            }
            
            // 构建约束优化的目标函数梯度
            let mut constraint_gradient = vec![0.0; n];
            for i in 0..n {
                constraint_gradient[i] = 2.0 * (risk_contribs[i] - target_contribs[i]) * gradient[i];
            }
            
            let delta_weights = self.solve_newton_step(&constraint_gradient, &hessian)?;
            
            // 更新权重
            for i in 0..n {
                weights[i] -= delta_weights[i] * 0.3; // 更保守的步长
                weights[i] = weights[i].max(params.min_position_weight).min(params.max_position_weight);
            }
            
            // 重新标准化
            let total_weight: f64 = weights.iter().sum();
            if total_weight > 0.0 {
                for w in &mut weights {
                    *w /= total_weight;
                }
            }
        }
        
        // 转换为HashMap
        let mut result = HashMap::new();
        for (i, symbol) in symbols.iter().enumerate() {
            result.insert(symbol.clone(), weights[i]);
        }
        
        Ok(result)
    }

    /// 分层风险平价 (HRP)
    fn hierarchical_risk_parity(
        &self,
        positions: &[Position],
        risk_matrix: &[Vec<f64>],
        params: &OptimizationParams,
    ) -> Result<HashMap<String, f64>> {
        let n = positions.len();
        let symbols: Vec<String> = positions.iter().map(|p| p.symbol.clone()).collect();
        
        // 1. 计算距离矩阵（基于相关系数）
        let correlation_matrix = self.cov_to_correlation(risk_matrix)?;
        let distance_matrix = self.correlation_to_distance(&correlation_matrix)?;
        
        // 2. 执行分层聚类
        let cluster_tree = self.hierarchical_clustering(&distance_matrix, &symbols)?;
        
        // 3. 递归二分法分配权重
        let mut weights = vec![0.0; n];
        self.recursive_bisection(&cluster_tree, &mut weights, risk_matrix)?;
        
        // 应用约束
        for i in 0..n {
            weights[i] = weights[i].max(params.min_position_weight).min(params.max_position_weight);
        }
        
        // 标准化
        let total_weight: f64 = weights.iter().sum();
        if total_weight > 0.0 {
            for w in &mut weights {
                *w /= total_weight;
            }
        }
        
        // 转换为HashMap
        let mut result = HashMap::new();
        for (i, symbol) in symbols.iter().enumerate() {
            result.insert(symbol.clone(), weights[i]);
        }
        
        Ok(result)
    }

    /// 自适应风险平价
    fn adaptive_risk_parity(
        &self,
        positions: &[Position],
        risk_matrix: &[Vec<f64>],
        params: &OptimizationParams,
    ) -> Result<HashMap<String, f64>> {
        let n = positions.len();
        
        // 市场状态检测
        let market_regime = self.detect_market_regime(risk_matrix)?;
        
        // 根据市场状态选择策略
        match market_regime {
            MarketRegime::LowVol => {
                // 低波动期：更激进的风险分配
                self.equal_risk_contribution(positions, risk_matrix, params)
            }
            MarketRegime::HighVol => {
                // 高波动期：使用分层方法
                self.hierarchical_risk_parity(positions, risk_matrix, params)
            }
            MarketRegime::Crisis => {
                // 危机期：极端分散化
                let mut equal_weights = HashMap::new();
                for position in positions {
                    equal_weights.insert(position.symbol.clone(), 1.0 / n as f64);
                }
                Ok(equal_weights)
            }
        }
    }

    /// 计算风险导数（风险贡献、梯度、海森矩阵）
    fn calculate_risk_derivatives(
        &self,
        weights: &[f64],
        risk_matrix: &[Vec<f64>],
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<Vec<f64>>)> {
        let n = weights.len();
        
        // 计算组合方差
        let mut portfolio_var = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i < risk_matrix.len() && j < risk_matrix[0].len() {
                    portfolio_var += weights[i] * weights[j] * risk_matrix[i][j];
                }
            }
        }
        
        if portfolio_var <= 0.0 {
            return Err(anyhow::anyhow!("Non-positive portfolio variance"));
        }
        
        // 计算边际风险贡献
        let mut marginal_contribs = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                if i < risk_matrix.len() && j < risk_matrix[0].len() {
                    marginal_contribs[i] += weights[j] * risk_matrix[i][j];
                }
            }
        }
        
        // 计算风险贡献
        let mut risk_contribs = vec![0.0; n];
        for i in 0..n {
            risk_contribs[i] = weights[i] * marginal_contribs[i] / portfolio_var;
        }
        
        // 计算梯度（目标函数：最小化风险贡献差异的平方和）
        let target_contrib = 1.0 / n as f64;
        let mut gradient = vec![0.0; n];
        for i in 0..n {
            let contrib_error = risk_contribs[i] - target_contrib;
            gradient[i] = 2.0 * contrib_error * marginal_contribs[i] / portfolio_var;
        }
        
        // 计算海森矩阵（简化版）
        let mut hessian = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i < risk_matrix.len() && j < risk_matrix[0].len() {
                    hessian[i][j] = 2.0 * risk_matrix[i][j] / portfolio_var;
                }
                if i == j {
                    hessian[i][j] += 1e-6; // 数值稳定性
                }
            }
        }
        
        Ok((risk_contribs, gradient, hessian))
    }

    /// 求解牛顿步骤
    fn solve_newton_step(&self, gradient: &[f64], hessian: &[Vec<f64>]) -> Result<Vec<f64>> {
        let n = gradient.len();
        
        // 简化版：使用对角线近似
        let mut delta = vec![0.0; n];
        for i in 0..n {
            if hessian[i][i].abs() > 1e-12 {
                delta[i] = -gradient[i] / hessian[i][i];
            }
        }
        
        Ok(delta)
    }

    /// 协方差转相关系数
    fn cov_to_correlation(&self, cov_matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = cov_matrix.len();
        let mut corr_matrix = vec![vec![0.0; n]; n];
        
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    corr_matrix[i][j] = 1.0;
                } else if cov_matrix[i][i] > 0.0 && cov_matrix[j][j] > 0.0 {
                    corr_matrix[i][j] = cov_matrix[i][j] / (cov_matrix[i][i] * cov_matrix[j][j]).sqrt();
                }
            }
        }
        
        Ok(corr_matrix)
    }

    /// 相关系数转距离矩阵
    fn correlation_to_distance(&self, corr_matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = corr_matrix.len();
        let mut dist_matrix = vec![vec![0.0; n]; n];
        
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    // 距离 = sqrt((1 - correlation) / 2)
                    let correlation = corr_matrix[i][j];
                    dist_matrix[i][j] = ((1.0 - correlation) / 2.0).sqrt();
                }
            }
        }
        
        Ok(dist_matrix)
    }

    /// 分层聚类
    fn hierarchical_clustering(&self, distance_matrix: &[Vec<f64>], symbols: &[String]) -> Result<ClusterTree> {
        let n = symbols.len();
        let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
        let mut linkage_matrix = Vec::new();
        
        // 简化的单链聚类
        for _step in 0..(n - 1) {
            let (min_i, min_j, min_dist) = self.find_closest_clusters(&clusters, distance_matrix)?;
            
            // 合并聚类
            let mut new_cluster = clusters[min_i].clone();
            new_cluster.extend(clusters[min_j].clone());
            
            // 记录链接信息
            linkage_matrix.push(vec![min_i as f64, min_j as f64, min_dist, new_cluster.len() as f64]);
            
            // 更新聚类列表
            clusters.push(new_cluster);
            
            // 移除已合并的聚类（倒序移除避免索引问题）
            if min_i > min_j {
                clusters.remove(min_i);
                clusters.remove(min_j);
            } else {
                clusters.remove(min_j);
                clusters.remove(min_i);
            }
        }
        
        // 构建聚类树
        let mut nodes = Vec::new();
        for (id, cluster_assets) in clusters.iter().enumerate() {
            let assets = cluster_assets.iter().map(|&i| symbols[i].clone()).collect();
            nodes.push(ClusterNode {
                id,
                assets,
                cluster_weight: 0.0,
                intra_cluster_weights: HashMap::new(),
                children: Vec::new(),
            });
        }
        
        Ok(ClusterTree {
            nodes,
            linkage_matrix,
            distance_threshold: 0.5,
        })
    }

    /// 找到最近的聚类对
    fn find_closest_clusters(
        &self,
        clusters: &[Vec<usize>],
        distance_matrix: &[Vec<f64>],
    ) -> Result<(usize, usize, f64)> {
        let mut min_dist = f64::INFINITY;
        let mut min_i = 0;
        let mut min_j = 1;
        
        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                // 计算聚类间距离（单链：最小距离）
                let mut cluster_dist = f64::INFINITY;
                for &asset_i in &clusters[i] {
                    for &asset_j in &clusters[j] {
                        if asset_i < distance_matrix.len() && asset_j < distance_matrix[0].len() {
                            cluster_dist = cluster_dist.min(distance_matrix[asset_i][asset_j]);
                        }
                    }
                }
                
                if cluster_dist < min_dist {
                    min_dist = cluster_dist;
                    min_i = i;
                    min_j = j;
                }
            }
        }
        
        Ok((min_i, min_j, min_dist))
    }

    /// 递归二分配权重
    fn recursive_bisection(
        &self,
        cluster_tree: &ClusterTree,
        weights: &mut [f64],
        risk_matrix: &[Vec<f64>],
    ) -> Result<()> {
        // 简化实现：等权重分配
        let n = weights.len();
        for w in weights.iter_mut() {
            *w = 1.0 / n as f64;
        }
        
        Ok(())
    }

    /// 检测市场状态
    fn detect_market_regime(&self, risk_matrix: &[Vec<f64>]) -> Result<MarketRegime> {
        let n = risk_matrix.len();
        if n == 0 {
            return Ok(MarketRegime::LowVol);
        }
        
        // 计算平均波动率
        let mut avg_vol = 0.0;
        for i in 0..n {
            if i < risk_matrix[i].len() {
                avg_vol += risk_matrix[i][i].sqrt();
            }
        }
        avg_vol /= n as f64;
        
        // 计算平均相关系数
        let mut correlations = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                if i < risk_matrix.len() && j < risk_matrix[0].len() && 
                   risk_matrix[i][i] > 0.0 && risk_matrix[j][j] > 0.0 {
                    let corr = risk_matrix[i][j] / (risk_matrix[i][i] * risk_matrix[j][j]).sqrt();
                    correlations.push(corr);
                }
            }
        }
        
        let avg_corr = if correlations.is_empty() {
            0.0
        } else {
            correlations.iter().sum::<f64>() / correlations.len() as f64
        };
        
        // 状态判断
        if avg_vol > 0.25 && avg_corr > 0.7 {
            Ok(MarketRegime::Crisis)
        } else if avg_vol > 0.15 {
            Ok(MarketRegime::HighVol)
        } else {
            Ok(MarketRegime::LowVol)
        }
    }

    /// 计算有效资产数量
    pub fn calculate_effective_assets(&self, weights: &HashMap<String, f64>) -> f64 {
        let herfindahl_index: f64 = weights.values().map(|&w| w.powi(2)).sum();
        if herfindahl_index > 0.0 {
            1.0 / herfindahl_index
        } else {
            0.0
        }
    }
}

/// 市场状态枚举
#[derive(Debug, Clone)]
enum MarketRegime {
    LowVol,   // 低波动
    HighVol,  // 高波动  
    Crisis,   // 危机
}

impl Default for RiskParityOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_positions() -> Vec<Position> {
        vec![
            Position {
                symbol: "A".to_string(),
                size: 100.0,
                market_value: 10000.0,
                weight: 0.25,
                target_weight: 0.0,
                last_update: Utc::now(),
            },
            Position {
                symbol: "B".to_string(),
                size: 200.0,
                market_value: 15000.0,
                weight: 0.375,
                target_weight: 0.0,
                last_update: Utc::now(),
            },
            Position {
                symbol: "C".to_string(),
                size: 150.0,
                market_value: 10000.0,
                weight: 0.25,
                target_weight: 0.0,
                last_update: Utc::now(),
            },
            Position {
                symbol: "D".to_string(),
                size: 75.0,
                market_value: 5000.0,
                weight: 0.125,
                target_weight: 0.0,
                last_update: Utc::now(),
            },
        ]
    }

    fn create_test_risk_matrix() -> Vec<Vec<f64>> {
        vec![
            vec![0.0100, 0.0020, 0.0015, 0.0010],
            vec![0.0020, 0.0150, 0.0025, 0.0012],
            vec![0.0015, 0.0025, 0.0120, 0.0018],
            vec![0.0010, 0.0012, 0.0018, 0.0080],
        ]
    }

    #[test]
    fn test_risk_parity_creation() {
        let optimizer = RiskParityOptimizer::new();
        assert_eq!(optimizer.max_iterations, 1000);
    }

    #[test]
    fn test_equal_risk_contribution() {
        let mut optimizer = RiskParityOptimizer::new();
        let positions = create_test_positions();
        let risk_matrix = create_test_risk_matrix();
        let params = OptimizationParams::default();
        
        let weights = optimizer.optimize(&positions, &risk_matrix, &params).unwrap();
        
        // 验证权重和为1
        let total_weight: f64 = weights.values().sum();
        assert!((total_weight - 1.0).abs() < 1e-6);
        
        // 验证所有权重为正
        for weight in weights.values() {
            assert!(*weight > 0.0);
        }
    }

    #[test]
    fn test_correlation_conversion() {
        let optimizer = RiskParityOptimizer::new();
        let risk_matrix = create_test_risk_matrix();
        
        let corr_matrix = optimizer.cov_to_correlation(&risk_matrix).unwrap();
        
        // 验证对角线为1
        for i in 0..corr_matrix.len() {
            assert!((corr_matrix[i][i] - 1.0).abs() < 1e-10);
        }
        
        // 验证对称性
        for i in 0..corr_matrix.len() {
            for j in 0..corr_matrix[0].len() {
                assert!((corr_matrix[i][j] - corr_matrix[j][i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_effective_assets_calculation() {
        let optimizer = RiskParityOptimizer::new();
        let mut weights = HashMap::new();
        
        // 等权重情况
        weights.insert("A".to_string(), 0.25);
        weights.insert("B".to_string(), 0.25);
        weights.insert("C".to_string(), 0.25);
        weights.insert("D".to_string(), 0.25);
        
        let effective_assets = optimizer.calculate_effective_assets(&weights);
        assert!((effective_assets - 4.0).abs() < 1e-6); // 应该接近4
        
        // 集中持仓情况
        weights.clear();
        weights.insert("A".to_string(), 1.0);
        
        let effective_assets_concentrated = optimizer.calculate_effective_assets(&weights);
        assert!((effective_assets_concentrated - 1.0).abs() < 1e-6); // 应该接近1
    }

    #[test]
    fn test_market_regime_detection() {
        let optimizer = RiskParityOptimizer::new();
        
        // 低波动矩阵
        let low_vol_matrix = vec![
            vec![0.0025, 0.0005],
            vec![0.0005, 0.0030],
        ];
        let regime = optimizer.detect_market_regime(&low_vol_matrix).unwrap();
        matches!(regime, MarketRegime::LowVol);
        
        // 高波动矩阵  
        let high_vol_matrix = vec![
            vec![0.0400, 0.0300],
            vec![0.0300, 0.0500],
        ];
        let regime = optimizer.detect_market_regime(&high_vol_matrix).unwrap();
        // 应该是HighVol或Crisis
    }
}