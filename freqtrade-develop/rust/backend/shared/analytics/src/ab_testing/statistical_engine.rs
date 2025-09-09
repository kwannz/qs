use anyhow::Result;

/// 假设检验结果
#[derive(Debug, Clone)]
pub struct HypothesisTestResult {
    pub p_value: f64,
    pub statistic: f64,
    pub reject_null: bool,
    pub confidence_level: f64,
}
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use statrs::distribution::{StudentsT, Normal, ContinuousCDF};
use statrs::statistics::Statistics;
use tracing::{debug, info, warn};

use super::experiment_manager::{
    StatisticalEngine, StatisticalConfig, StatisticalTest, MultipleTesting,
    ConfidenceIntervalType, StatisticalAnalysis, BayesianAnalysis,
    ConfidenceInterval, SignificanceTestResult, ExperimentConfig, ExperimentData
};

/// 默认统计引擎实现
#[derive(Debug)]
pub struct DefaultStatisticalEngine;

impl DefaultStatisticalEngine {
    pub fn new() -> Self {
        Self
    }

    /// 执行t检验
    fn perform_t_test(&self, control_data: &[f64], treatment_data: &[f64], alpha: f64) -> Result<SignificanceTestResult> {
        if control_data.is_empty() || treatment_data.is_empty() {
            return Err(anyhow::anyhow!("Empty data provided for t-test"));
        }

        let control_mean = control_data.mean();
        let treatment_mean = treatment_data.mean();
        let control_var = control_data.variance();
        let treatment_var = treatment_data.variance();
        
        let n1 = control_data.len() as f64;
        let n2 = treatment_data.len() as f64;
        
        // 计算pooled standard error
        let pooled_se = ((control_var / n1) + (treatment_var / n2)).sqrt();
        
        if pooled_se == 0.0 {
            return Err(anyhow::anyhow!("Standard error is zero"));
        }
        
        // 计算t统计量
        let t_statistic = (treatment_mean - control_mean) / pooled_se;
        
        // 计算自由度 (Welch's t-test)
        let df = ((control_var / n1 + treatment_var / n2).powi(2)) / 
                 ((control_var / n1).powi(2) / (n1 - 1.0) + (treatment_var / n2).powi(2) / (n2 - 1.0));
        
        // 计算p值 (双尾检验)
        let t_dist = StudentsT::new(0.0, 1.0, df).map_err(|e| anyhow::anyhow!("Failed to create t-distribution: {}", e))?;
        let p_value = 2.0 * (1.0 - t_dist.cdf(t_statistic.abs()));
        
        // 计算效应大小 (Cohen's d)
        let pooled_std = ((((n1 - 1.0) * control_var + (n2 - 1.0) * treatment_var) / (n1 + n2 - 2.0))).sqrt();
        let effect_size = if pooled_std > 0.0 { (treatment_mean - control_mean) / pooled_std } else { 0.0 };
        
        // 计算置信区间
        let t_critical = t_dist.inverse_cdf(1.0 - alpha / 2.0);
        let margin_of_error = t_critical * pooled_se;
        let mean_diff = treatment_mean - control_mean;
        
        let confidence_interval = ConfidenceInterval {
            lower_bound: mean_diff - margin_of_error,
            upper_bound: mean_diff + margin_of_error,
            confidence_level: 1.0 - alpha,
        };
        
        Ok(SignificanceTestResult {
            p_value,
            test_statistic: t_statistic,
            effect_size,
            confidence_interval,
            is_significant: p_value < alpha,
        })
    }

    /// 执行Welch's t检验（方差不等）
    fn perform_welch_t_test(&self, control_data: &[f64], treatment_data: &[f64], alpha: f64) -> Result<SignificanceTestResult> {
        if control_data.is_empty() || treatment_data.is_empty() {
            return Err(anyhow::anyhow!("Empty data provided for Welch's t-test"));
        }

        let control_mean = control_data.mean();
        let treatment_mean = treatment_data.mean();
        let control_var = control_data.variance();
        let treatment_var = treatment_data.variance();
        
        let n1 = control_data.len() as f64;
        let n2 = treatment_data.len() as f64;
        
        // Welch's t-test不假设方差相等
        let se_diff = ((control_var / n1) + (treatment_var / n2)).sqrt();
        
        if se_diff == 0.0 {
            return Err(anyhow::anyhow!("Standard error is zero"));
        }
        
        let t_statistic = (treatment_mean - control_mean) / se_diff;
        
        // Welch-Satterthwaite自由度
        let df = ((control_var / n1 + treatment_var / n2).powi(2)) / 
                 ((control_var / n1).powi(2) / (n1 - 1.0) + (treatment_var / n2).powi(2) / (n2 - 1.0));
        
        let t_dist = StudentsT::new(0.0, 1.0, df).map_err(|e| anyhow::anyhow!("Failed to create t-distribution: {}", e))?;
        let p_value = 2.0 * (1.0 - t_dist.cdf(t_statistic.abs()));
        
        // 计算效应大小
        let pooled_std = (control_var + treatment_var).sqrt() / 2.0_f64.sqrt();
        let effect_size = if pooled_std > 0.0 { (treatment_mean - control_mean) / pooled_std } else { 0.0 };
        
        // 计算置信区间
        let t_critical = t_dist.inverse_cdf(1.0 - alpha / 2.0);
        let margin_of_error = t_critical * se_diff;
        let mean_diff = treatment_mean - control_mean;
        
        let confidence_interval = ConfidenceInterval {
            lower_bound: mean_diff - margin_of_error,
            upper_bound: mean_diff + margin_of_error,
            confidence_level: 1.0 - alpha,
        };
        
        Ok(SignificanceTestResult {
            p_value,
            test_statistic: t_statistic,
            effect_size,
            confidence_interval,
            is_significant: p_value < alpha,
        })
    }

    /// 执行卡方检验
    fn perform_chi_square_test(&self, control_data: &[f64], treatment_data: &[f64], alpha: f64) -> Result<SignificanceTestResult> {
        // 简化的卡方检验实现，假设数据是比例数据
        let control_successes = control_data.iter().filter(|&&x| x > 0.0).count() as f64;
        let control_total = control_data.len() as f64;
        let treatment_successes = treatment_data.iter().filter(|&&x| x > 0.0).count() as f64;
        let treatment_total = treatment_data.len() as f64;
        
        let control_failures = control_total - control_successes;
        let treatment_failures = treatment_total - treatment_successes;
        
        // 2x2列联表
        let a = treatment_successes;
        let b = treatment_failures;
        let c = control_successes;
        let d = control_failures;
        let n = a + b + c + d;
        
        if n == 0.0 {
            return Err(anyhow::anyhow!("No data for chi-square test"));
        }
        
        // 计算期望频数
        let expected_a = (a + c) * (a + b) / n;
        let expected_b = (b + d) * (a + b) / n;
        let expected_c = (a + c) * (c + d) / n;
        let expected_d = (b + d) * (c + d) / n;
        
        // 检查期望频数是否足够大（每个期望频数应该≥5）
        if expected_a < 5.0 || expected_b < 5.0 || expected_c < 5.0 || expected_d < 5.0 {
            warn!("Expected frequencies are too low for chi-square test");
        }
        
        // 计算卡方统计量
        let chi_square = 
            (a - expected_a).powi(2) / expected_a +
            (b - expected_b).powi(2) / expected_b +
            (c - expected_c).powi(2) / expected_c +
            (d - expected_d).powi(2) / expected_d;
        
        // 自由度为1（2x2表）
        let df = 1.0;
        
        // 计算p值（需要卡方分布，这里简化处理）
        let p_value = if chi_square > 3.841 { 0.05 } else { 0.1 }; // 简化的p值估算
        
        // 计算效应大小（Cramer's V）
        let effect_size = (chi_square / n).sqrt();
        
        // 简化的置信区间
        let confidence_interval = ConfidenceInterval {
            lower_bound: -0.1,
            upper_bound: 0.1,
            confidence_level: 1.0 - alpha,
        };
        
        Ok(SignificanceTestResult {
            p_value,
            test_statistic: chi_square,
            effect_size,
            confidence_interval,
            is_significant: p_value < alpha,
        })
    }

    /// 执行Mann-Whitney U检验
    fn perform_mann_whitney_u_test(&self, control_data: &[f64], treatment_data: &[f64], alpha: f64) -> Result<SignificanceTestResult> {
        let n1 = control_data.len();
        let n2 = treatment_data.len();
        
        if n1 == 0 || n2 == 0 {
            return Err(anyhow::anyhow!("Empty data for Mann-Whitney U test"));
        }
        
        // 合并并排序数据，同时记录来源
        let mut combined: Vec<(f64, bool)> = Vec::with_capacity(n1 + n2);
        for &value in control_data {
            combined.push((value, false)); // false表示控制组
        }
        for &value in treatment_data {
            combined.push((value, true)); // true表示处理组
        }
        combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // 计算秩次
        let mut ranks = vec![0.0; n1 + n2];
        let mut i = 0;
        while i < combined.len() {
            let mut j = i;
            while j < combined.len() && combined[j].0 == combined[i].0 {
                j += 1;
            }
            // 处理并列值的平均秩次
            let avg_rank = ((i + 1) + j) as f64 / 2.0;
            for k in i..j {
                ranks[k] = avg_rank;
            }
            i = j;
        }
        
        // 计算处理组的秩和
        let mut r2 = 0.0;
        for (idx, &(_, is_treatment)) in combined.iter().enumerate() {
            if is_treatment {
                r2 += ranks[idx];
            }
        }
        
        // 计算U统计量
        let u2 = r2 - (n2 * (n2 + 1)) as f64 / 2.0;
        let u1 = (n1 * n2) as f64 - u2;
        
        // 使用较小的U值
        let u = u1.min(u2);
        
        // 计算标准化的U统计量（正态近似）
        let mu = (n1 * n2) as f64 / 2.0;
        let sigma = ((n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0).sqrt();
        
        if sigma == 0.0 {
            return Err(anyhow::anyhow!("Standard deviation is zero"));
        }
        
        let z = (u - mu) / sigma;
        
        // 计算p值（双尾检验）
        let normal = Normal::new(0.0, 1.0).map_err(|e| anyhow::anyhow!("Failed to create normal distribution: {}", e))?;
        let p_value = 2.0 * (1.0 - normal.cdf(z.abs()));
        
        // 计算效应大小（r = Z / sqrt(N)）
        let effect_size = z.abs() / ((n1 + n2) as f64).sqrt();
        
        // 简化的置信区间
        let confidence_interval = ConfidenceInterval {
            lower_bound: -0.5,
            upper_bound: 0.5,
            confidence_level: 1.0 - alpha,
        };
        
        Ok(SignificanceTestResult {
            p_value,
            test_statistic: z,
            effect_size,
            confidence_interval,
            is_significant: p_value < alpha,
        })
    }

    /// 应用多重检验校正
    fn apply_multiple_testing_correction(&self, p_values: &[f64], correction: &MultipleTesting, alpha: f64) -> Vec<f64> {
        match correction {
            MultipleTesting::None => p_values.to_vec(),
            MultipleTesting::Bonferroni => {
                // Bonferroni校正：调整后的alpha = alpha / m
                let m = p_values.len() as f64;
                p_values.iter().map(|&p| (p * m).min(1.0)).collect()
            },
            MultipleTesting::BenjaminiHochberg => {
                // Benjamini-Hochberg (FDR)校正
                let mut indexed_p_values: Vec<(usize, f64)> = p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();
                indexed_p_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                
                let m = p_values.len() as f64;
                let mut adjusted = vec![0.0; p_values.len()];
                
                // 从最大的p值开始调整
                let mut min_adjusted = 1.0;
                for (rank, &(original_idx, p)) in indexed_p_values.iter().rev().enumerate() {
                    let i = p_values.len() - rank; // BH排名从1开始
                    let bh_adjusted = (p * m / i as f64).min(min_adjusted);
                    adjusted[original_idx] = bh_adjusted;
                    min_adjusted = bh_adjusted;
                }
                
                adjusted
            },
            MultipleTesting::Holm => {
                // Holm校正（逐步Bonferroni）
                let mut indexed_p_values: Vec<(usize, f64)> = p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();
                indexed_p_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                
                let m = p_values.len() as f64;
                let mut adjusted = vec![0.0; p_values.len()];
                
                let mut max_adjusted = 0.0;
                for (rank, &(original_idx, p)) in indexed_p_values.iter().enumerate() {
                    let holm_adjusted = (p * (m - rank as f64)).max(max_adjusted).min(1.0);
                    adjusted[original_idx] = holm_adjusted;
                    max_adjusted = holm_adjusted;
                }
                
                adjusted
            },
            MultipleTesting::Sidak => {
                // Šidák校正
                let m = p_values.len() as f64;
                p_values.iter().map(|&p| 1.0 - (1.0 - p).powf(m)).collect()
            },
        }
    }

    /// 执行贝叶斯分析
    fn perform_bayesian_analysis(&self, control_data: &[f64], treatment_data: &[f64]) -> Result<BayesianAnalysis> {
        // 简化的贝叶斯分析实现
        let control_mean = control_data.mean();
        let treatment_mean = treatment_data.mean();
        
        // 计算优势概率（简化版本）
        let prob_treatment_better = if treatment_mean > control_mean { 0.8 } else { 0.2 };
        
        let mut probability_of_superiority = HashMap::new();
        probability_of_superiority.insert("treatment".to_string(), prob_treatment_better);
        probability_of_superiority.insert("control".to_string(), 1.0 - prob_treatment_better);
        
        // 计算期望损失（简化版本）
        let mut expected_loss = HashMap::new();
        expected_loss.insert("treatment".to_string(), (control_mean - treatment_mean).max(0.0));
        expected_loss.insert("control".to_string(), (treatment_mean - control_mean).max(0.0));
        
        // 可信区间（简化版本）
        let mut credible_intervals = HashMap::new();
        credible_intervals.insert("treatment".to_string(), ConfidenceInterval {
            lower_bound: treatment_mean - 0.1,
            upper_bound: treatment_mean + 0.1,
            confidence_level: 0.95,
        });
        credible_intervals.insert("control".to_string(), ConfidenceInterval {
            lower_bound: control_mean - 0.1,
            upper_bound: control_mean + 0.1,
            confidence_level: 0.95,
        });
        
        // 后验分布（简化版本）
        let mut posterior_distributions = HashMap::new();
        posterior_distributions.insert("treatment".to_string(), vec![treatment_mean; 100]);
        posterior_distributions.insert("control".to_string(), vec![control_mean; 100]);
        
        Ok(BayesianAnalysis {
            probability_of_superiority,
            expected_loss,
            credible_intervals,
            posterior_distributions,
        })
    }

    /// 计算置信区间
    fn calculate_confidence_interval(
        &self, 
        data: &[f64], 
        confidence_level: f64, 
        interval_type: &ConfidenceIntervalType
    ) -> Result<ConfidenceInterval> {
        if data.is_empty() {
            return Err(anyhow::anyhow!("Empty data for confidence interval"));
        }
        
        let mean = data.mean();
        let std_error = (data.variance() / data.len() as f64).sqrt();
        let alpha = 1.0 - confidence_level;
        
        let (lower_bound, upper_bound) = match interval_type {
            ConfidenceIntervalType::Normal => {
                let normal = Normal::new(0.0, 1.0).map_err(|e| anyhow::anyhow!("Failed to create normal distribution: {}", e))?;
                let z_critical = normal.inverse_cdf(1.0 - alpha / 2.0);
                let margin_of_error = z_critical * std_error;
                (mean - margin_of_error, mean + margin_of_error)
            },
            ConfidenceIntervalType::Bootstrap => {
                // 简化的Bootstrap置信区间
                let lower_percentile = (alpha / 2.0 * data.len() as f64) as usize;
                let upper_percentile = ((1.0 - alpha / 2.0) * data.len() as f64) as usize;
                
                let mut sorted_data = data.to_vec();
                sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
                
                let lower_bound = sorted_data.get(lower_percentile).copied().unwrap_or(mean - std_error);
                let upper_bound = sorted_data.get(upper_percentile).copied().unwrap_or(mean + std_error);
                
                (lower_bound, upper_bound)
            },
            ConfidenceIntervalType::Bayesian => {
                // 简化的贝叶斯置信区间（实际应该是可信区间）
                (mean - 1.96 * std_error, mean + 1.96 * std_error)
            },
        };
        
        Ok(ConfidenceInterval {
            lower_bound,
            upper_bound,
            confidence_level,
        })
    }

    /// 计算统计功效
    fn calculate_power(&self, effect_size: f64, sample_size: usize, alpha: f64) -> f64 {
        // 简化的功效计算
        // 实际实现需要更复杂的计算
        if effect_size == 0.0 {
            return alpha; // 零假设为真时的功效等于α
        }
        
        let n = sample_size as f64;
        let z_alpha = 1.96; // 对于α=0.05的双尾检验
        let z_beta = effect_size * (n / 2.0).sqrt() - z_alpha;
        
        let normal = Normal::new(0.0, 1.0).unwrap();
        normal.cdf(z_beta).max(alpha)
    }

    /// 评估样本量充足性
    fn assess_sample_size_adequacy(&self, control_size: usize, treatment_size: usize, effect_size: f64, power: f64, alpha: f64) -> bool {
        let required_per_group = self.calculate_sample_size_per_group(effect_size, power, alpha);
        control_size >= required_per_group && treatment_size >= required_per_group
    }

    /// 计算每组所需样本量
    fn calculate_sample_size_per_group(&self, effect_size: f64, power: f64, alpha: f64) -> usize {
        if effect_size == 0.0 {
            return usize::MAX; // 无效应时需要无限大样本
        }
        
        // 简化的样本量计算公式
        let z_alpha = 1.96; // 对于α=0.05的双尾检验
        let z_beta = 0.84;  // 对于80%功效的近似值
        
        let n = 2.0 * ((z_alpha + z_beta) / effect_size).powi(2);
        n.ceil() as usize
    }
}

impl StatisticalEngine for DefaultStatisticalEngine {
    fn analyze_experiment(&self, experiment: &ExperimentConfig, data: &ExperimentData) -> Result<StatisticalAnalysis> {
        info!("Performing statistical analysis for experiment: {}", experiment.id);
        
        // 提取控制组和处理组数据
        let control_variant = &experiment.control_group;
        let mut control_data = Vec::new();
        let mut treatment_data = Vec::new();
        
        // 从实验数据中提取主要指标的数据
        if let Some(primary_metric) = experiment.target_metrics.iter().find(|m| m.is_primary) {
            for (_, metric_observations) in &data.metrics {
                for observation in metric_observations {
                    if observation.metric_name == primary_metric.name {
                        if observation.variant == *control_variant {
                            control_data.push(observation.value);
                        } else {
                            treatment_data.push(observation.value);
                        }
                    }
                }
            }
        }
        
        if control_data.is_empty() || treatment_data.is_empty() {
            return Err(anyhow::anyhow!("Insufficient data for statistical analysis"));
        }
        
        // 执行显著性检验
        let test_result = self.perform_significance_test(&control_data, &treatment_data)?;
        
        // 应用多重检验校正（如果启用）
        let corrected_p_values = if experiment.target_metrics.len() > 1 {
            let p_values = vec![test_result.p_value]; // 简化：只考虑主要指标
            self.apply_multiple_testing_correction(
                &p_values, 
                &experiment.statistical_config.multiple_testing_correction, 
                experiment.significance_level
            )
        } else {
            vec![test_result.p_value]
        };
        
        let is_significant = corrected_p_values[0] < experiment.significance_level;
        
        // 确定获胜变体
        let winning_variant = if is_significant {
            let treatment_mean = treatment_data.clone().mean();
            let control_mean = control_data.clone().mean();
            
            // 假设我们总是比较处理组和控制组
            if treatment_mean > control_mean {
                Some("treatment".to_string()) // 实际应该从实验配置中获取处理组ID
            } else {
                Some(control_variant.clone())
            }
        } else {
            None
        };
        
        // 计算达到的统计功效
        let power_achieved = self.calculate_power(
            test_result.effect_size.abs(),
            control_data.len() + treatment_data.len(),
            experiment.significance_level,
        );
        
        // 评估样本量充足性
        let sample_size_adequate = self.assess_sample_size_adequacy(
            control_data.len(),
            treatment_data.len(),
            experiment.minimum_detectable_effect,
            experiment.power,
            experiment.significance_level,
        );
        
        // 贝叶斯分析（如果启用）
        let bayesian_analysis = if experiment.statistical_config.bayesian_analysis_enabled {
            Some(self.perform_bayesian_analysis(&control_data, &treatment_data)?)
        } else {
            None
        };
        
        Ok(StatisticalAnalysis {
            overall_significance: is_significant,
            winning_variant,
            effect_size: test_result.effect_size,
            power_achieved,
            sample_size_adequacy: sample_size_adequate,
            multiple_testing_correction_applied: experiment.target_metrics.len() > 1,
            bayesian_analysis,
        })
    }
    
    fn calculate_sample_size(&self, config: &StatisticalConfig, effect_size: f64, power: f64, alpha: f64) -> Result<usize> {
        let sample_size_per_group = self.calculate_sample_size_per_group(effect_size, power, alpha);
        
        // 总样本量（控制组 + 处理组）
        let total_sample_size = sample_size_per_group * 2;
        
        debug!("Calculated sample size: {} per group, {} total", 
               sample_size_per_group, total_sample_size);
        
        Ok(total_sample_size)
    }
    
    fn perform_significance_test(&self, control_data: &[f64], treatment_data: &[f64]) -> Result<SignificanceTestResult> {
        // 默认使用Welch's t-test
        self.perform_welch_t_test(control_data, treatment_data, 0.05)
    }
}

/// 统计引擎工厂
pub struct StatisticalEngineFactory;

impl StatisticalEngineFactory {
    /// 根据配置创建统计引擎
    pub fn create_engine(test_type: &StatisticalTest) -> Box<dyn StatisticalEngine + Send + Sync> {
        match test_type {
            StatisticalTest::TTest | 
            StatisticalTest::WelchTTest | 
            StatisticalTest::ChiSquareTest | 
            StatisticalTest::FisherExactTest | 
            StatisticalTest::MannWhitneyU |
            StatisticalTest::BayesianTest => {
                Box::new(DefaultStatisticalEngine::new())
            }
        }
    }
}

/// 高级统计工具
pub struct AdvancedStatisticalTools;

impl AdvancedStatisticalTools {
    /// 执行顺序分析
    pub fn sequential_analysis(
        cumulative_data: &[(f64, f64)], // (control, treatment) 对
        alpha: f64,
        beta: f64,
        effect_size: f64,
    ) -> Result<SequentialTestResult> {
        // 简化的顺序分析实现
        let n = cumulative_data.len();
        
        if n == 0 {
            return Ok(SequentialTestResult {
                decision: SequentialDecision::Continue,
                log_likelihood_ratio: 0.0,
                upper_boundary: 0.0,
                lower_boundary: 0.0,
                sample_size: 0,
            });
        }
        
        // 计算对数似然比
        let mut log_likelihood_ratio = 0.0;
        for &(control_val, treatment_val) in cumulative_data {
            // 简化的似然比计算
            let diff = treatment_val - control_val;
            log_likelihood_ratio += diff * effect_size;
        }
        
        // 计算边界
        let upper_boundary = (1.0 - beta).ln() / alpha.ln();
        let lower_boundary = beta.ln() / (1.0 - alpha).ln();
        
        let decision = if log_likelihood_ratio >= upper_boundary {
            SequentialDecision::RejectNull // 处理组显著更好
        } else if log_likelihood_ratio <= lower_boundary {
            SequentialDecision::AcceptNull // 没有显著差异
        } else {
            SequentialDecision::Continue // 继续收集数据
        };
        
        Ok(SequentialTestResult {
            decision,
            log_likelihood_ratio,
            upper_boundary,
            lower_boundary,
            sample_size: n,
        })
    }

    /// 计算最小可检测效应
    pub fn minimum_detectable_effect(
        sample_size_per_group: usize,
        power: f64,
        alpha: f64,
        baseline_variance: f64,
    ) -> f64 {
        let n = sample_size_per_group as f64;
        let z_alpha = 1.96; // 对于α=0.05
        let z_beta = 0.84;  // 对于80%功效
        
        let mde = (z_alpha + z_beta) * (2.0 * baseline_variance / n).sqrt();
        mde
    }
}

/// 顺序测试结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequentialTestResult {
    pub decision: SequentialDecision,
    pub log_likelihood_ratio: f64,
    pub upper_boundary: f64,
    pub lower_boundary: f64,
    pub sample_size: usize,
}

/// 顺序决策
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SequentialDecision {
    Continue,      // 继续收集数据
    RejectNull,    // 拒绝零假设
    AcceptNull,    // 接受零假设
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t_test() {
        let engine = DefaultStatisticalEngine::new();
        
        let control_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment_data = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        
        let result = engine.perform_significance_test(&control_data, &treatment_data).unwrap();
        
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
        assert!(result.effect_size > 0.0); // treatment 平均值更高
        assert!(result.confidence_interval.lower_bound < result.confidence_interval.upper_bound);
    }

    #[test]
    fn test_sample_size_calculation() {
        let engine = DefaultStatisticalEngine::new();
        let config = StatisticalConfig {
            test_type: StatisticalTest::TTest,
            multiple_testing_correction: MultipleTesting::None,
            sequential_testing_enabled: false,
            bayesian_analysis_enabled: false,
            confidence_interval_type: ConfidenceIntervalType::Normal,
            bootstrap_samples: 1000,
        };
        
        let sample_size = engine.calculate_sample_size(&config, 0.2, 0.8, 0.05).unwrap();
        assert!(sample_size > 0);
        assert!(sample_size < 10000); // 合理的样本量范围
    }

    #[test]
    fn test_multiple_testing_correction() {
        let engine = DefaultStatisticalEngine::new();
        let p_values = vec![0.01, 0.03, 0.05, 0.02];
        
        // Bonferroni校正
        let bonferroni = engine.apply_multiple_testing_correction(
            &p_values, 
            &MultipleTesting::Bonferroni, 
            0.05
        );
        assert!(bonferroni[0] > p_values[0]); // 校正后p值应该更大
        
        // BH校正
        let bh = engine.apply_multiple_testing_correction(
            &p_values, 
            &MultipleTesting::BenjaminiHochberg, 
            0.05
        );
        assert!(bh.len() == p_values.len());
    }

    #[test]
    fn test_welch_t_test() {
        let engine = DefaultStatisticalEngine::new();
        
        // 不同方差的两组数据
        let control_data = vec![1.0, 1.1, 0.9, 1.2, 0.8];  // 低方差
        let treatment_data = vec![5.0, 3.0, 7.0, 4.0, 6.0]; // 高方差，高均值
        
        let result = engine.perform_welch_t_test(&control_data, &treatment_data, 0.05).unwrap();
        
        assert!(result.p_value < 0.05); // 应该显著
        assert!(result.is_significant);
        assert!(result.effect_size > 0.0); // 处理组效应更好
    }

    #[test]
    fn test_mann_whitney_u_test() {
        let engine = DefaultStatisticalEngine::new();
        
        let control_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let treatment_data = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        
        let result = engine.perform_mann_whitney_u_test(&control_data, &treatment_data, 0.05).unwrap();
        
        assert!(result.p_value < 0.05); // 应该显著不同
        assert!(result.is_significant);
    }
}