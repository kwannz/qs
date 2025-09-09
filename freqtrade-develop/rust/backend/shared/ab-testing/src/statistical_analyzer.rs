//! 统计分析器

use anyhow::Result;
use serde::{Deserialize, Serialize};
use statrs::distribution::{StudentsT, ContinuousCDF};
use std::collections::HashMap;

use crate::{ABTestResult, VariantResult, TestRecommendation};
use crate::metrics_collector::ExperimentMetrics;

/// 统计分析器
pub struct StatisticalAnalyzer {
    significance_level: f64,
    power: f64,
}

/// 统计分析结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalResult {
    pub test_type: SignificanceTest,
    pub p_value: f64,
    pub confidence_interval: (f64, f64),
    pub effect_size: f64,
    pub statistical_significance: bool,
    pub practical_significance: bool,
    pub power_analysis: PowerAnalysis,
}

/// 显著性检验类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignificanceTest {
    TTest,
    WelchTTest,
    ChiSquare,
    FisherExact,
    MannWhitneyU,
    KolmogorovSmirnov,
}

/// 功效分析
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerAnalysis {
    pub achieved_power: f64,
    pub required_sample_size: usize,
    pub minimum_detectable_effect: f64,
    pub days_to_significance: Option<u32>,
}

impl StatisticalAnalyzer {
    pub fn new(significance_level: f64, power: f64) -> Self {
        Self {
            significance_level,
            power,
        }
    }
    
    /// 分析A/B测试结果
    pub fn analyze_experiment(
        &self,
        experiment_metrics: &ExperimentMetrics,
        primary_metric: &str,
        control_variant_id: &str,
    ) -> Result<ABTestResult> {
        let mut variant_results = HashMap::new();
        let mut control_result: Option<VariantResult> = None;
        
        // 计算各变体结果
        for (variant_id, variant_metrics) in &experiment_metrics.variant_metrics {
            let variant_result = self.calculate_variant_result(variant_metrics, primary_metric)?;
            
            if variant_id == control_variant_id {
                control_result = Some(variant_result.clone());
            }
            
            variant_results.insert(variant_id.clone(), variant_result);
        }
        
        let control_result = control_result
            .ok_or_else(|| anyhow::anyhow!("找不到控制组变体"))?;
        
        // 计算统计显著性
        let statistical_result = self.perform_significance_test(&control_result, &variant_results, primary_metric)?;
        
        // 生成建议
        let recommendation = self.generate_recommendation(&variant_results, &statistical_result)?;
        
        Ok(ABTestResult {
            experiment_id: experiment_metrics.experiment_id,
            variant_results,
            statistical_significance: statistical_result.statistical_significance,
            confidence_level: 1.0 - self.significance_level,
            p_value: statistical_result.p_value,
            effect_size: statistical_result.effect_size,
            recommendation,
        })
    }
    
    fn calculate_variant_result(
        &self,
        variant_metrics: &crate::metrics_collector::VariantMetrics,
        primary_metric: &str,
    ) -> Result<VariantResult> {
        let metric_summary = variant_metrics.metrics.get(primary_metric)
            .ok_or_else(|| anyhow::anyhow!("找不到主要指标: {}", primary_metric))?;
        
        // 计算置信区间
        let confidence_interval = self.calculate_confidence_interval(
            metric_summary.mean,
            metric_summary.std_dev,
            metric_summary.count,
        )?;
        
        // 构建指标映射
        let mut metrics = HashMap::new();
        for (name, summary) in &variant_metrics.metrics {
            metrics.insert(name.clone(), summary.mean);
        }
        
        Ok(VariantResult {
            variant_name: variant_metrics.variant_id.clone(),
            sample_size: variant_metrics.sample_size,
            conversion_rate: metric_summary.mean, // 假设主要指标是转化率
            mean_value: metric_summary.mean,
            std_deviation: metric_summary.std_dev,
            confidence_interval,
            metrics,
        })
    }
    
    fn calculate_confidence_interval(&self, mean: f64, std_dev: f64, n: usize) -> Result<(f64, f64)> {
        if n <= 1 {
            return Ok((mean, mean));
        }
        
        let df = n - 1;
        let t_dist = StudentsT::new(0.0, 1.0, df as f64)?;
        let t_critical = t_dist.inverse_cdf(1.0 - self.significance_level / 2.0);
        
        let margin_of_error = t_critical * (std_dev / (n as f64).sqrt());
        
        Ok((mean - margin_of_error, mean + margin_of_error))
    }
    
    fn perform_significance_test(
        &self,
        control: &VariantResult,
        variants: &HashMap<String, VariantResult>,
        primary_metric: &str,
    ) -> Result<StatisticalResult> {
        // 找到最佳treatment变体
        let best_treatment = variants.values()
            .filter(|v| v.variant_name != control.variant_name)
            .max_by(|a, b| a.mean_value.partial_cmp(&b.mean_value).unwrap())
            .ok_or_else(|| anyhow::anyhow!("没有找到治疗组变体"))?;
        
        // 执行t检验
        let t_stat = self.calculate_t_statistic(control, best_treatment)?;
        let df = control.sample_size + best_treatment.sample_size - 2;
        let t_dist = StudentsT::new(0.0, 1.0, df as f64)?;
        
        // 双尾检验的p值
        let p_value = 2.0 * (1.0 - t_dist.cdf(t_stat.abs()));
        
        // 效应大小（Cohen's d）
        let pooled_std = self.calculate_pooled_standard_deviation(control, best_treatment)?;
        let effect_size = (best_treatment.mean_value - control.mean_value) / pooled_std;
        
        // 置信区间
        let t_critical = t_dist.inverse_cdf(1.0 - self.significance_level / 2.0);
        let se_diff = pooled_std * ((1.0 / control.sample_size as f64) + (1.0 / best_treatment.sample_size as f64)).sqrt();
        let diff = best_treatment.mean_value - control.mean_value;
        let confidence_interval = (
            diff - t_critical * se_diff,
            diff + t_critical * se_diff,
        );
        
        // 功效分析
        let power_analysis = self.calculate_power_analysis(control, best_treatment, effect_size)?;
        
        Ok(StatisticalResult {
            test_type: SignificanceTest::TTest,
            p_value,
            confidence_interval,
            effect_size,
            statistical_significance: p_value < self.significance_level,
            practical_significance: effect_size.abs() > 0.2, // Cohen's d > 0.2 认为有实际意义
            power_analysis,
        })
    }
    
    fn calculate_t_statistic(&self, control: &VariantResult, treatment: &VariantResult) -> Result<f64> {
        let mean_diff = treatment.mean_value - control.mean_value;
        let pooled_std = self.calculate_pooled_standard_deviation(control, treatment)?;
        let se_diff = pooled_std * ((1.0 / control.sample_size as f64) + (1.0 / treatment.sample_size as f64)).sqrt();
        
        Ok(mean_diff / se_diff)
    }
    
    fn calculate_pooled_standard_deviation(&self, control: &VariantResult, treatment: &VariantResult) -> Result<f64> {
        let n1 = control.sample_size as f64;
        let n2 = treatment.sample_size as f64;
        let s1 = control.std_deviation;
        let s2 = treatment.std_deviation;
        
        let pooled_variance = ((n1 - 1.0) * s1.powi(2) + (n2 - 1.0) * s2.powi(2)) / (n1 + n2 - 2.0);
        
        Ok(pooled_variance.sqrt())
    }
    
    fn calculate_power_analysis(
        &self,
        control: &VariantResult,
        treatment: &VariantResult,
        effect_size: f64,
    ) -> Result<PowerAnalysis> {
        // 简化的功效计算
        let n = control.sample_size.min(treatment.sample_size) as f64;
        let achieved_power = self.calculate_achieved_power(effect_size, n)?;
        
        // 计算达到目标功效所需的样本量
        let required_sample_size = self.calculate_required_sample_size(effect_size, self.power)?;
        
        // 估算达到显著性的天数（假设每天新增100个样本）
        let current_total_samples = control.sample_size + treatment.sample_size;
        let days_to_significance = if current_total_samples < required_sample_size {
            Some(((required_sample_size - current_total_samples) / 100) as u32)
        } else {
            None
        };
        
        Ok(PowerAnalysis {
            achieved_power,
            required_sample_size,
            minimum_detectable_effect: effect_size.abs(),
            days_to_significance,
        })
    }
    
    fn calculate_achieved_power(&self, effect_size: f64, n: f64) -> Result<f64> {
        // 简化的功效计算公式
        let z_alpha = Self::z_score_for_alpha(self.significance_level / 2.0);
        let z_beta = effect_size * (n / 2.0).sqrt() - z_alpha;
        
        // 使用正态分布近似
        let power = if z_beta > 0.0 {
            0.5 + 0.5 * (1.0 - (-z_beta.powi(2) / 2.0).exp())
        } else {
            0.5 - 0.5 * (1.0 - (-z_beta.powi(2) / 2.0).exp())
        };
        
        Ok(power.clamp(0.0, 1.0))
    }
    
    fn calculate_required_sample_size(&self, effect_size: f64, target_power: f64) -> Result<usize> {
        let z_alpha = Self::z_score_for_alpha(self.significance_level / 2.0);
        let z_beta = Self::z_score_for_alpha(1.0 - target_power);
        
        let n = 2.0 * ((z_alpha + z_beta) / effect_size).powi(2);
        
        Ok(n.ceil() as usize)
    }
    
    fn z_score_for_alpha(alpha: f64) -> f64 {
        // 简化的z分数计算（正态分布反函数近似）
        if alpha <= 0.5 {
            // 使用近似公式
            let t = (-2.0 * alpha.ln()).sqrt();
            t - (2.515517 + 0.802853 * t + 0.010328 * t.powi(2)) / (1.0 + 1.432788 * t + 0.189269 * t.powi(2) + 0.001308 * t.powi(3))
        } else {
            -Self::z_score_for_alpha(1.0 - alpha)
        }
    }
    
    fn generate_recommendation(
        &self,
        variants: &HashMap<String, VariantResult>,
        statistical_result: &StatisticalResult,
    ) -> Result<TestRecommendation> {
        if !statistical_result.statistical_significance {
            // 检查是否需要更多样本
            if statistical_result.power_analysis.achieved_power < self.power {
                return Ok(TestRecommendation::ContinueTesting {
                    required_sample_size: statistical_result.power_analysis.required_sample_size,
                    estimated_days_remaining: statistical_result.power_analysis.days_to_significance.unwrap_or(30),
                });
            } else {
                return Ok(TestRecommendation::NoSignificantDifference {
                    power: statistical_result.power_analysis.achieved_power,
                });
            }
        }
        
        // 找到最佳变体
        let best_variant = variants.values()
            .max_by(|a, b| a.mean_value.partial_cmp(&b.mean_value).unwrap())
            .unwrap();
        
        let control_variant = variants.values().find(|v| v.variant_name.contains("control"))
            .or_else(|| variants.values().next())
            .unwrap();
        
        let improvement = (best_variant.mean_value - control_variant.mean_value) / control_variant.mean_value * 100.0;
        let confidence = (1.0 - statistical_result.p_value) * 100.0;
        
        if best_variant.variant_name == control_variant.variant_name {
            Ok(TestRecommendation::VariantAWins {
                confidence,
                improvement: improvement.abs(),
            })
        } else {
            Ok(TestRecommendation::VariantBWins {
                confidence,
                improvement: improvement.abs(),
            })
        }
    }
}