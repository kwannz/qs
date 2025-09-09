//! 实验管理器

use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;
use tracing::{info, warn, error};

use crate::{
    Experiment, ExperimentStatus, TrafficSplitter, SplittingStrategy,
    MetricsCollector, StatisticalAnalyzer, ABTestResult,
    traffic_splitter::{AllocationContext, AllocationResult},
    ABTestError,
};

/// 实验管理器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentManagerConfig {
    pub database_url: String,
    pub default_significance_level: f64,
    pub default_power: f64,
    pub auto_analysis_interval_minutes: u32,
    pub max_concurrent_experiments: usize,
}

/// 实验管理器
pub struct ExperimentManager {
    config: ExperimentManagerConfig,
    experiments: Arc<RwLock<HashMap<uuid::Uuid, Experiment>>>,
    traffic_splitter: Arc<RwLock<TrafficSplitter>>,
    metrics_collector: Arc<RwLock<MetricsCollector>>,
    statistical_analyzer: StatisticalAnalyzer,
}

impl ExperimentManager {
    pub async fn new(config: ExperimentManagerConfig) -> Result<Self> {
        let experiments = Arc::new(RwLock::new(HashMap::new()));
        let traffic_splitter = Arc::new(RwLock::new(TrafficSplitter::new()));
        let metrics_collector = Arc::new(RwLock::new(MetricsCollector::new()));
        let statistical_analyzer = StatisticalAnalyzer::new(
            config.default_significance_level,
            config.default_power,
        );
        
        let manager = Self {
            config,
            experiments,
            traffic_splitter,
            metrics_collector,
            statistical_analyzer,
        };
        
        // 启动后台任务
        manager.start_background_tasks().await;
        
        Ok(manager)
    }
    
    /// 创建新实验
    pub async fn create_experiment(&self, mut experiment: Experiment) -> Result<uuid::Uuid, ABTestError> {
        // 验证实验配置
        experiment.validate().map_err(|e| ABTestError::InvalidExperimentConfig {
            reason: e.to_string(),
        })?;
        
        let experiment_id = experiment.id;
        
        // 检查并发实验数量限制
        {
            let experiments = self.experiments.read().await;
            let running_count = experiments.values()
                .filter(|e| e.status == ExperimentStatus::Running)
                .count();
            
            if running_count >= self.config.max_concurrent_experiments {
                return Err(ABTestError::InvalidExperimentConfig {
                    reason: format!("超过最大并发实验数量限制: {}", self.config.max_concurrent_experiments),
                });
            }
        }
        
        // 保存实验
        {
            let mut experiments = self.experiments.write().await;
            experiments.insert(experiment_id, experiment.clone());
        }
        
        // 保存实验名称用于日志
        let experiment_name = experiment.name.clone();
        
        // 添加到流量分配器
        {
            let mut traffic_splitter = self.traffic_splitter.write().await;
            traffic_splitter.add_experiment(experiment);
        }
        
        info!("创建实验: {} ({})", experiment_name, experiment_id);
        Ok(experiment_id)
    }
    
    /// 启动实验
    pub async fn start_experiment(&self, experiment_id: &uuid::Uuid) -> Result<(), ABTestError> {
        let mut experiments = self.experiments.write().await;
        let experiment = experiments.get_mut(experiment_id)
            .ok_or_else(|| ABTestError::ExperimentNotFound { 
                experiment_id: *experiment_id 
            })?;
        
        experiment.start().map_err(|e| ABTestError::InvalidExperimentConfig {
            reason: e.to_string(),
        })?;
        
        info!("启动实验: {} ({})", experiment.name, experiment_id);
        Ok(())
    }
    
    /// 停止实验
    pub async fn stop_experiment(&self, experiment_id: &uuid::Uuid, reason: Option<String>) -> Result<(), ABTestError> {
        let mut experiments = self.experiments.write().await;
        let experiment = experiments.get_mut(experiment_id)
            .ok_or_else(|| ABTestError::ExperimentNotFound { 
                experiment_id: *experiment_id 
            })?;
        
        experiment.stop(reason).map_err(|e| ABTestError::InvalidExperimentConfig {
            reason: e.to_string(),
        })?;
        
        info!("停止实验: {} ({})", experiment.name, experiment_id);
        Ok(())
    }
    
    /// 为用户分配实验变体
    pub async fn allocate_user(
        &self,
        experiment_id: &uuid::Uuid,
        context: &AllocationContext,
        strategy: Option<&SplittingStrategy>,
    ) -> Result<AllocationResult, ABTestError> {
        let default_strategy = SplittingStrategy::default();
        let strategy = strategy.unwrap_or(&default_strategy);
        
        let mut traffic_splitter = self.traffic_splitter.write().await;
        traffic_splitter.allocate_user(experiment_id, context, strategy)
    }
    
    /// 记录实验指标
    pub async fn record_metric(
        &self,
        user_id: String,
        experiment_id: uuid::Uuid,
        variant_id: String,
        metric_name: String,
        metric_value: f64,
        properties: Option<HashMap<String, serde_json::Value>>,
    ) {
        let mut metrics_collector = self.metrics_collector.write().await;
        metrics_collector.record_metric(
            user_id,
            experiment_id,
            variant_id,
            metric_name,
            metric_value,
            properties,
        );
    }
    
    /// 分析实验结果
    pub async fn analyze_experiment(
        &self,
        experiment_id: &uuid::Uuid,
        primary_metric: &str,
    ) -> Result<ABTestResult, ABTestError> {
        // 聚合指标
        {
            let mut metrics_collector = self.metrics_collector.write().await;
            metrics_collector.aggregate_metrics().await
                .map_err(|e| ABTestError::StatisticalAnalysisError {
                    reason: e.to_string(),
                })?;
        }
        
        // 获取实验和指标数据
        let experiment = {
            let experiments = self.experiments.read().await;
            experiments.get(experiment_id)
                .cloned()
                .ok_or_else(|| ABTestError::ExperimentNotFound {
                    experiment_id: *experiment_id,
                })?
        };
        
        let experiment_metrics = {
            let metrics_collector = self.metrics_collector.read().await;
            metrics_collector.get_experiment_metrics(experiment_id)
                .cloned()
                .ok_or_else(|| ABTestError::StatisticalAnalysisError {
                    reason: "没有找到实验指标数据".to_string(),
                })?
        };
        
        // 找到控制组变体
        let control_variant = experiment.get_control_variant()
            .ok_or_else(|| ABTestError::StatisticalAnalysisError {
                reason: "没有找到控制组变体".to_string(),
            })?;
        
        // 执行统计分析
        self.statistical_analyzer.analyze_experiment(
            &experiment_metrics,
            primary_metric,
            &control_variant.id,
        ).map_err(|e| ABTestError::StatisticalAnalysisError {
            reason: e.to_string(),
        })
    }
    
    /// 获取实验列表
    pub async fn list_experiments(&self) -> Vec<Experiment> {
        let experiments = self.experiments.read().await;
        experiments.values().cloned().collect()
    }
    
    /// 获取实验详情
    pub async fn get_experiment(&self, experiment_id: &uuid::Uuid) -> Option<Experiment> {
        let experiments = self.experiments.read().await;
        experiments.get(experiment_id).cloned()
    }
    
    /// 获取正在运行的实验
    pub async fn get_running_experiments(&self) -> Vec<Experiment> {
        let experiments = self.experiments.read().await;
        experiments.values()
            .filter(|e| e.status == ExperimentStatus::Running)
            .cloned()
            .collect()
    }
    
    /// 启动后台任务
    async fn start_background_tasks(&self) {
        let manager_clone = self.clone_for_background();
        
        // 自动分析任务
        tokio::spawn(async move {
            manager_clone.auto_analysis_loop().await;
        });
        
        // 清理过期实验任务
        let manager_clone = self.clone_for_background();
        tokio::spawn(async move {
            manager_clone.cleanup_expired_experiments_loop().await;
        });
    }
    
    /// 自动分析循环
    async fn auto_analysis_loop(&self) {
        let mut interval = tokio::time::interval(
            std::time::Duration::from_secs(self.config.auto_analysis_interval_minutes as u64 * 60)
        );
        
        loop {
            interval.tick().await;
            
            let running_experiments = self.get_running_experiments().await;
            
            for experiment in running_experiments {
                // 检查是否启用了自动停止
                if !experiment.config.auto_stop_enabled {
                    continue;
                }
                
                // 获取主要指标
                let primary_metrics = experiment.get_primary_metrics();
                if primary_metrics.is_empty() {
                    continue;
                }
                
                let primary_metric = &primary_metrics[0].name;
                
                // 执行分析
                match self.analyze_experiment(&experiment.id, primary_metric).await {
                    Ok(result) => {
                        if let Err(e) = self.handle_auto_stop_decision(&experiment, &result).await {
                            error!("自动停止决策失败: {}", e);
                        }
                    }
                    Err(e) => {
                        warn!("实验 {} 自动分析失败: {}", experiment.id, e);
                    }
                }
            }
        }
    }
    
    /// 处理自动停止决策
    async fn handle_auto_stop_decision(
        &self,
        experiment: &Experiment,
        result: &ABTestResult,
    ) -> Result<()> {
        if !experiment.config.auto_stop_enabled {
            return Ok(());
        }
        
        let auto_stop_config = experiment.config.auto_stop_config.as_ref()
            .ok_or_else(|| anyhow::anyhow!("自动停止配置不存在"))?;
        
        // 检查最小运行时间
        if let Some(started_at) = experiment.started_at {
            let runtime = Utc::now() - started_at;
            let min_runtime = Duration::hours(auto_stop_config.minimum_runtime_hours as i64);
            
            if runtime < min_runtime {
                return Ok(()); // 还未达到最小运行时间
            }
        }
        
        // 检查早停条件
        for condition in &auto_stop_config.early_stop_conditions {
            if self.evaluate_early_stop_condition(condition, result) {
                let reason = format!("满足自动停止条件: {:?}", condition.condition_type);
                self.stop_experiment(&experiment.id, Some(reason)).await
                    .map_err(|e| anyhow::anyhow!("自动停止实验失败: {}", e))?;
                
                info!("自动停止实验: {} ({})", experiment.name, experiment.id);
                break;
            }
        }
        
        Ok(())
    }
    
    /// 评估早停条件
    fn evaluate_early_stop_condition(
        &self,
        condition: &crate::experiment::EarlyStopCondition,
        result: &ABTestResult,
    ) -> bool {
        use crate::experiment::EarlyStopType;
        
        match condition.condition_type {
            EarlyStopType::StatisticalSignificance => {
                result.statistical_significance && result.confidence_level >= condition.threshold
            }
            EarlyStopType::HighRisk => {
                // 检查是否有变体表现明显差于控制组
                result.effect_size < -condition.threshold
            }
            EarlyStopType::NegligibleEffect => {
                // 效应太小且有足够把握
                result.effect_size.abs() < condition.threshold && result.confidence_level > 0.8
            }
            EarlyStopType::AdequateSampleSize => {
                // 检查样本量是否足够
                let total_samples: usize = result.variant_results.values()
                    .map(|v| v.sample_size)
                    .sum();
                total_samples >= condition.threshold as usize
            }
        }
    }
    
    /// 清理过期实验循环
    async fn cleanup_expired_experiments_loop(&self) {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(3600)); // 每小时检查一次
        
        loop {
            interval.tick().await;
            
            let experiments_to_cleanup: Vec<uuid::Uuid> = {
                let experiments = self.experiments.read().await;
                experiments.values()
                    .filter(|e| e.is_expired() && e.status == ExperimentStatus::Running)
                    .map(|e| e.id)
                    .collect()
            };
            
            for experiment_id in experiments_to_cleanup {
                if let Err(e) = self.stop_experiment(&experiment_id, Some("实验已过期".to_string())).await {
                    error!("停止过期实验 {} 失败: {}", experiment_id, e);
                }
            }
        }
    }
    
    /// 为后台任务克隆管理器
    fn clone_for_background(&self) -> Self {
        Self {
            config: self.config.clone(),
            experiments: Arc::clone(&self.experiments),
            traffic_splitter: Arc::clone(&self.traffic_splitter),
            metrics_collector: Arc::clone(&self.metrics_collector),
            statistical_analyzer: StatisticalAnalyzer::new(
                self.config.default_significance_level,
                self.config.default_power,
            ),
        }
    }
}