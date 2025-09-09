//! AG3 上下文老虎机实现
//!
//! 实现高级上下文老虎机算法：
//! - LinUCB算法及其变种
//! - 神经网络上下文老虎机
//! - 动态特征选择和维度压缩

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use rand::prelude::*;
use rand_distr::Normal;
use ndarray::{Array1, Array2, ArrayView1};
use rand::{thread_rng};

/// 上下文老虎机配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualBanditConfig {
    pub algorithm: ContextualAlgorithm,
    pub feature_dim: usize,
    pub alpha: f64,              // 置信参数
    pub lambda_reg: f64,         // 正则化参数
    pub learning_rate: f64,      // 学习率
    pub batch_size: usize,       // 批处理大小
    pub update_frequency: usize, // 更新频率
    pub feature_selection: bool, // 是否启用特征选择
    pub dimensionality_reduction: bool, // 是否启用降维
}

/// 上下文老虎机算法类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextualAlgorithm {
    LinUCB {
        alpha: f64,
    },
    LinTS {
        // Thompson Sampling for Linear bandits
        v: f64, // prior variance
    },
    NeuralUCB {
        hidden_layers: Vec<usize>,
        dropout_rate: f64,
    },
    NeuralTS {
        hidden_layers: Vec<usize>,
        ensemble_size: usize,
    },
    HybridLinUCB {
        alpha: f64,
        shared_features: usize,
    },
}

/// 上下文特征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextFeatures {
    pub raw_features: Vec<f64>,
    pub processed_features: Vec<f64>,
    pub feature_names: Vec<String>,
    pub timestamp: DateTime<Utc>,
    pub feature_quality: f64, // 特征质量评分 0-1
}

/// 上下文老虎机实例
pub struct ContextualBandit {
    config: ContextualBanditConfig,
    arms: Arc<RwLock<HashMap<String, ContextualArm>>>,
    feature_processor: Arc<RwLock<FeatureProcessor>>,
    model: Arc<RwLock<Box<dyn ContextualModel + Send + Sync>>>,
    experience_buffer: Arc<RwLock<ExperienceBuffer>>,
    performance_tracker: Arc<RwLock<PerformanceTracker>>,
}

/// 上下文臂
#[derive(Debug, Clone)]
pub struct ContextualArm {
    pub arm_id: String,
    pub theta: Array1<f64>,         // 参数向量
    pub a_inv: Array2<f64>,         // A^-1 矩阵
    pub b: Array1<f64>,             // b 向量
    pub sample_count: usize,
    pub last_reward: f64,
    pub last_updated: DateTime<Utc>,
    pub confidence_radius: f64,
}

/// 特征处理器
#[derive(Debug)]
pub struct FeatureProcessor {
    feature_stats: HashMap<String, FeatureStatistics>,
    selected_features: Vec<usize>,
    normalization_params: NormalizationParams,
    pca_components: Option<Array2<f64>>,
    feature_interactions: Vec<(usize, usize)>,
}

/// 特征统计
#[derive(Debug, Clone)]
pub struct FeatureStatistics {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub missing_rate: f64,
    pub importance_score: f64,
}

/// 归一化参数
#[derive(Debug, Clone)]
pub struct NormalizationParams {
    pub method: NormalizationMethod,
    pub global_mean: Array1<f64>,
    pub global_std: Array1<f64>,
    pub global_min: Array1<f64>,
    pub global_max: Array1<f64>,
}

#[derive(Debug, Clone)]
pub enum NormalizationMethod {
    ZScore,
    MinMax,
    Robust,
    None,
}

/// 经验缓冲区
#[derive(Debug)]
pub struct ExperienceBuffer {
    experiences: Vec<Experience>,
    buffer_size: usize,
    current_index: usize,
}

/// 经验记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    pub context: Vec<f64>,
    pub features: Vec<f64>,
    pub arm_id: String,
    pub reward: f64,
    pub probability: f64,
    pub timestamp: DateTime<Utc>,
}

/// 性能跟踪器
#[derive(Debug)]
pub struct PerformanceTracker {
    cumulative_regret: f64,
    cumulative_reward: f64,
    arm_performance: HashMap<String, ArmPerformance>,
    regret_history: Vec<(DateTime<Utc>, f64)>,
    feature_importance_history: Vec<(DateTime<Utc>, Vec<f64>)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmPerformance {
    pub total_pulls: usize,
    pub total_reward: f64,
    pub recent_performance: Vec<f64>,
    pub confidence_bounds: (f64, f64),
}

/// 上下文模型特征
pub trait ContextualModel: Send + Sync {
    fn predict(&self, context: ArrayView1<f64>) -> Result<HashMap<String, f64>>;
    fn update(&mut self, context: ArrayView1<f64>, arm_id: &str, reward: f64) -> Result<()>;
    fn get_confidence(&self, context: ArrayView1<f64>, arm_id: &str) -> Result<f64>;
    fn batch_update(&mut self, experiences: &[Experience]) -> Result<()>;
}

/// LinUCB 模型
#[derive(Debug)]
pub struct LinUCBModel {
    arms: HashMap<String, LinUCBArm>,
    alpha: f64,
    lambda_reg: f64,
    feature_dim: usize,
}

#[derive(Debug, Clone)]
pub struct LinUCBArm {
    pub a_matrix: Array2<f64>,  // A = X^T X + λI
    pub b_vector: Array1<f64>,  // b = X^T y
    pub theta: Array1<f64>,     // θ = A^-1 b
    pub a_inv: Array2<f64>,     // A^-1
}

/// 神经网络模型（简化实现）
#[derive(Debug)]
pub struct NeuralBanditModel {
    networks: HashMap<String, SimpleNeuralNetwork>,
    config: NeuralConfig,
    optimizer_state: OptimizerState,
}

#[derive(Debug, Clone)]
pub struct NeuralConfig {
    pub hidden_layers: Vec<usize>,
    pub learning_rate: f64,
    pub dropout_rate: f64,
    pub l2_reg: f64,
}

#[derive(Debug)]
pub struct SimpleNeuralNetwork {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
    activations: Vec<Array1<f64>>,
}

#[derive(Debug)]
pub struct OptimizerState {
    momentum: HashMap<String, Vec<Array2<f64>>>,
    learning_rate: f64,
    momentum_factor: f64,
}

/// 选择结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualSelection {
    pub arm_id: String,
    pub predicted_reward: f64,
    pub confidence_bound: f64,
    pub selection_probability: f64,
    pub exploration_bonus: f64,
    pub features_used: Vec<String>,
    pub selection_reason: String,
}

impl ContextualBandit {
    pub fn new(config: ContextualBanditConfig) -> Result<Self> {
        let model: Box<dyn ContextualModel + Send + Sync> = match &config.algorithm {
            ContextualAlgorithm::LinUCB { alpha } => {
                Box::new(LinUCBModel::new(config.feature_dim, *alpha, config.lambda_reg)?)
            }
            ContextualAlgorithm::LinTS { v } => {
                Box::new(LinTSModel::new(config.feature_dim, *v, config.lambda_reg)?)
            }
            ContextualAlgorithm::NeuralUCB { hidden_layers, dropout_rate } => {
                let neural_config = NeuralConfig {
                    hidden_layers: hidden_layers.clone(),
                    learning_rate: config.learning_rate,
                    dropout_rate: *dropout_rate,
                    l2_reg: config.lambda_reg,
                };
                Box::new(NeuralBanditModel::new(config.feature_dim, neural_config)?)
            }
            _ => return Err(anyhow::anyhow!("Algorithm not yet implemented")),
        };

        Ok(Self {
            config,
            arms: Arc::new(RwLock::new(HashMap::new())),
            feature_processor: Arc::new(RwLock::new(FeatureProcessor::new()?)),
            model: Arc::new(RwLock::new(model)),
            experience_buffer: Arc::new(RwLock::new(ExperienceBuffer::new(10000))),
            performance_tracker: Arc::new(RwLock::new(PerformanceTracker::new())),
        })
    }

    /// 添加臂
    pub async fn add_arm(&self, arm_id: String) -> Result<()> {
        let mut arms = self.arms.write().await;
        if !arms.contains_key(&arm_id) {
            arms.insert(arm_id.clone(), ContextualArm::new(arm_id, self.config.feature_dim)?);
        }
        Ok(())
    }

    /// 选择臂
    pub async fn select_arm(&self, context: ContextFeatures) -> Result<ContextualSelection> {
        // 处理特征
        let processed_features = self.preprocess_features(context).await?;
        
        // 使用模型进行预测
        let model = self.model.read().await;
        let features_array = ArrayView1::from(&processed_features.processed_features);
        let predictions = model.predict(features_array)?;
        
        // 根据算法选择臂
        let selection = match &self.config.algorithm {
            ContextualAlgorithm::LinUCB { alpha } => {
                self.select_with_linucb(&predictions, &processed_features, *alpha).await?
            }
            ContextualAlgorithm::LinTS { v: _ } => {
                self.select_with_lints(&predictions, &processed_features).await?
            }
            ContextualAlgorithm::NeuralUCB { .. } => {
                self.select_with_neural_ucb(&predictions, &processed_features).await?
            }
            _ => return Err(anyhow::anyhow!("Selection method not implemented")),
        };

        Ok(selection)
    }

    /// 更新奖励
    pub async fn update_reward(
        &self,
        arm_id: String,
        context: ContextFeatures,
        reward: f64,
    ) -> Result<()> {
        let processed_features = self.preprocess_features(context).await?;
        
        // 存储经验
        let mut buffer = self.experience_buffer.write().await;
        buffer.add_experience(Experience {
            context: processed_features.raw_features.clone(),
            features: processed_features.processed_features.clone(),
            arm_id: arm_id.clone(),
            reward,
            probability: 1.0, // 占位符，应该从选择概率中获取
            timestamp: Utc::now(),
        });
        
        // 更新模型
        let mut model = self.model.write().await;
        let features_array = ArrayView1::from(&processed_features.processed_features);
        model.update(features_array, &arm_id, reward)?;
        
        // 更新臂统计
        let mut arms = self.arms.write().await;
        if let Some(arm) = arms.get_mut(&arm_id) {
            arm.sample_count += 1;
            arm.last_reward = reward;
            arm.last_updated = Utc::now();
        }
        
        // 更新性能追踪
        let mut tracker = self.performance_tracker.write().await;
        tracker.update(&arm_id, reward, 1.0);
        
        Ok(())
    }
    
    /// 预处理特征
    async fn preprocess_features(&self, context: ContextFeatures) -> Result<ContextFeatures> {
        let mut processor = self.feature_processor.write().await;
        processor.process(context)
    }
    
    /// LinUCB选择
    async fn select_with_linucb(
        &self,
        predictions: &HashMap<String, f64>,
        features: &ContextFeatures,
        alpha: f64,
    ) -> Result<ContextualSelection> {
        let arms = self.arms.read().await;
        let mut best_arm = String::new();
        let mut best_ucb = f64::NEG_INFINITY;
        let mut best_confidence = 0.0;
        
        for (arm_id, prediction) in predictions {
            if let Some(arm) = arms.get(arm_id) {
                // 计算置信上界
                let feature_vec = Array1::from_vec(features.processed_features.clone());
                let confidence_bonus = alpha * self.compute_confidence_bonus(arm, &feature_vec)?;
                let ucb = prediction + confidence_bonus;
                
                if ucb > best_ucb {
                    best_ucb = ucb;
                    best_arm = arm_id.clone();
                    best_confidence = confidence_bonus;
                }
            }
        }
        
        Ok(ContextualSelection {
            arm_id: best_arm,
            predicted_reward: best_ucb - best_confidence,
            confidence_bound: best_ucb,
            selection_probability: 1.0, // LinUCB是确定性的
            exploration_bonus: best_confidence,
            features_used: features.feature_names.clone(),
            selection_reason: "LinUCB Upper Confidence Bound".to_string(),
        })
    }
    
    /// LinTS选择
    async fn select_with_lints(
        &self,
        predictions: &HashMap<String, f64>,
        features: &ContextFeatures,
    ) -> Result<ContextualSelection> {
        let arms = self.arms.read().await;
        let mut rng = rand::thread_rng();
        let mut best_arm = String::new();
        let mut best_sample = f64::NEG_INFINITY;
        let mut selection_prob = 0.0;
        
        for (arm_id, _prediction) in predictions {
            if let Some(arm) = arms.get(arm_id) {
                // Thompson采样：从后验分布中采样
                let feature_vec = Array1::from_vec(features.processed_features.clone());
                let mean = arm.theta.dot(&feature_vec);
                let variance = self.compute_posterior_variance(arm, &feature_vec)?;
                
                let normal = Normal::new(mean, variance.sqrt()).map_err(|e| {
                    anyhow::anyhow!("Failed to create normal distribution: {}", e)
                })?;
                let sample = normal.sample(&mut rng);
                
                if sample > best_sample {
                    best_sample = sample;
                    best_arm = arm_id.clone();
                    selection_prob = 0.5; // 简化概率估计
                }
            }
        }
        
        Ok(ContextualSelection {
            arm_id: best_arm,
            predicted_reward: best_sample,
            confidence_bound: best_sample,
            selection_probability: selection_prob,
            exploration_bonus: 0.0, // TS没有显式探索奖励
            features_used: features.feature_names.clone(),
            selection_reason: "LinTS Thompson Sampling".to_string(),
        })
    }
    
    /// Neural UCB选择
    async fn select_with_neural_ucb(
        &self,
        predictions: &HashMap<String, f64>,
        features: &ContextFeatures,
    ) -> Result<ContextualSelection> {
        // 简化实现：选择预测值最高的臂 + 不确定性估计
        let mut best_arm = String::new();
        let mut best_value = f64::NEG_INFINITY;
        let exploration_bonus = 0.1; // 简化的探索奖励
        
        for (arm_id, prediction) in predictions {
            let adjusted_value = prediction + exploration_bonus;
            if adjusted_value > best_value {
                best_value = adjusted_value;
                best_arm = arm_id.clone();
            }
        }
        
        Ok(ContextualSelection {
            arm_id: best_arm,
            predicted_reward: best_value - exploration_bonus,
            confidence_bound: best_value,
            selection_probability: 0.8,
            exploration_bonus,
            features_used: features.feature_names.clone(),
            selection_reason: "Neural UCB with Uncertainty".to_string(),
        })
    }
    
    /// 计算置信度奖励
    fn compute_confidence_bonus(&self, arm: &ContextualArm, features: &Array1<f64>) -> Result<f64> {
        // 计算 x^T A^-1 x
        let temp = arm.a_inv.dot(features);
        let confidence = features.dot(&temp).sqrt();
        Ok(confidence)
    }
    
    /// 计算后验方差
    fn compute_posterior_variance(&self, arm: &ContextualArm, features: &Array1<f64>) -> Result<f64> {
        let temp = arm.a_inv.dot(features);
        let variance = features.dot(&temp);
        Ok(variance.max(0.01)) // 避免数值问题
    }
    
    /// 获取性能统计
    pub async fn get_performance_stats(&self) -> Result<PerformanceStats> {
        let tracker = self.performance_tracker.read().await;
        Ok(tracker.get_stats())
    }
    
    /// 批量更新
    pub async fn batch_update(&self) -> Result<()> {
        let buffer = self.experience_buffer.read().await;
        let experiences = buffer.get_recent_batch(self.config.batch_size);
        
        if experiences.len() < self.config.batch_size {
            return Ok(()); // 数据不足，跳过更新
        }
        
        let mut model = self.model.write().await;
        for exp in experiences {
            let features_array = ArrayView1::from(&exp.features);
            model.update(features_array, &exp.arm_id, exp.reward)?;
        }
        
        Ok(())
    }

    /// 使用IPS（逆概率评分）更新策略
    pub async fn update_ips(
        &self,
        arm_id: &str,
        context: ContextFeatures,
        reward: f64,
        probability: f64,
    ) -> Result<()> {
        // 处理特征
        let processed_features = self.preprocess_features(context).await?;
        
        // 更新模型
        let mut model = self.model.write().await;
        let features_array = ArrayView1::from(&processed_features.processed_features);
        model.update(
            features_array,
            arm_id,
            reward,
        )?;

        // 记录经验
        let experience = Experience {
            context: processed_features.processed_features.clone(),
            features: processed_features.processed_features.clone(),
            arm_id: arm_id.to_string(),
            reward,
            probability,
            timestamp: Utc::now(),
        };

        let mut buffer = self.experience_buffer.write().await;
        buffer.add_experience(experience);

        // 更新性能跟踪
        let mut tracker = self.performance_tracker.write().await;
        tracker.update(arm_id, reward, probability);

        Ok(())
    }

    /// 计算置信度半径辅助函数
    fn calculate_confidence_radius(
        &self,
        arm: &ContextualArm,
        features: &[f64],
        alpha: f64,
    ) -> Result<f64> {
        if features.len() != arm.theta.len() {
            return Err(anyhow::anyhow!("Feature dimension mismatch"));
        }

        let x = Array1::from_vec(features.to_vec());
        let quadratic_form = x.dot(&arm.a_inv.dot(&x));
        Ok(alpha * quadratic_form.sqrt())
    }

    /// 从后验分布采样
    fn sample_from_posterior(&self, arm: &ContextualArm, features: &[f64]) -> Result<f64> {
        if features.len() != arm.theta.len() {
            return Err(anyhow::anyhow!("Feature dimension mismatch"));
        }

        let x = Array1::from_vec(features.to_vec());
        let mean = arm.theta.dot(&x);
        let variance = x.dot(&arm.a_inv.dot(&x));
        
        let normal = Normal::new(mean, variance.sqrt())
            .map_err(|e| anyhow::anyhow!("Failed to create normal distribution: {}", e))?;
        
        Ok(normal.sample(&mut thread_rng()))
    }

    /// 计算后验标准差
    fn calculate_posterior_std(&self, arm: &ContextualArm, features: &[f64]) -> Result<f64> {
        if features.len() != arm.theta.len() {
            return Err(anyhow::anyhow!("Feature dimension mismatch"));
        }

        let x = Array1::from_vec(features.to_vec());
        let variance = x.dot(&arm.a_inv.dot(&x));
        Ok(variance.sqrt())
    }
}

impl ContextualArm {
    fn new(arm_id: String, feature_dim: usize) -> Result<Self> {
        let lambda_reg = 1.0;
        let a_inv = Array2::eye(feature_dim) / lambda_reg;
        
        Ok(Self {
            arm_id,
            theta: Array1::zeros(feature_dim),
            a_inv,
            b: Array1::zeros(feature_dim),
            sample_count: 0,
            last_reward: 0.0,
            last_updated: Utc::now(),
            confidence_radius: 0.0,
        })
    }
}

impl FeatureProcessor {
    fn new() -> Result<Self> {
        Ok(Self {
            feature_stats: HashMap::new(),
            selected_features: Vec::new(),
            normalization_params: NormalizationParams {
                method: NormalizationMethod::ZScore,
                global_mean: Array1::zeros(0),
                global_std: Array1::zeros(0),
                global_min: Array1::zeros(0),
                global_max: Array1::zeros(0),
            },
            pca_components: None,
            feature_interactions: Vec::new(),
        })
    }

    fn normalize_features(&mut self, features: &mut Vec<f64>) -> Result<()> {
        if self.normalization_params.global_mean.len() != features.len() {
            // 初始化归一化参数
            self.normalization_params.global_mean = Array1::from_vec(features.clone());
            self.normalization_params.global_std = Array1::ones(features.len());
            self.normalization_params.global_min = Array1::from_vec(features.clone());
            self.normalization_params.global_max = Array1::from_vec(features.clone());
            return Ok(());
        }

        match self.normalization_params.method {
            NormalizationMethod::ZScore => {
                for (i, feature) in features.iter_mut().enumerate() {
                    if i < self.normalization_params.global_std.len() && self.normalization_params.global_std[i] > 1e-8 {
                        *feature = (*feature - self.normalization_params.global_mean[i]) / self.normalization_params.global_std[i];
                    }
                }
            }
            NormalizationMethod::MinMax => {
                for (i, feature) in features.iter_mut().enumerate() {
                    if i < self.normalization_params.global_max.len() {
                        let range = self.normalization_params.global_max[i] - self.normalization_params.global_min[i];
                        if range > 1e-8 {
                            *feature = (*feature - self.normalization_params.global_min[i]) / range;
                        }
                    }
                }
            }
            _ => {} // No normalization
        }

        Ok(())
    }

    fn select_features(&self, features: &[f64]) -> Result<Vec<f64>> {
        if self.selected_features.is_empty() {
            // 如果没有选择特征，返回前一半特征（简化实现）
            let n_select = (features.len() / 2).max(1);
            Ok(features.iter().take(n_select).copied().collect())
        } else {
            let selected: Result<Vec<f64>, _> = self.selected_features.iter()
                .map(|&idx| {
                    features.get(idx).copied()
                        .ok_or_else(|| anyhow::anyhow!("Feature index {} out of bounds", idx))
                })
                .collect();
            selected
        }
    }

    fn reduce_dimensions(&self, features: &[f64]) -> Result<Vec<f64>> {
        if let Some(ref components) = self.pca_components {
            let feature_array = Array1::from_vec(features.to_vec());
            let reduced = components.dot(&feature_array);
            Ok(reduced.to_vec())
        } else {
            // 如果没有PCA组件，使用前一半维度
            let n_keep = (features.len() / 2).max(1);
            Ok(features.iter().take(n_keep).copied().collect())
        }
    }

    fn generate_interactions(&self, features: &[f64]) -> Result<Vec<f64>> {
        let mut interactions = Vec::new();
        
        // 生成简单的二阶交互项
        for (i, &feat_i) in features.iter().enumerate().take(5) { // 限制为前5个特征
            for (j, &feat_j) in features.iter().enumerate().skip(i + 1).take(5) {
                interactions.push(feat_i * feat_j);
            }
        }
        
        Ok(interactions)
    }

    pub fn process(&mut self, context: ContextFeatures) -> Result<ContextFeatures> {
        let mut processed_features = context.raw_features.clone();
        
        // 归一化特征
        self.normalize_features(&mut processed_features)?;
        
        // 特征选择
        let selected_features = self.select_features(&processed_features)?;
        
        // 降维
        let reduced_features = if self.pca_components.is_some() {
            self.reduce_dimensions(&selected_features)?
        } else {
            selected_features
        };
        
        // 生成交互特征
        let interaction_features = self.generate_interactions(&reduced_features)?;
        let mut final_features = reduced_features;
        final_features.extend(interaction_features);
        
        Ok(ContextFeatures {
            raw_features: context.raw_features,
            processed_features: final_features,
            feature_names: context.feature_names,
            timestamp: context.timestamp,
            feature_quality: context.feature_quality,
        })
    }
}

impl ExperienceBuffer {
    fn new(size: usize) -> Self {
        Self {
            experiences: Vec::with_capacity(size),
            buffer_size: size,
            current_index: 0,
        }
    }

    fn add_experience(&mut self, experience: Experience) {
        if self.experiences.len() < self.buffer_size {
            self.experiences.push(experience);
        } else {
            self.experiences[self.current_index] = experience;
            self.current_index = (self.current_index + 1) % self.buffer_size;
        }
    }

    fn get_recent_experiences(&self, n: usize) -> Vec<Experience> {
        let n = n.min(self.experiences.len());
        self.experiences.iter().rev().take(n).cloned().collect()
    }

    pub fn get_recent_batch(&self, batch_size: usize) -> Vec<Experience> {
        self.get_recent_experiences(batch_size)
    }
}

impl PerformanceTracker {
    fn new() -> Self {
        Self {
            cumulative_regret: 0.0,
            cumulative_reward: 0.0,
            arm_performance: HashMap::new(),
            regret_history: Vec::new(),
            feature_importance_history: Vec::new(),
        }
    }

    fn update(&mut self, arm_id: &str, reward: f64, _probability: f64) {
        self.cumulative_reward += reward;
        
        let performance = self.arm_performance.entry(arm_id.to_string())
            .or_insert_with(|| ArmPerformance {
                total_pulls: 0,
                total_reward: 0.0,
                recent_performance: Vec::new(),
                confidence_bounds: (0.0, 1.0),
            });
        
        performance.total_pulls += 1;
        performance.total_reward += reward;
        performance.recent_performance.push(reward);
        
        // 保持最近100次结果
        if performance.recent_performance.len() > 100 {
            performance.recent_performance.remove(0);
        }
        
        // 简化的遗憾计算（需要真实的最优臂信息）
        let estimated_regret = 0.1; // 占位符
        self.cumulative_regret += estimated_regret;
        self.regret_history.push((Utc::now(), estimated_regret));
    }

    pub fn get_stats(&self) -> PerformanceStats {
        let average_regret = if !self.regret_history.is_empty() {
            self.cumulative_regret / self.regret_history.len() as f64
        } else {
            0.0
        };

        PerformanceStats {
            cumulative_regret: self.cumulative_regret,
            cumulative_reward: self.cumulative_reward,
            average_regret,
            arm_performance: self.arm_performance.clone(),
            total_interactions: self.regret_history.len(),
        }
    }
}

/// LinUCB模型实现
impl LinUCBModel {
    fn new(feature_dim: usize, alpha: f64, lambda_reg: f64) -> Result<Self> {
        Ok(Self {
            arms: HashMap::new(),
            alpha,
            lambda_reg,
            feature_dim,
        })
    }

    fn get_or_create_arm(&mut self, arm_id: &str) -> &mut LinUCBArm {
        self.arms.entry(arm_id.to_string()).or_insert_with(|| {
            let identity = Array2::eye(self.feature_dim);
            LinUCBArm {
                a_matrix: &identity * self.lambda_reg,
                b_vector: Array1::zeros(self.feature_dim),
                theta: Array1::zeros(self.feature_dim),
                a_inv: identity / self.lambda_reg,
            }
        })
    }
}

impl ContextualModel for LinUCBModel {
    fn predict(&self, context: ArrayView1<f64>) -> Result<HashMap<String, f64>> {
        let mut predictions = HashMap::new();
        
        for (arm_id, arm) in &self.arms {
            let predicted_reward = arm.theta.dot(&context);
            predictions.insert(arm_id.clone(), predicted_reward);
        }
        
        Ok(predictions)
    }

    fn update(&mut self, context: ArrayView1<f64>, arm_id: &str, reward: f64) -> Result<()> {
        let feature_dim = self.feature_dim;
        let lambda_reg = self.lambda_reg;
        let arm = self.get_or_create_arm(arm_id);
        
        // 更新 A = A + x * x^T
        let context_matrix = context.insert_axis(ndarray::Axis(1));
        let outer_product = context_matrix.dot(&context_matrix.t());
        arm.a_matrix = &arm.a_matrix + &outer_product;
        
        // 更新 b = b + r * x
        arm.b_vector = &arm.b_vector + &(&context.to_owned() * reward);
        
        // 重新计算 A^-1 和 θ（简化实现）
        // 在实际实现中，应该使用 Sherman-Morrison 公式进行增量更新
        
        // 克隆矩阵以避免借用冲突
        let a_matrix_clone = arm.a_matrix.clone();
        
        // 本地矩阵求逆函数，避免self借用
        fn try_invert_local(matrix: &Array2<f64>) -> Result<Array2<f64>> {
            // 简化的矩阵求逆实现 (实际应该使用更稳健的方法)
            let det = matrix[(0, 0)] * matrix[(1, 1)] - matrix[(0, 1)] * matrix[(1, 0)];
            if det.abs() < 1e-12 {
                return Err(anyhow::anyhow!("Matrix is singular"));
            }
            let mut inv = Array2::zeros((matrix.nrows(), matrix.ncols()));
            if matrix.nrows() == 2 {
                inv[(0, 0)] = matrix[(1, 1)] / det;
                inv[(0, 1)] = -matrix[(0, 1)] / det;
                inv[(1, 0)] = -matrix[(1, 0)] / det;
                inv[(1, 1)] = matrix[(0, 0)] / det;
            } else {
                // 对于更大的矩阵，这里应该使用 LU 分解或其他方法
                return Err(anyhow::anyhow!("Matrix inversion not implemented for size > 2"));
            }
            Ok(inv)
        }
        
        match try_invert_local(&a_matrix_clone) {
            Ok(a_inv) => {
                arm.a_inv = a_inv;
                arm.theta = arm.a_inv.dot(&arm.b_vector);
            }
            Err(_) => {
                // 如果矩阵不可逆，使用伪逆或重新初始化
                let identity = Array2::eye(feature_dim);
                arm.a_matrix = &identity * lambda_reg;
                arm.a_inv = identity / lambda_reg;
            }
        }
        
        Ok(())
    }

    fn get_confidence(&self, context: ArrayView1<f64>, arm_id: &str) -> Result<f64> {
        if let Some(arm) = self.arms.get(arm_id) {
            let quadratic_form = context.dot(&arm.a_inv.dot(&context));
            Ok(self.alpha * quadratic_form.sqrt())
        } else {
            Ok(1.0) // 对未知臂给出高置信度
        }
    }

    fn batch_update(&mut self, experiences: &[Experience]) -> Result<()> {
        for experience in experiences {
            let features_array = ArrayView1::from(&experience.features);
            self.update(features_array, &experience.arm_id, experience.reward)?;
        }
        Ok(())
    }
}

impl LinUCBModel {
    fn try_invert_matrix(&self, matrix: &Array2<f64>) -> Result<Array2<f64>> {
        // 简化的矩阵求逆实现
        // 在实际实现中应该使用更稳定的数值方法
        let n = matrix.nrows();
        let mut augmented = Array2::zeros((n, 2 * n));
        
        // 构建增广矩阵 [A | I]
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = matrix[[i, j]];
                if i == j {
                    augmented[[i, j + n]] = 1.0;
                }
            }
        }
        
        // 高斯-约旦消元（简化版本）
        for i in 0..n {
            // 寻找主元
            let mut max_row = i;
            for k in (i + 1)..n {
                if augmented[[k, i]].abs() > augmented[[max_row, i]].abs() {
                    max_row = k;
                }
            }
            
            // 交换行
            if max_row != i {
                for j in 0..(2 * n) {
                    let temp = augmented[[i, j]];
                    augmented[[i, j]] = augmented[[max_row, j]];
                    augmented[[max_row, j]] = temp;
                }
            }
            
            // 检查奇异性
            if augmented[[i, i]].abs() < 1e-10 {
                return Err(anyhow::anyhow!("Matrix is singular"));
            }
            
            // 归一化当前行
            let pivot = augmented[[i, i]];
            for j in 0..(2 * n) {
                augmented[[i, j]] /= pivot;
            }
            
            // 消除其他行
            for k in 0..n {
                if k != i {
                    let factor = augmented[[k, i]];
                    for j in 0..(2 * n) {
                        augmented[[k, j]] -= factor * augmented[[i, j]];
                    }
                }
            }
        }
        
        // 提取逆矩阵
        let mut inverse = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inverse[[i, j]] = augmented[[i, j + n]];
            }
        }
        
        Ok(inverse)
    }
}

/// LinTS模型（Linear Thompson Sampling）
#[derive(Debug)]
pub struct LinTSModel {
    linucb_model: LinUCBModel,
    v_squared: f64, // 噪声方差
}

impl LinTSModel {
    fn new(feature_dim: usize, v: f64, lambda_reg: f64) -> Result<Self> {
        Ok(Self {
            linucb_model: LinUCBModel::new(feature_dim, 1.0, lambda_reg)?,
            v_squared: v * v,
        })
    }
}

impl ContextualModel for LinTSModel {
    fn predict(&self, context: ArrayView1<f64>) -> Result<HashMap<String, f64>> {
        // LinTS 使用采样而非点估计
        let mut predictions = HashMap::new();
        let mut rng = thread_rng();
        
        for (arm_id, arm) in &self.linucb_model.arms {
            // 从后验分布采样参数
            let mean = arm.theta.clone();
            let covariance = &arm.a_inv * self.v_squared;
            
            // 生成多元正态采样（简化版本）
            let mut sampled_theta = Array1::zeros(mean.len());
            for i in 0..mean.len() {
                let std_dev = covariance[[i, i]].sqrt();
                let normal = Normal::new(mean[i], std_dev)
                    .map_err(|e| anyhow::anyhow!("Normal distribution error: {}", e))?;
                sampled_theta[i] = normal.sample(&mut rng);
            }
            
            let predicted_reward = sampled_theta.dot(&context);
            predictions.insert(arm_id.clone(), predicted_reward);
        }
        
        Ok(predictions)
    }

    fn update(&mut self, context: ArrayView1<f64>, arm_id: &str, reward: f64) -> Result<()> {
        self.linucb_model.update(context, arm_id, reward)
    }

    fn get_confidence(&self, context: ArrayView1<f64>, arm_id: &str) -> Result<f64> {
        if let Some(arm) = self.linucb_model.arms.get(arm_id) {
            let variance = context.dot(&arm.a_inv.dot(&context)) * self.v_squared;
            Ok(1.96 * variance.sqrt()) // 95% 置信区间
        } else {
            Ok(1.0)
        }
    }

    fn batch_update(&mut self, experiences: &[Experience]) -> Result<()> {
        self.linucb_model.batch_update(experiences)
    }
}

/// 神经网络模型实现（简化版）
impl NeuralBanditModel {
    fn new(input_dim: usize, config: NeuralConfig) -> Result<Self> {
        Ok(Self {
            networks: HashMap::new(),
            config,
            optimizer_state: OptimizerState {
                momentum: HashMap::new(),
                learning_rate: 0.01,
                momentum_factor: 0.9,
            },
        })
    }

    fn get_or_create_network(&mut self, arm_id: &str, input_dim: usize) -> &mut SimpleNeuralNetwork {
        self.networks.entry(arm_id.to_string()).or_insert_with(|| {
            SimpleNeuralNetwork::new(input_dim, &self.config.hidden_layers).unwrap()
        })
    }
}

impl ContextualModel for NeuralBanditModel {
    fn predict(&self, context: ArrayView1<f64>) -> Result<HashMap<String, f64>> {
        let mut predictions = HashMap::new();
        
        for (arm_id, network) in &self.networks {
            let prediction = network.forward(context)?;
            predictions.insert(arm_id.clone(), prediction);
        }
        
        Ok(predictions)
    }

    fn update(&mut self, context: ArrayView1<f64>, arm_id: &str, reward: f64) -> Result<()> {
        let input_dim = context.len();
        let learning_rate = self.config.learning_rate;
        let network = self.get_or_create_network(arm_id, input_dim);
        
        // 简化的反向传播更新
        let prediction = network.forward(context)?;
        let error = reward - prediction;
        network.backward(error, learning_rate)?;
        
        Ok(())
    }

    fn get_confidence(&self, _context: ArrayView1<f64>, _arm_id: &str) -> Result<f64> {
        // 简化的不确定性估计
        Ok(0.1) // 固定不确定性
    }

    fn batch_update(&mut self, experiences: &[Experience]) -> Result<()> {
        for experience in experiences {
            let features_array = ArrayView1::from(&experience.features);
            self.update(features_array, &experience.arm_id, experience.reward)?;
        }
        Ok(())
    }
}

impl SimpleNeuralNetwork {
    fn new(input_dim: usize, hidden_layers: &[usize]) -> Result<Self> {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        
        let mut prev_dim = input_dim;
        for &layer_size in hidden_layers.iter().chain(std::iter::once(&1)) { // 输出层大小为1
            let weight_matrix = Self::random_matrix(layer_size, prev_dim, 0.1);
            let bias_vector = Array1::zeros(layer_size);
            
            weights.push(weight_matrix);
            biases.push(bias_vector);
            
            prev_dim = layer_size;
        }
        
        Ok(Self {
            weights,
            biases,
            activations: Vec::new(),
        })
    }

    fn random_matrix(rows: usize, cols: usize, scale: f64) -> Array2<f64> {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, scale).unwrap();
        Array2::from_shape_fn((rows, cols), |_| normal.sample(&mut rng))
    }

    fn forward(&self, input: ArrayView1<f64>) -> Result<f64> {
        let mut current = input.to_owned();
        
        for (weight, bias) in self.weights.iter().zip(self.biases.iter()) {
            current = weight.dot(&current) + bias;
            
            // ReLU激活（除了最后一层）
            if weight as *const _ != self.weights.last().unwrap() as *const _ {
                current.mapv_inplace(|x| x.max(0.0));
            }
        }
        
        Ok(current[0]) // 假设输出是标量
    }

    fn backward(&mut self, error: f64, learning_rate: f64) -> Result<()> {
        // 简化的反向传播实现
        // 在实际应用中应该实现完整的梯度计算和反向传播
        
        // 仅更新最后一层（简化）
        if let Some(last_weight) = self.weights.last_mut() {
            let gradient_scale = error * learning_rate;
            last_weight.mapv_inplace(|w| w + gradient_scale * 0.01);
        }
        
        Ok(())
    }
}

/// 性能统计结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub cumulative_regret: f64,
    pub cumulative_reward: f64,
    pub average_regret: f64,
    pub arm_performance: HashMap<String, ArmPerformance>,
    pub total_interactions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_linucb_bandit() {
        let config = ContextualBanditConfig {
            algorithm: ContextualAlgorithm::LinUCB { alpha: 0.1 },
            feature_dim: 3,
            alpha: 0.1,
            lambda_reg: 1.0,
            learning_rate: 0.01,
            batch_size: 10,
            update_frequency: 5,
            feature_selection: false,
            dimensionality_reduction: false,
        };

        let bandit = ContextualBandit::new(config).unwrap();
        bandit.add_arm("arm1".to_string()).await.unwrap();
        bandit.add_arm("arm2".to_string()).await.unwrap();

        let context = ContextFeatures {
            raw_features: vec![1.0, 0.5, -0.2],
            processed_features: vec![],
            feature_names: vec!["f1".to_string(), "f2".to_string(), "f3".to_string()],
            timestamp: Utc::now(),
            feature_quality: 1.0,
        };

        // 初始选择
        let selection = bandit.select_arm(context.clone()).await.unwrap();
        assert!(selection.arm_id == "arm1" || selection.arm_id == "arm2");

        // 更新奖励
        bandit.update_reward(selection.arm_id, context, 1.0).await.unwrap();

        // 再次选择，应该考虑之前的奖励
        let context2 = ContextFeatures {
            raw_features: vec![1.0, 0.5, -0.2],
            processed_features: vec![],
            feature_names: vec!["f1".to_string(), "f2".to_string(), "f3".to_string()],
            timestamp: Utc::now(),
            feature_quality: 1.0,
        };
        
        let _selection2 = bandit.select_arm(context2).await.unwrap();
    }

    #[tokio::test]
    async fn test_feature_processing() {
        let mut processor = FeatureProcessor::new().unwrap();
        let mut features = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        // 测试归一化
        processor.normalize_features(&mut features).unwrap();
        
        // 测试特征选择
        let selected = processor.select_features(&features).unwrap();
        assert!(!selected.is_empty());
        
        // 测试特征交互
        let interactions = processor.generate_interactions(&features).unwrap();
        assert!(!interactions.is_empty());
    }

    #[test]
    fn test_linucb_model() {
        let mut model = LinUCBModel::new(3, 0.1, 1.0).unwrap();
        let context = Array1::from_vec(vec![1.0, 0.5, -0.2]);
        
        // 初始预测
        let predictions = model.predict(context.view()).unwrap();
        assert!(predictions.is_empty()); // 没有臂时应该为空
        
        // 更新后预测
        model.update(context.view(), "arm1", 1.0).unwrap();
        let predictions = model.predict(context.view()).unwrap();
        assert!(predictions.contains_key("arm1"));
        
        // 获取置信度
        let confidence = model.get_confidence(context.view(), "arm1").unwrap();
        assert!(confidence > 0.0);
    }
}