//! 多臂老虎机算法实现
//! 支持UCB、Thompson Sampling等算法，用于参与率/时间窗/venue选择优化

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use anyhow::{Result, Context};
use rand::prelude::*;
use rand_distr::Beta;

/// 多臂老虎机配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditConfig {
    pub algorithm: BanditAlgorithm,
    pub exploration_parameter: f64,
    pub decay_factor: f64,
    pub min_samples: usize,
    pub max_arms: usize,
    pub reward_window: chrono::Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BanditAlgorithm {
    EpsilonGreedy { epsilon: f64 },
    UCB { confidence: f64 },
    ThompsonSampling,
    LinUCB { alpha: f64 },
}

/// 臂的统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmStats {
    pub arm_id: String,
    pub total_rewards: f64,
    pub sample_count: usize,
    pub alpha: f64,  // Beta分布参数（成功次数 + 1）
    pub beta: f64,   // Beta分布参数（失败次数 + 1）
    pub last_updated: DateTime<Utc>,
    pub confidence_interval: (f64, f64),
}

impl ArmStats {
    pub fn new(arm_id: String) -> Self {
        Self {
            arm_id,
            total_rewards: 0.0,
            sample_count: 0,
            alpha: 1.0,
            beta: 1.0,
            last_updated: Utc::now(),
            confidence_interval: (0.0, 1.0),
        }
    }

    pub fn mean_reward(&self) -> f64 {
        if self.sample_count == 0 {
            0.0
        } else {
            self.total_rewards / self.sample_count as f64
        }
    }

    pub fn update_reward(&mut self, reward: f64, is_success: bool) {
        self.total_rewards += reward;
        self.sample_count += 1;
        self.last_updated = Utc::now();

        // 更新Beta分布参数
        if is_success {
            self.alpha += 1.0;
        } else {
            self.beta += 1.0;
        }

        // 更新置信区间
        self.update_confidence_interval();
    }

    fn update_confidence_interval(&mut self) {
        // 使用Beta分布计算95%置信区间
        let beta_dist = Beta::new(self.alpha, self.beta).unwrap();
        let samples: Vec<f64> = (0..1000).map(|_| beta_dist.sample(&mut thread_rng())).collect();
        let mut sorted_samples = samples;
        sorted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        self.confidence_interval = (
            sorted_samples[25],  // 2.5%
            sorted_samples[975], // 97.5%
        );
    }
}

/// 多臂老虎机实例
pub struct MultiArmedBandit {
    config: BanditConfig,
    arms: Arc<RwLock<HashMap<String, ArmStats>>>,
    context_features: Arc<RwLock<HashMap<String, Vec<f64>>>>,
    reward_history: Arc<RwLock<Vec<RewardRecord>>>,
}

/// 奖励记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardRecord {
    pub arm_id: String,
    pub context: Vec<f64>,
    pub reward: f64,
    pub timestamp: DateTime<Utc>,
}

/// 选择结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionResult {
    pub arm_id: String,
    pub confidence: f64,
    pub expected_reward: f64,
    pub exploration_bonus: f64,
    pub selection_reason: String,
}

impl MultiArmedBandit {
    pub fn new(config: BanditConfig) -> Self {
        Self {
            config,
            arms: Arc::new(RwLock::new(HashMap::new())),
            context_features: Arc::new(RwLock::new(HashMap::new())),
            reward_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// 添加新臂
    pub async fn add_arm(&self, arm_id: String) -> Result<()> {
        let mut arms = self.arms.write().await;
        if !arms.contains_key(&arm_id) {
            arms.insert(arm_id.clone(), ArmStats::new(arm_id));
        }
        Ok(())
    }

    /// 选择最优臂
    pub async fn select_arm(&self, context: Option<Vec<f64>>) -> Result<SelectionResult> {
        let arms = self.arms.read().await;
        
        if arms.is_empty() {
            return Err(anyhow::anyhow!("No arms available"));
        }

        match self.config.algorithm {
            BanditAlgorithm::EpsilonGreedy { epsilon } => {
                self.epsilon_greedy_selection(&arms, epsilon).await
            }
            BanditAlgorithm::UCB { confidence } => {
                self.ucb_selection(&arms, confidence).await
            }
            BanditAlgorithm::ThompsonSampling => {
                self.thompson_sampling_selection(&arms).await
            }
            BanditAlgorithm::LinUCB { alpha } => {
                self.linucb_selection(&arms, context.unwrap_or_default(), alpha).await
            }
        }
    }

    /// Epsilon-Greedy算法
    async fn epsilon_greedy_selection(
        &self, 
        arms: &HashMap<String, ArmStats>, 
        epsilon: f64
    ) -> Result<SelectionResult> {
        let mut rng = thread_rng();
        
        if rng.gen::<f64>() < epsilon {
            // 探索：随机选择
            let arm_ids: Vec<_> = arms.keys().collect();
            let selected_arm = arm_ids[rng.gen_range(0..arm_ids.len())].clone();
            let arm_stats = &arms[&selected_arm];
            
            Ok(SelectionResult {
                arm_id: selected_arm,
                confidence: 0.5,
                expected_reward: arm_stats.mean_reward(),
                exploration_bonus: epsilon,
                selection_reason: "Exploration (random)".to_string(),
            })
        } else {
            // 利用：选择最佳臂
            let best_arm = arms.iter()
                .max_by(|(_, a), (_, b)| a.mean_reward().partial_cmp(&b.mean_reward()).unwrap())
                .map(|(id, stats)| (id.clone(), stats));

            if let Some((arm_id, stats)) = best_arm {
                Ok(SelectionResult {
                    arm_id,
                    confidence: 1.0 - epsilon,
                    expected_reward: stats.mean_reward(),
                    exploration_bonus: 0.0,
                    selection_reason: "Exploitation (best arm)".to_string(),
                })
            } else {
                Err(anyhow::anyhow!("No arms available for selection"))
            }
        }
    }

    /// UCB算法
    async fn ucb_selection(
        &self,
        arms: &HashMap<String, ArmStats>,
        confidence: f64,
    ) -> Result<SelectionResult> {
        let total_samples: usize = arms.values().map(|a| a.sample_count).sum();
        
        if total_samples == 0 {
            // 所有臂都没有样本，随机选择
            let arm_ids: Vec<_> = arms.keys().collect();
            let mut rng = thread_rng();
            let selected_arm = arm_ids[rng.gen_range(0..arm_ids.len())].clone();
            
            return Ok(SelectionResult {
                arm_id: selected_arm,
                confidence: 0.0,
                expected_reward: 0.0,
                exploration_bonus: f64::INFINITY,
                selection_reason: "Cold start (no samples)".to_string(),
            });
        }

        let mut best_arm = None;
        let mut best_ucb_value = f64::NEG_INFINITY;

        for (arm_id, stats) in arms.iter() {
            let ucb_value = if stats.sample_count == 0 {
                f64::INFINITY
            } else {
                let mean_reward = stats.mean_reward();
                let exploration_bonus = confidence * ((total_samples as f64).ln() / stats.sample_count as f64).sqrt();
                mean_reward + exploration_bonus
            };

            if ucb_value > best_ucb_value {
                best_ucb_value = ucb_value;
                best_arm = Some((arm_id.clone(), stats, ucb_value));
            }
        }

        if let Some((arm_id, stats, ucb_value)) = best_arm {
            let exploration_bonus = if stats.sample_count == 0 {
                f64::INFINITY
            } else {
                confidence * ((total_samples as f64).ln() / stats.sample_count as f64).sqrt()
            };

            Ok(SelectionResult {
                arm_id,
                confidence: if ucb_value == f64::INFINITY { 0.0 } else { 0.8 },
                expected_reward: stats.mean_reward(),
                exploration_bonus,
                selection_reason: format!("UCB value: {:.4}", ucb_value),
            })
        } else {
            Err(anyhow::anyhow!("No arms available for UCB selection"))
        }
    }

    /// Thompson Sampling算法
    async fn thompson_sampling_selection(
        &self,
        arms: &HashMap<String, ArmStats>,
    ) -> Result<SelectionResult> {
        let mut rng = thread_rng();
        let mut best_arm = None;
        let mut best_sample = f64::NEG_INFINITY;

        for (arm_id, stats) in arms.iter() {
            let beta_dist = Beta::new(stats.alpha, stats.beta)
                .map_err(|e| anyhow::anyhow!("Failed to create Beta distribution: {}", e))?;
            
            let sample = beta_dist.sample(&mut rng);
            
            if sample > best_sample {
                best_sample = sample;
                best_arm = Some((arm_id.clone(), stats, sample));
            }
        }

        if let Some((arm_id, stats, sampled_value)) = best_arm {
            Ok(SelectionResult {
                arm_id,
                confidence: sampled_value,
                expected_reward: stats.mean_reward(),
                exploration_bonus: (stats.confidence_interval.1 - stats.confidence_interval.0) / 2.0,
                selection_reason: format!("Thompson sample: {:.4}", sampled_value),
            })
        } else {
            Err(anyhow::anyhow!("No arms available for Thompson Sampling"))
        }
    }

    /// LinUCB算法（用于上下文老虎机）
    async fn linucb_selection(
        &self,
        arms: &HashMap<String, ArmStats>,
        context: Vec<f64>,
        alpha: f64,
    ) -> Result<SelectionResult> {
        if context.is_empty() {
            return Err(anyhow::anyhow!("Context required for LinUCB"));
        }

        // 简化的LinUCB实现
        let mut best_arm = None;
        let mut best_ucb_value = f64::NEG_INFINITY;

        for (arm_id, stats) in arms.iter() {
            // 简化版本：使用上下文特征的加权平均
            let context_weight: f64 = context.iter().sum::<f64>() / context.len() as f64;
            let adjusted_reward = stats.mean_reward() * (1.0 + context_weight);
            
            let confidence_bonus = if stats.sample_count == 0 {
                alpha
            } else {
                alpha / (stats.sample_count as f64).sqrt()
            };

            let ucb_value = adjusted_reward + confidence_bonus;

            if ucb_value > best_ucb_value {
                best_ucb_value = ucb_value;
                best_arm = Some((arm_id.clone(), stats, ucb_value, confidence_bonus));
            }
        }

        if let Some((arm_id, stats, ucb_value, confidence_bonus)) = best_arm {
            Ok(SelectionResult {
                arm_id,
                confidence: 0.9,
                expected_reward: stats.mean_reward(),
                exploration_bonus: confidence_bonus,
                selection_reason: format!("LinUCB value: {:.4}", ucb_value),
            })
        } else {
            Err(anyhow::anyhow!("No arms available for LinUCB selection"))
        }
    }

    /// 更新奖励
    pub async fn update_reward(
        &self,
        arm_id: &str,
        reward: f64,
        is_success: bool,
        context: Option<Vec<f64>>,
    ) -> Result<()> {
        let mut arms = self.arms.write().await;
        
        if let Some(arm_stats) = arms.get_mut(arm_id) {
            arm_stats.update_reward(reward, is_success);
        } else {
            return Err(anyhow::anyhow!("Arm {} not found", arm_id));
        }

        // 记录奖励历史
        let mut history = self.reward_history.write().await;
        history.push(RewardRecord {
            arm_id: arm_id.to_string(),
            context: context.unwrap_or_default(),
            reward,
            timestamp: Utc::now(),
        });

        // 保持历史记录在合理大小
        if history.len() > 10000 {
            history.drain(0..1000);
        }

        Ok(())
    }

    /// 获取所有臂的统计信息
    pub async fn get_arms_stats(&self) -> HashMap<String, ArmStats> {
        self.arms.read().await.clone()
    }

    /// 获取最佳臂推荐
    pub async fn get_best_arms(&self, top_k: usize) -> Vec<(String, f64, f64)> {
        let arms = self.arms.read().await;
        
        let mut arm_performances: Vec<_> = arms.iter()
            .map(|(id, stats)| {
                let mean_reward = stats.mean_reward();
                let confidence = if stats.sample_count > self.config.min_samples {
                    stats.confidence_interval.1 - stats.confidence_interval.0
                } else {
                    1.0
                };
                (id.clone(), mean_reward, confidence)
            })
            .collect();

        // 按平均奖励排序
        arm_performances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        arm_performances.into_iter().take(top_k).collect()
    }

    /// 获取探索进度
    pub async fn get_exploration_progress(&self) -> ExplorationProgress {
        let arms = self.arms.read().await;
        let total_samples: usize = arms.values().map(|a| a.sample_count).sum();
        let arms_with_min_samples = arms.values()
            .filter(|a| a.sample_count >= self.config.min_samples)
            .count();

        let exploration_rate = if arms.is_empty() {
            0.0
        } else {
            arms_with_min_samples as f64 / arms.len() as f64
        };

        let regret = self.calculate_cumulative_regret(&arms).await;

        ExplorationProgress {
            total_arms: arms.len(),
            arms_with_sufficient_samples: arms_with_min_samples,
            total_samples,
            exploration_rate,
            estimated_regret: regret,
        }
    }

    async fn calculate_cumulative_regret(&self, arms: &HashMap<String, ArmStats>) -> f64 {
        let history = self.reward_history.read().await;
        
        if history.is_empty() || arms.is_empty() {
            return 0.0;
        }

        // 找到最佳臂的真实平均奖励
        let best_arm_reward = arms.values()
            .map(|a| a.mean_reward())
            .fold(f64::NEG_INFINITY, f64::max);

        // 计算累积遗憾
        history.iter()
            .map(|record| {
                let arm_reward = arms.get(&record.arm_id)
                    .map(|a| a.mean_reward())
                    .unwrap_or(0.0);
                best_arm_reward - arm_reward
            })
            .sum()
    }
}

/// 探索进度
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationProgress {
    pub total_arms: usize,
    pub arms_with_sufficient_samples: usize,
    pub total_samples: usize,
    pub exploration_rate: f64,
    pub estimated_regret: f64,
}

/// 参与率优化老虎机
pub struct ParticipationRateBandit {
    bandit: MultiArmedBandit,
    rate_options: Vec<f64>,
}

impl ParticipationRateBandit {
    pub fn new(config: BanditConfig, rate_options: Vec<f64>) -> Self {
        let bandit = MultiArmedBandit::new(config);
        
        Self {
            bandit,
            rate_options,
        }
    }

    pub async fn initialize(&self) -> Result<()> {
        for (i, rate) in self.rate_options.iter().enumerate() {
            let arm_id = format!("rate_{:.3}", rate);
            self.bandit.add_arm(arm_id).await?;
        }
        Ok(())
    }

    /// 选择最优参与率
    pub async fn select_participation_rate(&self, market_conditions: MarketConditions) -> Result<f64> {
        let context = vec![
            market_conditions.volatility,
            market_conditions.spread_bps,
            market_conditions.volume_ratio,
            market_conditions.time_of_day,
        ];

        let selection = self.bandit.select_arm(Some(context)).await?;
        
        // 从arm_id解析参与率
        if let Some(rate_str) = selection.arm_id.strip_prefix("rate_") {
            rate_str.parse().map_err(|e| anyhow::anyhow!("Failed to parse rate: {}", e))
        } else {
            Err(anyhow::anyhow!("Invalid arm_id format: {}", selection.arm_id))
        }
    }

    /// 更新参与率表现
    pub async fn update_performance(
        &self,
        rate: f64,
        execution_result: ExecutionResult,
        market_conditions: MarketConditions,
    ) -> Result<()> {
        let arm_id = format!("rate_{:.3}", rate);
        let context = vec![
            market_conditions.volatility,
            market_conditions.spread_bps, 
            market_conditions.volume_ratio,
            market_conditions.time_of_day,
        ];

        // 计算复合奖励：考虑成本、达成率、市场冲击
        let reward = self.calculate_execution_reward(&execution_result);
        let is_success = execution_result.fill_rate > 0.9 && execution_result.slippage_bps < 5.0;

        self.bandit.update_reward(&arm_id, reward, is_success, Some(context)).await
    }

    fn calculate_execution_reward(&self, result: &ExecutionResult) -> f64 {
        let fill_rate_score = result.fill_rate.min(1.0);
        let slippage_penalty = (-result.slippage_bps / 10.0).exp();
        let timing_bonus = if result.execution_time_ms < 30000.0 { 1.1 } else { 1.0 };
        
        fill_rate_score * slippage_penalty * timing_bonus
    }
}

/// 市场条件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub volatility: f64,
    pub spread_bps: f64,
    pub volume_ratio: f64,
    pub time_of_day: f64, // 0-1表示一天中的时间
}

/// 执行结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    pub fill_rate: f64,
    pub slippage_bps: f64,
    pub execution_time_ms: f64,
    pub market_impact_bps: f64,
}

/// Venue选择老虎机
pub struct VenueSelectionBandit {
    bandit: MultiArmedBandit,
    venues: Vec<String>,
}

impl VenueSelectionBandit {
    pub fn new(config: BanditConfig, venues: Vec<String>) -> Self {
        let bandit = MultiArmedBandit::new(config);
        
        Self {
            bandit,
            venues,
        }
    }

    pub async fn initialize(&self) -> Result<()> {
        for venue in &self.venues {
            self.bandit.add_arm(venue.clone()).await?;
        }
        Ok(())
    }

    /// 选择最优交易所
    pub async fn select_venue(&self, order_characteristics: OrderCharacteristics) -> Result<String> {
        let context = vec![
            order_characteristics.size_ratio,
            order_characteristics.urgency,
            order_characteristics.spread_sensitivity,
        ];

        let selection = self.bandit.select_arm(Some(context)).await?;
        Ok(selection.arm_id)
    }

    /// 更新venue表现
    pub async fn update_venue_performance(
        &self,
        venue: &str,
        execution_quality: VenueExecutionQuality,
        order_characteristics: OrderCharacteristics,
    ) -> Result<()> {
        let context = vec![
            order_characteristics.size_ratio,
            order_characteristics.urgency,
            order_characteristics.spread_sensitivity,
        ];

        let reward = self.calculate_venue_reward(&execution_quality);
        let is_success = execution_quality.rejection_rate < 0.05 && execution_quality.latency_ms < 100.0;

        self.bandit.update_reward(venue, reward, is_success, Some(context)).await
    }

    fn calculate_venue_reward(&self, quality: &VenueExecutionQuality) -> f64 {
        let fill_quality = (1.0 - quality.rejection_rate).max(0.0);
        let latency_quality = (200.0 / (quality.latency_ms + 100.0)).min(1.0);
        let cost_quality = (-quality.effective_spread_bps / 5.0).exp();
        
        (fill_quality * latency_quality * cost_quality).powf(0.33) // 几何平均
    }
}

/// 订单特征
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderCharacteristics {
    pub size_ratio: f64,      // 相对于平均订单大小
    pub urgency: f64,         // 0-1，紧急程度
    pub spread_sensitivity: f64, // 对价差的敏感度
}

/// Venue执行质量
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VenueExecutionQuality {
    pub rejection_rate: f64,
    pub latency_ms: f64,
    pub effective_spread_bps: f64,
    pub market_depth_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_epsilon_greedy_bandit() {
        let config = BanditConfig {
            algorithm: BanditAlgorithm::EpsilonGreedy { epsilon: 0.1 },
            exploration_parameter: 0.1,
            decay_factor: 0.99,
            min_samples: 10,
            max_arms: 5,
            reward_window: chrono::Duration::hours(24),
        };

        let bandit = MultiArmedBandit::new(config);
        
        // 添加臂
        bandit.add_arm("arm1".to_string()).await.unwrap();
        bandit.add_arm("arm2".to_string()).await.unwrap();
        bandit.add_arm("arm3".to_string()).await.unwrap();

        // 模拟选择和奖励更新
        for _ in 0..100 {
            let selection = bandit.select_arm(None).await.unwrap();
            let reward = if selection.arm_id == "arm2" { 0.8 } else { 0.3 };
            bandit.update_reward(&selection.arm_id, reward, reward > 0.5, None).await.unwrap();
        }

        let stats = bandit.get_arms_stats().await;
        
        // arm2应该有最高的平均奖励
        assert!(stats["arm2"].mean_reward() > stats["arm1"].mean_reward());
        assert!(stats["arm2"].mean_reward() > stats["arm3"].mean_reward());
    }

    #[tokio::test]
    async fn test_thompson_sampling_bandit() {
        let config = BanditConfig {
            algorithm: BanditAlgorithm::ThompsonSampling,
            exploration_parameter: 1.0,
            decay_factor: 0.99,
            min_samples: 5,
            max_arms: 5,
            reward_window: chrono::Duration::hours(24),
        };

        let bandit = MultiArmedBandit::new(config);
        
        bandit.add_arm("good_arm".to_string()).await.unwrap();
        bandit.add_arm("bad_arm".to_string()).await.unwrap();

        // 给good_arm更多成功奖励
        for _ in 0..50 {
            bandit.update_reward("good_arm", 1.0, true, None).await.unwrap();
            bandit.update_reward("bad_arm", 0.0, false, None).await.unwrap();
        }

        let mut good_arm_selections = 0;
        for _ in 0..100 {
            let selection = bandit.select_arm(None).await.unwrap();
            if selection.arm_id == "good_arm" {
                good_arm_selections += 1;
            }
        }

        // good_arm应该被选择更多次
        assert!(good_arm_selections > 70);
    }

    #[tokio::test] 
    async fn test_participation_rate_bandit() {
        let config = BanditConfig {
            algorithm: BanditAlgorithm::UCB { confidence: 2.0 },
            exploration_parameter: 2.0,
            decay_factor: 0.99,
            min_samples: 3,
            max_arms: 10,
            reward_window: chrono::Duration::hours(1),
        };

        let rate_options = vec![0.05, 0.10, 0.15, 0.20, 0.25];
        let bandit = ParticipationRateBandit::new(config, rate_options);
        
        bandit.initialize().await.unwrap();

        let market_conditions = MarketConditions {
            volatility: 0.02,
            spread_bps: 5.0,
            volume_ratio: 1.2,
            time_of_day: 0.5,
        };

        let selected_rate = bandit.select_participation_rate(market_conditions.clone()).await.unwrap();
        assert!(selected_rate >= 0.05 && selected_rate <= 0.25);

        // 模拟执行结果更新
        let execution_result = ExecutionResult {
            fill_rate: 0.95,
            slippage_bps: 2.0,
            execution_time_ms: 25000.0,
            market_impact_bps: 3.0,
        };

        bandit.update_performance(selected_rate, execution_result, market_conditions).await.unwrap();
    }
}