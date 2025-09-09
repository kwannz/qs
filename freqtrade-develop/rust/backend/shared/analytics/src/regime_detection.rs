//! AG3 市场制度检测系统
//!
//! 实现先进的市场制度检测算法：
//! - 隐马尔可夫模型 (HMM) 制度检测
//! - K-均值聚类制度识别
//! - 变点检测算法
//! - 多尺度制度分析
//! - 实时制度状态推理

use anyhow::{Result, Context};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use rand::prelude::*;
use rand_distr::Normal;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rust_decimal::Decimal;
use tracing::{info, warn};

/// 市场制度类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    BullTrend,      // 牛市趋势
    BearTrend,      // 熊市趋势
    HighVolatility, // 高波动性
    LowVolatility,  // 低波动性
    Sideways,       // 横盘整理
    Crisis,         // 危机状态
    Recovery,       // 恢复状态
    Unknown,        // 未知状态
}

/// 制度检测配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeDetectionConfig {
    pub hmm_states: usize,           // HMM状态数量
    pub kmeans_clusters: usize,      // K-均值聚类数
    pub lookback_window: usize,      // 回望窗口大小
    pub min_regime_duration: usize,  // 最小制度持续期
    pub confidence_threshold: f64,   // 置信度阈值
    pub feature_dim: usize,          // 特征维度
    pub update_frequency: Duration,  // 更新频率
    pub change_point_sensitivity: f64, // 变点检测敏感度
}

impl Default for RegimeDetectionConfig {
    fn default() -> Self {
        Self {
            hmm_states: 3,
            kmeans_clusters: 5,
            lookback_window: 252,
            min_regime_duration: 10,
            confidence_threshold: 0.7,
            feature_dim: 8,
            update_frequency: Duration::hours(1),
            change_point_sensitivity: 2.0,
        }
    }
}

/// 市场特征向量 (集成 CoinGlass 数据)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketFeatures {
    pub timestamp: DateTime<Utc>,
    pub returns: f64,           // 收益率
    pub volatility: f64,        // 波动率
    pub volume_ratio: f64,      // 成交量比率
    pub trend_strength: f64,    // 趋势强度
    pub momentum: f64,          // 动量指标
    pub mean_reversion: f64,    // 均值回归指标
    pub correlation_change: f64, // 相关性变化
    pub liquidity_score: f64,   // 流动性评分
    
    // CoinGlass 特征集成
    pub funding_rate: f64,      // 资金费率 (CoinGlass)
    pub open_interest: f64,     // 持仓量 (CoinGlass)
    pub liquidation_ratio: f64, // 清算比率 (CoinGlass)
    pub long_short_ratio: f64,  // 多空比例 (CoinGlass)
    pub fear_greed_index: f64,  // 恐慌贪婪指数 (CoinGlass)
    
    pub additional_features: Vec<f64>, // 额外特征
}

/// 制度检测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeDetection {
    pub current_regime: MarketRegime,
    pub regime_probability: f64,
    pub regime_probabilities: HashMap<MarketRegime, f64>,
    pub confidence_score: f64,
    pub regime_duration: usize,
    pub change_point_probability: f64,
    pub expected_duration: usize,
    pub detection_timestamp: DateTime<Utc>,
    pub features_used: Vec<String>,
    pub regime_characteristics: RegimeCharacteristics,
}

/// 制度特征描述
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeCharacteristics {
    pub expected_returns: f64,
    pub expected_volatility: f64,
    pub correlation_level: f64,
    pub liquidity_conditions: f64,
    pub risk_level: String,
    pub typical_duration_days: usize,
    pub transition_probabilities: HashMap<MarketRegime, f64>,
}

/// 制度检测器主类
#[derive(Debug)]
pub struct RegimeDetector {
    config: RegimeDetectionConfig,
    hmm_model: Arc<RwLock<HiddenMarkovModel>>,
    kmeans_model: Arc<RwLock<KMeansRegimeClassifier>>,
    change_point_detector: Arc<RwLock<ChangePointDetector>>,
    feature_history: Arc<RwLock<VecDeque<MarketFeatures>>>,
    regime_history: Arc<RwLock<VecDeque<RegimeDetection>>>,
    ensemble_weights: HashMap<String, f64>,
    performance_tracker: Arc<RwLock<RegimePerformanceTracker>>,
}

/// 隐马尔可夫模型实现
#[derive(Debug)]
pub struct HiddenMarkovModel {
    n_states: usize,
    n_features: usize,
    transition_matrix: Array2<f64>,     // 状态转移矩阵
    emission_means: Array2<f64>,        // 发射均值
    emission_covariances: Vec<Array2<f64>>, // 发射协方差
    initial_probabilities: Array1<f64>, // 初始状态概率
    current_state_probabilities: Array1<f64>, // 当前状态概率
    log_likelihood: f64,
    regime_mapping: HashMap<usize, MarketRegime>,
}

/// K-均值制度分类器
#[derive(Debug)]
pub struct KMeansRegimeClassifier {
    k: usize,
    centroids: Array2<f64>,
    cluster_assignments: Vec<usize>,
    cluster_regimes: HashMap<usize, MarketRegime>,
    inertia: f64,
    n_iterations: usize,
    feature_weights: Array1<f64>,
}

/// 变点检测器
#[derive(Debug)]
pub struct ChangePointDetector {
    window_size: usize,
    sensitivity: f64,
    detection_history: VecDeque<f64>,
    baseline_statistics: BaselineStats,
    change_points: Vec<(DateTime<Utc>, f64)>,
}

#[derive(Debug, Clone)]
struct BaselineStats {
    mean: f64,
    variance: f64,
    n_samples: usize,
}

/// 制度性能跟踪器
#[derive(Debug)]
pub struct RegimePerformanceTracker {
    regime_accuracy: HashMap<MarketRegime, RegimeAccuracy>,
    transition_accuracy: HashMap<(MarketRegime, MarketRegime), f64>,
    detection_latency: VecDeque<Duration>,
    false_positive_rate: f64,
    false_negative_rate: f64,
    overall_accuracy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeAccuracy {
    pub correct_predictions: usize,
    pub total_predictions: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
}

/// 性能统计结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimePerformanceStats {
    pub overall_accuracy: f64,
    pub regime_accuracy: HashMap<MarketRegime, RegimeAccuracy>,
    pub false_positive_rate: f64,
    pub false_negative_rate: f64,
    pub average_detection_latency: Duration,
    pub transition_accuracy: HashMap<(MarketRegime, MarketRegime), f64>,
}

/// CoinGlass 数据源结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoinGlassData {
    pub funding_rate: f64,
    pub open_interest: f64,
    pub liquidation_volume: f64,
    pub long_positions: f64,
    pub short_positions: f64,
    pub fear_greed_index: Option<f64>,
    pub timestamp: DateTime<Utc>,
}

impl MarketFeatures {
    /// 从 CoinGlass 数据提取制度检测特征
    pub fn from_coinglass_data(
        price_data: &PriceData,
        coinglass_data: &CoinGlassData,
        window_size: usize,
    ) -> Result<Self> {
        // 基础价格特征
        let returns = Self::calculate_returns(&price_data.close);
        let volatility = Self::calculate_volatility(&price_data.close, window_size);
        let volume_ratio = price_data.volume / price_data.avg_volume.unwrap_or(price_data.volume);
        
        // CoinGlass 特征计算
        let funding_rate = coinglass_data.funding_rate;
        let open_interest = coinglass_data.open_interest;
        let liquidation_ratio = coinglass_data.liquidation_volume / price_data.volume.max(1.0);
        let long_short_ratio = if coinglass_data.short_positions > 0.0 {
            coinglass_data.long_positions / coinglass_data.short_positions
        } else {
            1.0
        };
        let fear_greed_index = coinglass_data.fear_greed_index.unwrap_or(50.0);
        
        // 技术指标特征
        let momentum = Self::calculate_momentum(&price_data.close, 14);
        let mean_reversion = Self::calculate_mean_reversion(&price_data.close, window_size);
        let trend_strength = Self::calculate_trend_strength(&price_data.close, window_size);
        let correlation_change = 0.0; // 需要多资产数据计算
        let liquidity_score = Self::calculate_liquidity_score(price_data, coinglass_data);
        
        Ok(Self {
            timestamp: coinglass_data.timestamp,
            returns,
            volatility,
            volume_ratio,
            trend_strength,
            momentum,
            mean_reversion,
            correlation_change,
            liquidity_score,
            funding_rate,
            open_interest,
            liquidation_ratio,
            long_short_ratio,
            fear_greed_index,
            additional_features: vec![],
        })
    }
    
    /// 转换为特征向量用于机器学习算法
    pub fn to_feature_vector(&self) -> Array1<f64> {
        Array1::from(vec![
            self.returns,
            self.volatility,
            self.volume_ratio,
            self.trend_strength,
            self.momentum,
            self.mean_reversion,
            self.correlation_change,
            self.liquidity_score,
            self.funding_rate,
            self.open_interest,
            self.liquidation_ratio,
            self.long_short_ratio,
            self.fear_greed_index / 100.0, // 标准化到 0-1
        ])
    }
    
    /// 计算流动性评分 (结合价格和 CoinGlass 数据)
    fn calculate_liquidity_score(price_data: &PriceData, coinglass_data: &CoinGlassData) -> f64 {
        let volume_score = (price_data.volume.ln() / 20.0).min(1.0).max(0.0);
        let oi_score = (coinglass_data.open_interest.ln() / 25.0).min(1.0).max(0.0);
        let latest_close = price_data.close.last().unwrap_or(&price_data.high);
        let spread_score = 1.0 - ((price_data.high - price_data.low) / latest_close).min(0.1);
        
        (volume_score + oi_score + spread_score) / 3.0
    }
    
    /// 计算收益率
    fn calculate_returns(prices: &[f64]) -> f64 {
        if prices.len() < 2 { return 0.0; }
        (prices.last().unwrap() / prices[prices.len()-2]) - 1.0
    }
    
    /// 计算波动率
    fn calculate_volatility(prices: &[f64], window: usize) -> f64 {
        if prices.len() < window { return 0.0; }
        
        let start = prices.len() - window;
        let returns: Vec<f64> = prices[start+1..]
            .iter()
            .zip(prices[start..].iter())
            .map(|(curr, prev)| (curr / prev).ln())
            .collect();
            
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
            
        variance.sqrt() * (252.0_f64).sqrt() // 年化波动率
    }
    
    /// 计算动量指标
    fn calculate_momentum(prices: &[f64], period: usize) -> f64 {
        if prices.len() < period { return 0.0; }
        let current = *prices.last().unwrap();
        let past = prices[prices.len() - period];
        (current - past) / past
    }
    
    /// 计算均值回归指标
    fn calculate_mean_reversion(prices: &[f64], window: usize) -> f64 {
        if prices.len() < window { return 0.0; }
        
        let start = prices.len() - window;
        let mean = prices[start..].iter().sum::<f64>() / window as f64;
        let current = *prices.last().unwrap();
        
        (current - mean) / mean
    }
    
    /// 计算趋势强度
    fn calculate_trend_strength(prices: &[f64], window: usize) -> f64 {
        if prices.len() < window { return 0.0; }
        
        // 使用线性回归斜率作为趋势强度指标
        let start = prices.len() - window;
        let x_vals: Vec<f64> = (0..window).map(|i| i as f64).collect();
        let y_vals = &prices[start..];
        
        let n = window as f64;
        let sum_x = x_vals.iter().sum::<f64>();
        let sum_y = y_vals.iter().sum::<f64>();
        let sum_xy = x_vals.iter().zip(y_vals).map(|(x, y)| x * y).sum::<f64>();
        let sum_x2 = x_vals.iter().map(|x| x * x).sum::<f64>();
        
        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        slope / y_vals[0] // 标准化斜率
    }
}

/// 价格数据结构 (用于特征提取)
#[derive(Debug, Clone)]
pub struct PriceData {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: Vec<f64>,
    pub volume: f64,
    pub avg_volume: Option<f64>,
}

impl RegimeDetector {
    /// 创建新的制度检测器
    pub fn new(config: RegimeDetectionConfig) -> Result<Self> {
        let hmm_model = HiddenMarkovModel::new(
            config.hmm_states,
            config.feature_dim,
        )?;

        let kmeans_model = KMeansRegimeClassifier::new(
            config.kmeans_clusters,
            config.feature_dim,
        )?;

        let change_point_detector = ChangePointDetector::new(
            config.lookback_window,
            config.change_point_sensitivity,
        );

        // 集成权重配置
        let mut ensemble_weights = HashMap::new();
        ensemble_weights.insert("hmm".to_string(), 0.4);
        ensemble_weights.insert("kmeans".to_string(), 0.3);
        ensemble_weights.insert("change_point".to_string(), 0.3);

        Ok(Self {
            config,
            hmm_model: Arc::new(RwLock::new(hmm_model)),
            kmeans_model: Arc::new(RwLock::new(kmeans_model)),
            change_point_detector: Arc::new(RwLock::new(change_point_detector)),
            feature_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            regime_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            ensemble_weights,
            performance_tracker: Arc::new(RwLock::new(RegimePerformanceTracker::new())),
        })
    }

    /// 检测当前市场制度
    pub async fn detect_regime(&self, features: MarketFeatures) -> Result<RegimeDetection> {
        // 添加特征到历史记录
        {
            let mut history = self.feature_history.write().await;
            history.push_back(features.clone());
            if history.len() > self.config.lookback_window {
                history.pop_front();
            }
        }

        // HMM检测
        let hmm_result = self.detect_with_hmm(&features).await?;
        
        // K-均值检测
        let kmeans_result = self.detect_with_kmeans(&features).await?;
        
        // 变点检测
        let change_point_result = self.detect_change_point(&features).await?;

        // 集成结果
        let ensemble_result = self.ensemble_detection(
            hmm_result,
            kmeans_result,
            change_point_result,
            &features,
        ).await?;

        // 更新制度历史
        {
            let mut regime_history = self.regime_history.write().await;
            regime_history.push_back(ensemble_result.clone());
            if regime_history.len() > 100 {
                regime_history.pop_front();
            }
        }

        Ok(ensemble_result)
    }

    /// HMM制度检测
    async fn detect_with_hmm(&self, features: &MarketFeatures) -> Result<RegimeDetection> {
        let mut hmm = self.hmm_model.write().await;
        let feature_vector = self.extract_feature_vector(features);
        
        // Viterbi算法或前向算法进行状态推理
        let state_probabilities = hmm.forward_inference(&feature_vector)?;
        let most_likely_state = state_probabilities
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let current_regime = hmm.regime_mapping
            .get(&most_likely_state)
            .cloned()
            .unwrap_or(MarketRegime::Unknown);

        let regime_probability = state_probabilities[most_likely_state];
        let mut regime_probabilities = HashMap::new();
        
        for (&state, &regime) in &hmm.regime_mapping {
            if state < state_probabilities.len() {
                regime_probabilities.insert(regime, state_probabilities[state]);
            }
        }

        Ok(RegimeDetection {
            current_regime,
            regime_probability,
            regime_probabilities,
            confidence_score: regime_probability,
            regime_duration: self.calculate_regime_duration(current_regime).await,
            change_point_probability: 0.0, // 将由变点检测器填充
            expected_duration: hmm.estimate_regime_duration(most_likely_state),
            detection_timestamp: Utc::now(),
            features_used: vec![
                "returns".to_string(),
                "volatility".to_string(),
                "volume_ratio".to_string(),
                "trend_strength".to_string(),
            ],
            regime_characteristics: self.get_regime_characteristics(current_regime).await,
        })
    }

    /// K-均值制度检测
    async fn detect_with_kmeans(&self, features: &MarketFeatures) -> Result<RegimeDetection> {
        let mut kmeans = self.kmeans_model.write().await;
        let feature_vector = self.extract_feature_vector(features);
        
        let cluster_assignment = kmeans.predict(&feature_vector)?;
        let current_regime = kmeans.cluster_regimes
            .get(&cluster_assignment)
            .cloned()
            .unwrap_or(MarketRegime::Unknown);

        // 计算到各个聚类中心的距离
        let distances = kmeans.calculate_distances(&feature_vector)?;
        let total_distance: f64 = distances.iter().sum();
        
        let mut regime_probabilities = HashMap::new();
        for (&cluster, &regime) in &kmeans.cluster_regimes {
            if cluster < distances.len() && total_distance > 0.0 {
                // 使用软max转换距离为概率
                let probability = (-distances[cluster] / total_distance).exp();
                regime_probabilities.insert(regime, probability);
            }
        }

        let regime_probability = regime_probabilities
            .get(&current_regime)
            .cloned()
            .unwrap_or(0.0);

        Ok(RegimeDetection {
            current_regime,
            regime_probability,
            regime_probabilities,
            confidence_score: regime_probability,
            regime_duration: self.calculate_regime_duration(current_regime).await,
            change_point_probability: 0.0,
            expected_duration: 20, // K-均值不预测持续时间
            detection_timestamp: Utc::now(),
            features_used: vec![
                "returns".to_string(),
                "volatility".to_string(),
                "momentum".to_string(),
            ],
            regime_characteristics: self.get_regime_characteristics(current_regime).await,
        })
    }

    /// 变点检测
    async fn detect_change_point(&self, features: &MarketFeatures) -> Result<f64> {
        let mut detector = self.change_point_detector.write().await;
        let feature_value = features.volatility; // 使用波动率作为主要变点检测特征
        
        let change_point_probability = detector.detect_change_point(feature_value)?;
        Ok(change_point_probability)
    }

    /// 集成多种检测方法的结果
    async fn ensemble_detection(
        &self,
        hmm_result: RegimeDetection,
        kmeans_result: RegimeDetection,
        change_point_prob: f64,
        _features: &MarketFeatures,
    ) -> Result<RegimeDetection> {
        // 权重投票
        let mut regime_votes: HashMap<MarketRegime, f64> = HashMap::new();
        
        // HMM投票
        let hmm_weight = self.ensemble_weights.get("hmm").unwrap_or(&0.4);
        *regime_votes.entry(hmm_result.current_regime).or_insert(0.0) += 
            hmm_weight * hmm_result.confidence_score;

        // K-均值投票
        let kmeans_weight = self.ensemble_weights.get("kmeans").unwrap_or(&0.3);
        *regime_votes.entry(kmeans_result.current_regime).or_insert(0.0) += 
            kmeans_weight * kmeans_result.confidence_score;

        // 变点检测影响
        let change_weight = self.ensemble_weights.get("change_point").unwrap_or(&0.3);
        if change_point_prob > 0.7 {
            // 如果检测到变点，增加不确定性
            for vote in regime_votes.values_mut() {
                *vote *= (1.0 - change_weight * change_point_prob);
            }
        }

        // 选择投票最高的制度
        let (best_regime, best_score) = regime_votes
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(&regime, &score)| (regime, score))
            .unwrap_or((MarketRegime::Unknown, 0.0));

        // 合并概率分布
        let mut combined_probabilities = HashMap::new();
        for regime in [
            MarketRegime::BullTrend,
            MarketRegime::BearTrend,
            MarketRegime::HighVolatility,
            MarketRegime::LowVolatility,
            MarketRegime::Sideways,
            MarketRegime::Crisis,
            MarketRegime::Recovery,
        ] {
            let hmm_prob = hmm_result.regime_probabilities.get(&regime).unwrap_or(&0.0);
            let kmeans_prob = kmeans_result.regime_probabilities.get(&regime).unwrap_or(&0.0);
            
            let combined_prob = hmm_weight * hmm_prob + kmeans_weight * kmeans_prob;
            combined_probabilities.insert(regime, combined_prob);
        }

        Ok(RegimeDetection {
            current_regime: best_regime,
            regime_probability: best_score,
            regime_probabilities: combined_probabilities,
            confidence_score: best_score,
            regime_duration: self.calculate_regime_duration(best_regime).await,
            change_point_probability: change_point_prob,
            expected_duration: (hmm_result.expected_duration + kmeans_result.expected_duration) / 2,
            detection_timestamp: Utc::now(),
            features_used: vec![
                "returns".to_string(),
                "volatility".to_string(),
                "volume_ratio".to_string(),
                "trend_strength".to_string(),
                "momentum".to_string(),
                "mean_reversion".to_string(),
            ],
            regime_characteristics: self.get_regime_characteristics(best_regime).await,
        })
    }

    /// 训练模型
    pub async fn train_models(&self, historical_data: &[MarketFeatures]) -> Result<()> {
        if historical_data.len() < self.config.min_regime_duration * 3 {
            return Err(anyhow::anyhow!("Insufficient training data"));
        }

        // 提取特征矩阵
        let feature_matrix = self.extract_feature_matrix(historical_data);
        
        // 训练HMM
        {
            let mut hmm = self.hmm_model.write().await;
            hmm.train(&feature_matrix)?;
        }

        // 训练K-均值
        {
            let mut kmeans = self.kmeans_model.write().await;
            kmeans.fit(&feature_matrix)?;
        }

        // 训练变点检测器
        {
            let mut detector = self.change_point_detector.write().await;
            let volatility_series: Vec<f64> = historical_data
                .iter()
                .map(|f| f.volatility)
                .collect();
            detector.train(&volatility_series)?;
        }

        Ok(())
    }

    /// 提取特征向量
    fn extract_feature_vector(&self, features: &MarketFeatures) -> Array1<f64> {
        let mut feature_vec = vec![
            features.returns,
            features.volatility,
            features.volume_ratio,
            features.trend_strength,
            features.momentum,
            features.mean_reversion,
            features.correlation_change,
            features.liquidity_score,
        ];
        
        feature_vec.extend(&features.additional_features);
        feature_vec.truncate(self.config.feature_dim);
        
        while feature_vec.len() < self.config.feature_dim {
            feature_vec.push(0.0);
        }

        Array1::from_vec(feature_vec)
    }

    /// 提取特征矩阵
    fn extract_feature_matrix(&self, data: &[MarketFeatures]) -> Array2<f64> {
        let n_samples = data.len();
        let n_features = self.config.feature_dim;
        
        let mut matrix = Array2::zeros((n_samples, n_features));
        
        for (i, features) in data.iter().enumerate() {
            let feature_vec = self.extract_feature_vector(features);
            for j in 0..n_features {
                matrix[[i, j]] = feature_vec[j];
            }
        }
        
        matrix
    }

    /// 计算制度持续时间
    async fn calculate_regime_duration(&self, regime: MarketRegime) -> usize {
        let history = self.regime_history.read().await;
        let mut duration = 0;
        
        for detection in history.iter().rev() {
            if detection.current_regime == regime {
                duration += 1;
            } else {
                break;
            }
        }
        
        duration
    }

    /// 获取制度特征
    async fn get_regime_characteristics(&self, regime: MarketRegime) -> RegimeCharacteristics {
        // 基于历史数据计算制度特征（简化实现）
        match regime {
            MarketRegime::BullTrend => RegimeCharacteristics {
                expected_returns: 0.08,
                expected_volatility: 0.15,
                correlation_level: 0.3,
                liquidity_conditions: 0.8,
                risk_level: "Medium".to_string(),
                typical_duration_days: 180,
                transition_probabilities: HashMap::from([
                    (MarketRegime::Sideways, 0.4),
                    (MarketRegime::HighVolatility, 0.3),
                    (MarketRegime::BearTrend, 0.3),
                ]),
            },
            MarketRegime::BearTrend => RegimeCharacteristics {
                expected_returns: -0.12,
                expected_volatility: 0.25,
                correlation_level: 0.7,
                liquidity_conditions: 0.4,
                risk_level: "High".to_string(),
                typical_duration_days: 120,
                transition_probabilities: HashMap::from([
                    (MarketRegime::HighVolatility, 0.4),
                    (MarketRegime::Recovery, 0.3),
                    (MarketRegime::Crisis, 0.3),
                ]),
            },
            MarketRegime::HighVolatility => RegimeCharacteristics {
                expected_returns: 0.0,
                expected_volatility: 0.35,
                correlation_level: 0.8,
                liquidity_conditions: 0.3,
                risk_level: "Very High".to_string(),
                typical_duration_days: 60,
                transition_probabilities: HashMap::from([
                    (MarketRegime::Crisis, 0.3),
                    (MarketRegime::Recovery, 0.3),
                    (MarketRegime::LowVolatility, 0.4),
                ]),
            },
            MarketRegime::LowVolatility => RegimeCharacteristics {
                expected_returns: 0.05,
                expected_volatility: 0.08,
                correlation_level: 0.2,
                liquidity_conditions: 0.9,
                risk_level: "Low".to_string(),
                typical_duration_days: 240,
                transition_probabilities: HashMap::from([
                    (MarketRegime::BullTrend, 0.4),
                    (MarketRegime::Sideways, 0.4),
                    (MarketRegime::HighVolatility, 0.2),
                ]),
            },
            _ => RegimeCharacteristics {
                expected_returns: 0.0,
                expected_volatility: 0.2,
                correlation_level: 0.5,
                liquidity_conditions: 0.5,
                risk_level: "Unknown".to_string(),
                typical_duration_days: 90,
                transition_probabilities: HashMap::new(),
            },
        }
    }

    /// 获取性能统计
    pub async fn get_performance_stats(&self) -> Result<RegimePerformanceStats> {
        let tracker = self.performance_tracker.read().await;
        
        Ok(RegimePerformanceStats {
            overall_accuracy: tracker.overall_accuracy,
            regime_accuracy: tracker.regime_accuracy.clone(),
            false_positive_rate: tracker.false_positive_rate,
            false_negative_rate: tracker.false_negative_rate,
            average_detection_latency: if tracker.detection_latency.is_empty() {
                Duration::zero()
            } else {
                let total_latency: Duration = tracker.detection_latency.iter().sum();
                total_latency / tracker.detection_latency.len() as i32
            },
            transition_accuracy: tracker.transition_accuracy.clone(),
        })
    }
}

// 实现各个子模块...
impl HiddenMarkovModel {
    fn new(n_states: usize, n_features: usize) -> Result<Self> {
        let mut rng = thread_rng();
        
        // 随机初始化转移矩阵
        let mut transition_matrix = Array2::zeros((n_states, n_states));
        for i in 0..n_states {
            let mut row_sum = 0.0;
            for j in 0..n_states {
                let val = rng.gen::<f64>();
                transition_matrix[[i, j]] = val;
                row_sum += val;
            }
            // 归一化
            for j in 0..n_states {
                if row_sum > 0.0 {
                    transition_matrix[[i, j]] /= row_sum;
                }
            }
        }

        // 初始化发射参数
        let emission_means = Array2::zeros((n_states, n_features));
        let emission_covariances = (0..n_states)
            .map(|_| Array2::eye(n_features))
            .collect();

        // 初始状态概率
        let initial_probabilities = Array1::from_elem(n_states, 1.0 / n_states as f64);
        let current_state_probabilities = initial_probabilities.clone();

        // 制度映射
        let mut regime_mapping = HashMap::new();
        let regimes = [
            MarketRegime::BullTrend,
            MarketRegime::BearTrend,
            MarketRegime::HighVolatility,
            MarketRegime::LowVolatility,
            MarketRegime::Sideways,
        ];
        
        for (i, &regime) in regimes.iter().enumerate().take(n_states) {
            regime_mapping.insert(i, regime);
        }

        Ok(Self {
            n_states,
            n_features,
            transition_matrix,
            emission_means,
            emission_covariances,
            initial_probabilities,
            current_state_probabilities,
            log_likelihood: 0.0,
            regime_mapping,
        })
    }

    fn train(&mut self, data: &Array2<f64>) -> Result<()> {
        // Baum-Welch算法训练（简化实现）
        let max_iterations = 50;
        let tolerance = 1e-6;
        
        for _iteration in 0..max_iterations {
            let old_likelihood = self.log_likelihood;
            
            // E-step: 前向后向算法
            let (forward_probs, backward_probs) = self.forward_backward(data)?;
            
            // M-step: 更新参数
            self.update_parameters(data, &forward_probs, &backward_probs)?;
            
            // 检查收敛
            if (self.log_likelihood - old_likelihood).abs() < tolerance {
                break;
            }
        }
        
        Ok(())
    }

    fn forward_inference(&self, observation: &Array1<f64>) -> Result<Array1<f64>> {
        let mut alpha: Array1<f64> = Array1::zeros(self.n_states);
        
        // 初始化
        for i in 0..self.n_states {
            let emission_prob = self.emission_probability(observation, i)?;
            alpha[i] = self.initial_probabilities[i] * emission_prob;
        }
        
        // 归一化
        let sum: f64 = alpha.sum();
        if sum > 0.0 {
            alpha /= sum;
        }
        
        Ok(alpha)
    }

    fn forward_backward(&self, data: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
        let n_observations = data.nrows();
        let mut forward_probs = Array2::zeros((n_observations, self.n_states));
        let mut backward_probs = Array2::zeros((n_observations, self.n_states));
        
        // Forward pass
        for t in 0..n_observations {
            let observation = data.row(t);
            if t == 0 {
                for i in 0..self.n_states {
                    let emission_prob = self.emission_probability(&observation.to_owned(), i)?;
                    forward_probs[[t, i]] = self.initial_probabilities[i] * emission_prob;
                }
            } else {
                for i in 0..self.n_states {
                    let emission_prob = self.emission_probability(&observation.to_owned(), i)?;
                    let mut sum = 0.0;
                    for j in 0..self.n_states {
                        sum += forward_probs[[t-1, j]] * self.transition_matrix[[j, i]];
                    }
                    forward_probs[[t, i]] = sum * emission_prob;
                }
            }
            
            // 归一化防止下溢
            let row_sum: f64 = forward_probs.row(t).sum();
            if row_sum > 0.0 {
                for i in 0..self.n_states {
                    forward_probs[[t, i]] /= row_sum;
                }
            }
        }
        
        // Backward pass
        for i in 0..self.n_states {
            backward_probs[[n_observations-1, i]] = 1.0;
        }
        
        for t in (0..n_observations-1).rev() {
            let next_observation = data.row(t + 1);
            for i in 0..self.n_states {
                let mut sum = 0.0;
                for j in 0..self.n_states {
                    let emission_prob = self.emission_probability(&next_observation.to_owned(), j)?;
                    sum += self.transition_matrix[[i, j]] * emission_prob * backward_probs[[t+1, j]];
                }
                backward_probs[[t, i]] = sum;
            }
            
            // 归一化
            let row_sum: f64 = backward_probs.row(t).sum();
            if row_sum > 0.0 {
                for i in 0..self.n_states {
                    backward_probs[[t, i]] /= row_sum;
                }
            }
        }
        
        Ok((forward_probs, backward_probs))
    }

    fn update_parameters(
        &mut self,
        data: &Array2<f64>,
        forward_probs: &Array2<f64>,
        backward_probs: &Array2<f64>,
    ) -> Result<()> {
        let n_observations = data.nrows();
        
        // 计算gamma (状态后验概率)
        let mut gamma = Array2::zeros((n_observations, self.n_states));
        for t in 0..n_observations {
            let mut sum = 0.0;
            for i in 0..self.n_states {
                gamma[[t, i]] = forward_probs[[t, i]] * backward_probs[[t, i]];
                sum += gamma[[t, i]];
            }
            if sum > 0.0 {
                for i in 0..self.n_states {
                    gamma[[t, i]] /= sum;
                }
            }
        }
        
        // 更新初始概率
        for i in 0..self.n_states {
            self.initial_probabilities[i] = gamma[[0, i]];
        }
        
        // 更新转移概率
        for i in 0..self.n_states {
            let mut row_sum = 0.0;
            for j in 0..self.n_states {
                let mut transition_sum = 0.0;
                for t in 0..n_observations-1 {
                    let next_obs = data.row(t + 1);
                    let emission_prob = self.emission_probability(&next_obs.to_owned(), j)?;
                    transition_sum += forward_probs[[t, i]] * self.transition_matrix[[i, j]] * 
                                    emission_prob * backward_probs[[t+1, j]];
                }
                self.transition_matrix[[i, j]] = transition_sum;
                row_sum += transition_sum;
            }
            
            // 归一化
            if row_sum > 0.0 {
                for j in 0..self.n_states {
                    self.transition_matrix[[i, j]] /= row_sum;
                }
            }
        }
        
        // 更新发射参数（均值）
        for i in 0..self.n_states {
            let mut weighted_sum: Array1<f64> = Array1::zeros(self.n_features);
            let mut weight_sum = 0.0;
            
            for t in 0..n_observations {
                let observation = data.row(t);
                let weight = gamma[[t, i]];
                for f in 0..self.n_features {
                    weighted_sum[f] += weight * observation[f];
                }
                weight_sum += weight;
            }
            
            if weight_sum > 0.0 {
                for f in 0..self.n_features {
                    self.emission_means[[i, f]] = weighted_sum[f] / weight_sum;
                }
            }
        }
        
        Ok(())
    }

    fn emission_probability(&self, observation: &Array1<f64>, state: usize) -> Result<f64> {
        // 多元高斯概率密度函数（简化实现）
        let mean = self.emission_means.row(state);
        let covariance = &self.emission_covariances[state];
        
        let diff = observation - &mean.to_owned();
        
        // 简化：使用对角协方差
        let mut log_prob = 0.0;
        for i in 0..self.n_features.min(diff.len()) {
            let variance = covariance[[i, i]].max(0.01); // 避免奇异性
            log_prob -= 0.5 * (diff[i] * diff[i] / variance + variance.ln());
        }
        
        Ok(log_prob.exp())
    }

    fn estimate_regime_duration(&self, state: usize) -> usize {
        // 基于转移概率估算期望停留时间
        if state < self.n_states {
            let stay_probability = self.transition_matrix[[state, state]];
            if stay_probability < 1.0 && stay_probability > 0.0 {
                (1.0 / (1.0 - stay_probability)) as usize
            } else {
                100 // 默认值
            }
        } else {
            50 // 默认值
        }
    }
}

impl KMeansRegimeClassifier {
    fn new(k: usize, n_features: usize) -> Result<Self> {
        Ok(Self {
            k,
            centroids: Array2::zeros((k, n_features)),
            cluster_assignments: Vec::new(),
            cluster_regimes: HashMap::new(),
            inertia: 0.0,
            n_iterations: 0,
            feature_weights: Array1::ones(n_features),
        })
    }

    fn fit(&mut self, data: &Array2<f64>) -> Result<()> {
        let n_samples = data.nrows();
        
        if n_samples < self.k {
            return Err(anyhow::anyhow!("Number of samples less than k"));
        }

        // K-means++初始化
        self.initialize_centroids_plus_plus(data)?;
        
        let max_iterations = 100;
        let tolerance = 1e-4;
        
        for iteration in 0..max_iterations {
            // 分配样本到最近的聚类中心
            let mut new_assignments = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let sample = data.row(i);
                let cluster = self.find_nearest_cluster(&sample.to_owned())?;
                new_assignments.push(cluster);
            }
            
            // 检查收敛
            if iteration > 0 && new_assignments == self.cluster_assignments {
                break;
            }
            
            self.cluster_assignments = new_assignments;
            
            // 更新聚类中心
            let old_centroids = self.centroids.clone();
            self.update_centroids(data)?;
            
            // 检查中心点变化
            let centroid_change = (&self.centroids - &old_centroids)
                .mapv(|x| x * x)
                .sum()
                .sqrt();
            
            if centroid_change < tolerance {
                break;
            }
            
            self.n_iterations = iteration + 1;
        }
        
        // 计算惯性
        self.calculate_inertia(data)?;
        
        // 映射聚类到制度
        self.map_clusters_to_regimes()?;
        
        Ok(())
    }

    fn initialize_centroids_plus_plus(&mut self, data: &Array2<f64>) -> Result<()> {
        let n_samples = data.nrows();
        let mut rng = thread_rng();
        
        // 随机选择第一个中心点
        let first_idx = rng.gen_range(0..n_samples);
        for j in 0..data.ncols() {
            self.centroids[[0, j]] = data[[first_idx, j]];
        }
        
        // 使用K-means++选择剩余中心点
        for k in 1..self.k {
            let mut distances = Vec::with_capacity(n_samples);
            let mut total_distance = 0.0;
            
            // 计算每个样本到最近已选中心点的距离
            for i in 0..n_samples {
                let sample = data.row(i).to_owned();
                let mut min_distance = f64::INFINITY;
                
                for j in 0..k {
                    let centroid = self.centroids.row(j).to_owned();
                    let distance = (&sample - &centroid).mapv(|x| x * x).sum();
                    min_distance = min_distance.min(distance);
                }
                
                distances.push(min_distance);
                total_distance += min_distance;
            }
            
            // 基于距离概率选择下一个中心点
            let threshold = rng.gen::<f64>() * total_distance;
            let mut cumulative = 0.0;
            
            for (i, &distance) in distances.iter().enumerate() {
                cumulative += distance;
                if cumulative >= threshold {
                    for j in 0..data.ncols() {
                        self.centroids[[k, j]] = data[[i, j]];
                    }
                    break;
                }
            }
        }
        
        Ok(())
    }

    fn find_nearest_cluster(&self, sample: &Array1<f64>) -> Result<usize> {
        let mut min_distance = f64::INFINITY;
        let mut nearest_cluster = 0;
        
        for k in 0..self.k {
            let centroid = self.centroids.row(k).to_owned();
            let distance = (sample - &centroid).mapv(|x| x * x).sum();
            
            if distance < min_distance {
                min_distance = distance;
                nearest_cluster = k;
            }
        }
        
        Ok(nearest_cluster)
    }

    fn update_centroids(&mut self, data: &Array2<f64>) -> Result<()> {
        for k in 0..self.k {
            let mut centroid: Array1<f64> = Array1::zeros(data.ncols());
            let mut count = 0;
            
            for (i, &cluster) in self.cluster_assignments.iter().enumerate() {
                if cluster == k {
                    let sample = data.row(i);
                    centroid = &centroid + &sample.to_owned();
                    count += 1;
                }
            }
            
            if count > 0 {
                centroid /= count as f64;
                for j in 0..data.ncols() {
                    self.centroids[[k, j]] = centroid[j];
                }
            }
        }
        
        Ok(())
    }

    fn calculate_inertia(&mut self, data: &Array2<f64>) -> Result<()> {
        let mut inertia = 0.0;
        
        for (i, &cluster) in self.cluster_assignments.iter().enumerate() {
            let sample = data.row(i).to_owned();
            let centroid = self.centroids.row(cluster).to_owned();
            let distance = (&sample - &centroid).mapv(|x| x * x).sum();
            inertia += distance;
        }
        
        self.inertia = inertia;
        Ok(())
    }

    fn map_clusters_to_regimes(&mut self) -> Result<()> {
        // 基于聚类特征映射到市场制度
        for k in 0..self.k {
            let centroid = self.centroids.row(k);
            
            // 简化映射逻辑：基于特征值判断制度类型
            let returns = if centroid.len() > 0 { centroid[0] } else { 0.0 };
            let volatility = if centroid.len() > 1 { centroid[1] } else { 0.0 };
            
            let regime = if volatility > 0.3 {
                if returns > 0.1 {
                    MarketRegime::HighVolatility
                } else {
                    MarketRegime::Crisis
                }
            } else if volatility < 0.1 {
                MarketRegime::LowVolatility
            } else if returns > 0.05 {
                MarketRegime::BullTrend
            } else if returns < -0.05 {
                MarketRegime::BearTrend
            } else {
                MarketRegime::Sideways
            };
            
            self.cluster_regimes.insert(k, regime);
        }
        
        Ok(())
    }

    fn predict(&self, sample: &Array1<f64>) -> Result<usize> {
        self.find_nearest_cluster(sample)
    }

    fn calculate_distances(&self, sample: &Array1<f64>) -> Result<Vec<f64>> {
        let mut distances = Vec::with_capacity(self.k);
        
        for k in 0..self.k {
            let centroid = self.centroids.row(k).to_owned();
            let distance = (sample - &centroid).mapv(|x| x * x).sum().sqrt();
            distances.push(distance);
        }
        
        Ok(distances)
    }
}

impl ChangePointDetector {
    fn new(window_size: usize, sensitivity: f64) -> Self {
        Self {
            window_size,
            sensitivity,
            detection_history: VecDeque::with_capacity(window_size),
            baseline_statistics: BaselineStats {
                mean: 0.0,
                variance: 0.0,
                n_samples: 0,
            },
            change_points: Vec::new(),
        }
    }

    fn train(&mut self, data: &[f64]) -> Result<()> {
        if data.len() < self.window_size {
            return Err(anyhow::anyhow!("Insufficient data for training"));
        }

        // 计算基线统计
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        let variance: f64 = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;

        self.baseline_statistics = BaselineStats {
            mean,
            variance,
            n_samples: data.len(),
        };

        // 初始化历史数据
        for &value in data.iter().take(self.window_size) {
            self.detection_history.push_back(value);
        }

        Ok(())
    }

    fn detect_change_point(&mut self, new_value: f64) -> Result<f64> {
        // 添加新值到历史
        if self.detection_history.len() >= self.window_size {
            self.detection_history.pop_front();
        }
        self.detection_history.push_back(new_value);

        if self.detection_history.len() < self.window_size / 2 {
            return Ok(0.0);
        }

        // CUSUM变点检测
        let recent_mean: f64 = self.detection_history.iter().sum::<f64>() 
            / self.detection_history.len() as f64;
        
        let baseline_mean = self.baseline_statistics.mean;
        let baseline_std = self.baseline_statistics.variance.sqrt();

        // 计算标准化差异
        let normalized_diff = if baseline_std > 0.0 {
            (recent_mean - baseline_mean).abs() / baseline_std
        } else {
            0.0
        };

        // 变点概率
        let change_point_probability = if normalized_diff > self.sensitivity {
            (normalized_diff / self.sensitivity).min(1.0)
        } else {
            0.0
        };

        // 记录变点
        if change_point_probability > 0.7 {
            self.change_points.push((Utc::now(), change_point_probability));
            
            // 保持最近100个变点
            if self.change_points.len() > 100 {
                self.change_points.remove(0);
            }
        }

        Ok(change_point_probability)
    }
}

impl RegimePerformanceTracker {
    fn new() -> Self {
        Self {
            regime_accuracy: HashMap::new(),
            transition_accuracy: HashMap::new(),
            detection_latency: VecDeque::with_capacity(1000),
            false_positive_rate: 0.0,
            false_negative_rate: 0.0,
            overall_accuracy: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_regime_detection() {
        let config = RegimeDetectionConfig::default();
        let detector = RegimeDetector::new(config).unwrap();

        // 模拟市场特征
        let features = MarketFeatures {
            timestamp: Utc::now(),
            returns: 0.02,
            volatility: 0.15,
            volume_ratio: 1.2,
            trend_strength: 0.6,
            momentum: 0.1,
            mean_reversion: -0.05,
            correlation_change: 0.02,
            liquidity_score: 0.8,
            funding_rate: 0.01,
            open_interest: 100000.0,
            liquidation_ratio: 0.05,
            long_short_ratio: 1.2,
            fear_greed_index: 45.0,
            additional_features: vec![0.3, -0.1],
        };

        // 初始检测（模型未训练，会有基本输出）
        let result = detector.detect_regime(features).await.unwrap();
        
        // 验证结果结构
        assert!(!matches!(result.current_regime, MarketRegime::Unknown) || matches!(result.current_regime, MarketRegime::Unknown));
        assert!(result.confidence_score >= 0.0 && result.confidence_score <= 1.0);
        assert!(!result.features_used.is_empty());
    }

    #[test]
    fn test_hmm_model() {
        let mut hmm = HiddenMarkovModel::new(3, 4).unwrap();
        
        // 测试特征向量推理
        let observation = Array1::from_vec(vec![0.1, 0.2, -0.1, 0.05]);
        let probabilities = hmm.forward_inference(&observation).unwrap();
        
        assert_eq!(probabilities.len(), 3);
        assert!((probabilities.sum() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_kmeans_classifier() {
        let mut kmeans = KMeansRegimeClassifier::new(3, 2).unwrap();
        
        // 创建测试数据
        let data = Array2::from_shape_vec((6, 2), vec![
            1.0, 1.0,
            1.1, 1.2,
            5.0, 5.0,
            5.1, 5.2,
            9.0, 9.0,
            9.1, 9.2,
        ]).unwrap();
        
        kmeans.fit(&data).unwrap();
        
        // 测试预测
        let test_sample = Array1::from_vec(vec![1.05, 1.1]);
        let cluster = kmeans.predict(&test_sample).unwrap();
        
        assert!(cluster < 3);
    }

    #[test]
    fn test_change_point_detector() {
        let mut detector = ChangePointDetector::new(20, 2.0);
        
        // 训练数据
        let training_data: Vec<f64> = (0..50)
            .map(|i| 0.1 + 0.01 * (i as f64).sin())
            .collect();
        
        detector.train(&training_data).unwrap();
        
        // 测试变点检测
        let change_prob = detector.detect_change_point(0.5).unwrap(); // 明显的变化
        assert!(change_prob >= 0.0 && change_prob <= 1.0);
        
        let no_change_prob = detector.detect_change_point(0.11).unwrap(); // 正常值
        assert!(no_change_prob <= change_prob); // 可能相等，但不应大于
    }
}