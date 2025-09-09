//! AG3三重障碍标签系统
//! 
//! 实现了基于三重障碍方法的样本标签生成，包括：
//! - 动态止盈/止损障碍计算
//! - 时间障碍管理
//! - 元标签生成
//! - 分数差分标签
//! - 体制自适应标签

use anyhow::Result;
use rust_decimal_macros::dec;
use chrono::{DateTime, Duration, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::convert::TryInto;

/// 三重障碍标签生成器
#[derive(Debug, Clone)]
pub struct TripleBarrierLabeler {
    config: TripleBarrierConfig,
    price_cache: Arc<RwLock<VecDeque<PricePoint>>>,
    volume_cache: Arc<RwLock<VecDeque<VolumePoint>>>,
    active_barriers: Arc<RwLock<HashMap<String, ActiveBarrier>>>,
    label_history: Arc<RwLock<VecDeque<TripleBarrierLabel>>>,
    volatility_estimator: Arc<VolatilityEstimator>,
    regime_detector: Arc<RegimeDetector>,
    meta_labeler: Arc<MetaLabeler>,
}

/// 三重障碍配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleBarrierConfig {
    /// 默认时间障碍（小时）
    pub default_time_horizon_hours: u32,
    /// 默认止盈倍数
    pub default_profit_multiplier: f64,
    /// 默认止损倍数
    pub default_loss_multiplier: f64,
    /// 波动率窗口大小
    pub volatility_window: usize,
    /// 最小持有期（分钟）
    pub min_holding_period_minutes: u32,
    /// 最大活跃障碍数
    pub max_active_barriers: usize,
    /// 是否启用动态调整
    pub enable_dynamic_adjustment: bool,
    /// 是否启用元标签
    pub enable_meta_labeling: bool,
    /// 体制感知模式
    pub regime_aware: bool,
    /// 分数差分参数
    pub fractional_diff_threshold: Option<f64>,
    /// 自适应因子
    pub adaptation_factor: f64,
}

/// 价格点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricePoint {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub price: Decimal,
    pub bid: Option<Decimal>,
    pub ask: Option<Decimal>,
    pub volume: Option<Decimal>,
}

/// 成交量点
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumePoint {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub volume: Decimal,
    pub dollar_volume: Option<Decimal>,
}

/// 活跃障碍
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveBarrier {
    pub id: String,
    pub symbol: String,
    pub entry_time: DateTime<Utc>,
    pub entry_price: Decimal,
    pub upper_barrier: Decimal,
    pub lower_barrier: Decimal,
    pub time_barrier: DateTime<Utc>,
    pub current_price: Decimal,
    pub max_price: Decimal,
    pub min_price: Decimal,
    pub volatility_at_entry: f64,
    pub regime_at_entry: Option<String>,
    pub signal_strength: Option<f64>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// 三重障碍标签
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleBarrierLabel {
    pub id: String,
    pub symbol: String,
    pub entry_time: DateTime<Utc>,
    pub entry_price: Decimal,
    pub upper_barrier: Decimal,
    pub lower_barrier: Decimal,
    pub time_barrier: DateTime<Utc>,
    pub exit_type: BarrierExitType,
    pub exit_price: Decimal,
    pub exit_time: DateTime<Utc>,
    pub return_pct: Decimal,
    pub holding_period_hours: f64,
    pub max_favorable_excursion: Decimal, // MFE
    pub max_adverse_excursion: Decimal,   // MAE
    pub volatility_regime: Option<String>,
    pub meta_label: Option<MetaLabel>,
    pub fractional_diff_label: Option<FractionalDiffLabel>,
    pub confidence: f64,
    pub quality_score: f64,
    pub information_content: f64,
}

/// 障碍退出类型
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BarrierExitType {
    TakeProfit,  // 触及上障碍（止盈）
    StopLoss,    // 触及下障碍（止损）
    TimeExpiry,  // 时间到期
    Neutral,     // 中性退出（接近入场价）
}

/// 元标签
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLabel {
    pub primary_label: i8, // -1, 0, 1
    pub meta_score: f64,   // 置信度/概率
    pub feature_importance: HashMap<String, f64>,
    pub model_version: String,
}

/// 分数差分标签
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FractionalDiffLabel {
    pub differencing_order: f64,
    pub stationarity_score: f64,
    pub adf_statistic: f64,
    pub adf_p_value: f64,
    pub memory_length: u32,
}

/// 波动率估计器
#[derive(Debug)]
pub struct VolatilityEstimator {
    config: VolatilityConfig,
    price_history: Arc<RwLock<VecDeque<f64>>>,
    volatility_history: Arc<RwLock<VecDeque<VolatilityPoint>>>,
    models: HashMap<String, Box<dyn VolatilityModel>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityConfig {
    pub window_size: usize,
    pub update_frequency_seconds: u64,
    pub models: Vec<VolatilityModelType>,
    pub ensemble_weights: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VolatilityModelType {
    RealizedVolatility,
    EWMA,
    GARCH,
    YangZhang,
    Parkinson,
    GarmanKlass,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityPoint {
    pub timestamp: DateTime<Utc>,
    pub realized_vol: f64,
    pub ewma_vol: f64,
    pub garch_vol: Option<f64>,
    pub high_low_vol: Option<f64>,
    pub ensemble_vol: f64,
    pub regime_vol: Option<f64>,
}

/// 波动率模型接口
pub trait VolatilityModel: std::fmt::Debug + Send + Sync {
    fn name(&self) -> &str;
    fn estimate(&self, prices: &[f64]) -> Result<f64>;
    fn update(&mut self, price: f64) -> Result<f64>;
    fn reset(&mut self);
}

/// 体制检测器
#[derive(Debug)]
pub struct RegimeDetector {
    config: RegimeConfig,
    hmm_model: Option<HiddenMarkovModel>,
    clustering_model: Option<ClusteringModel>,
    current_regime: Arc<RwLock<Option<String>>>,
    regime_history: Arc<RwLock<VecDeque<RegimePoint>>>,
    feature_extractor: Arc<FeatureExtractor>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeConfig {
    pub detection_method: RegimeDetectionMethod,
    pub lookback_window: usize,
    pub min_regime_duration: Duration,
    pub confidence_threshold: f64,
    pub features: Vec<RegimeFeature>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegimeDetectionMethod {
    HiddenMarkov,
    KMeansClustering,
    GaussianMixture,
    ThresholdBased,
    Ensemble,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegimeFeature {
    Volatility,
    Returns,
    Volume,
    Spread,
    Momentum,
    MeanReversion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimePoint {
    pub timestamp: DateTime<Utc>,
    pub regime_id: String,
    pub confidence: f64,
    pub features: HashMap<String, f64>,
    pub transition_probability: f64,
}

/// 隐马尔可夫模型
#[derive(Debug, Clone)]
pub struct HiddenMarkovModel {
    pub n_states: usize,
    pub transition_matrix: Vec<Vec<f64>>,
    pub emission_params: Vec<EmissionParameters>,
    pub initial_probs: Vec<f64>,
    pub current_state: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct EmissionParameters {
    pub mean: f64,
    pub variance: f64,
    pub distribution_type: DistributionType,
}

#[derive(Debug, Clone)]
pub enum DistributionType {
    Gaussian,
    StudentT { df: f64 },
    Laplace,
}

/// 聚类模型
#[derive(Debug, Clone)]
pub struct ClusteringModel {
    pub n_clusters: usize,
    pub cluster_centers: Vec<Vec<f64>>,
    pub cluster_assignments: Vec<usize>,
    pub inertia: f64,
}

/// 特征提取器
#[derive(Debug)]
pub struct FeatureExtractor {
    config: FeatureConfig,
    feature_history: Arc<RwLock<VecDeque<FeatureVector>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    pub features: Vec<FeatureType>,
    pub lookback_periods: Vec<usize>,
    pub normalization_method: NormalizationMethod,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureType {
    Price,
    Returns,
    LogReturns,
    Volatility,
    Volume,
    VolumePrice,
    RSI,
    MACD,
    BollingerBands,
    ATR,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationMethod {
    None,
    ZScore,
    MinMax,
    RobustScaler,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    pub timestamp: DateTime<Utc>,
    pub features: HashMap<String, f64>,
    pub regime: Option<String>,
}

/// 元标签生成器
#[derive(Debug)]
pub struct MetaLabeler {
    config: MetaLabelConfig,
    model: Option<Box<dyn MetaLabelModel>>,
    feature_importance: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLabelConfig {
    pub model_type: MetaLabelModelType,
    pub training_window: usize,
    pub retrain_frequency: usize,
    pub min_samples: usize,
    pub confidence_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetaLabelModelType {
    LogisticRegression,
    RandomForest,
    XGBoost,
    LightGBM,
    NeuralNetwork,
}

/// 元标签模型接口
pub trait MetaLabelModel: std::fmt::Debug + Send + Sync {
    fn train(&mut self, features: &[FeatureVector], labels: &[i8]) -> Result<()>;
    fn predict(&self, features: &FeatureVector) -> Result<MetaLabel>;
    fn get_feature_importance(&self) -> HashMap<String, f64>;
    fn model_info(&self) -> String;
}

impl TripleBarrierLabeler {
    /// 创建新的三重障碍标签生成器
    pub fn new(config: TripleBarrierConfig) -> Result<Self> {
        let volatility_estimator = Arc::new(VolatilityEstimator::new(
            VolatilityConfig::default()
        )?);
        
        let regime_detector = Arc::new(RegimeDetector::new(
            RegimeConfig::default()
        )?);
        
        let meta_labeler = Arc::new(MetaLabeler::new(
            MetaLabelConfig::default()
        )?);

        Ok(Self {
            config,
            price_cache: Arc::new(RwLock::new(VecDeque::new())),
            volume_cache: Arc::new(RwLock::new(VecDeque::new())),
            active_barriers: Arc::new(RwLock::new(HashMap::new())),
            label_history: Arc::new(RwLock::new(VecDeque::new())),
            volatility_estimator,
            regime_detector,
            meta_labeler,
        })
    }

    /// 更新价格数据
    pub async fn update_price(&self, price_point: PricePoint) -> Result<()> {
        // 更新价格缓存
        {
            let mut cache = self.price_cache.write().unwrap();
            cache.push_back(price_point.clone());
            
            // 保持缓存大小
            while cache.len() > self.config.volatility_window * 2 {
                cache.pop_front();
            }
        }

        // 更新波动率估计
        let price_f64: f64 = price_point.price.try_into().unwrap_or(0.0);
        self.volatility_estimator.update(price_f64).await?;

        // 更新体制检测
        if self.config.regime_aware {
            self.regime_detector.update(&price_point).await?;
        }

        // 检查活跃障碍
        self.check_active_barriers(&price_point).await?;

        Ok(())
    }

    /// 创建新的三重障碍
    pub async fn create_barrier(
        &self,
        symbol: String,
        entry_price: Decimal,
        signal_strength: Option<f64>,
        custom_params: Option<BarrierParameters>,
    ) -> Result<String> {
        let barrier_id = format!("barrier_{}_{}", symbol, Utc::now().timestamp_millis());
        let entry_time = Utc::now();

        // 获取当前波动率
        let current_volatility = self.volatility_estimator.get_current_volatility().await?;
        
        // 获取当前体制
        let current_regime = if self.config.regime_aware {
            self.regime_detector.get_current_regime().await?
        } else {
            None
        };

        // 计算障碍参数
        let params = custom_params.unwrap_or_else(|| {
            self.calculate_dynamic_parameters(
                &symbol,
                entry_price,
                current_volatility,
                signal_strength.unwrap_or(0.5),
                current_regime.as_deref(),
            )
        });

        // 创建活跃障碍
        let active_barrier = ActiveBarrier {
            id: barrier_id.clone(),
            symbol: symbol.clone(),
            entry_time,
            entry_price,
            upper_barrier: entry_price * (Decimal::ONE + Decimal::try_from(params.profit_threshold).unwrap_or(Decimal::ZERO)),
            lower_barrier: entry_price * (Decimal::ONE - Decimal::try_from(params.loss_threshold).unwrap_or(Decimal::ZERO)),
            time_barrier: entry_time + Duration::hours(params.time_horizon_hours as i64),
            current_price: entry_price,
            max_price: entry_price,
            min_price: entry_price,
            volatility_at_entry: current_volatility,
            regime_at_entry: current_regime,
            signal_strength,
            metadata: HashMap::new(),
        };

        // 存储活跃障碍
        {
            let mut barriers = self.active_barriers.write().unwrap();
            
            // 检查最大活跃障碍数
            if barriers.len() >= self.config.max_active_barriers {
                // 移除最旧的障碍
                if let Some((oldest_id, _)) = barriers.iter()
                    .min_by_key(|(_, b)| b.entry_time)
                    .map(|(id, b)| (id.clone(), b.clone())) {
                    
                    barriers.remove(&oldest_id);
                    log::warn!("Removed oldest barrier {} due to capacity limit", oldest_id);
                }
            }
            
            barriers.insert(barrier_id.clone(), active_barrier);
        }

        log::info!("Created barrier {} for {} at price {}", 
            barrier_id, symbol, entry_price);

        Ok(barrier_id)
    }

    /// 检查活跃障碍
    async fn check_active_barriers(&self, price_point: &PricePoint) -> Result<()> {
        let mut completed_barriers = Vec::new();
        
        // 检查每个活跃障碍
        {
            let mut barriers = self.active_barriers.write().unwrap();
            
            for (barrier_id, barrier) in barriers.iter_mut() {
                if barrier.symbol != price_point.symbol {
                    continue;
                }

                // 更新当前价格和极值
                barrier.current_price = price_point.price;
                barrier.max_price = barrier.max_price.max(price_point.price);
                barrier.min_price = barrier.min_price.min(price_point.price);

                // 检查障碍触发条件
                let exit_type = if price_point.price >= barrier.upper_barrier {
                    Some(BarrierExitType::TakeProfit)
                } else if price_point.price <= barrier.lower_barrier {
                    Some(BarrierExitType::StopLoss)
                } else if price_point.timestamp >= barrier.time_barrier {
                    Some(BarrierExitType::TimeExpiry)
                } else {
                    None
                };

                // 如果触发退出条件，标记为完成
                if let Some(exit_type) = exit_type {
                    completed_barriers.push((
                        barrier_id.clone(),
                        barrier.clone(),
                        exit_type,
                        price_point.price,
                        price_point.timestamp,
                    ));
                }
            }

            // 移除已完成的障碍
            for (barrier_id, _, _, _, _) in &completed_barriers {
                barriers.remove(barrier_id);
            }
        }

        // 生成标签
        for (barrier_id, barrier, exit_type, exit_price, exit_time) in completed_barriers {
            self.generate_label(barrier, exit_type, exit_price, exit_time).await?;
        }

        Ok(())
    }

    /// 生成三重障碍标签
    async fn generate_label(
        &self,
        barrier: ActiveBarrier,
        exit_type: BarrierExitType,
        exit_price: Decimal,
        exit_time: DateTime<Utc>,
    ) -> Result<()> {
        let holding_period_hours = (exit_time - barrier.entry_time).num_seconds() as f64 / 3600.0;
        let return_pct = (exit_price - barrier.entry_price) / barrier.entry_price * Decimal::from(100);
        
        // 计算MFE和MAE
        let mfe = (barrier.max_price - barrier.entry_price) / barrier.entry_price * Decimal::from(100);
        let mae = (barrier.entry_price - barrier.min_price) / barrier.entry_price * Decimal::from(100);

        // 获取当前体制
        let volatility_regime = if self.config.regime_aware {
            self.regime_detector.get_current_regime().await?
        } else {
            None
        };

        // 生成元标签
        let meta_label = if self.config.enable_meta_labeling {
            self.generate_meta_label(&barrier, &exit_type, return_pct).await?
        } else {
            None
        };

        // 生成分数差分标签
        let fractional_diff_label = if let Some(threshold) = self.config.fractional_diff_threshold {
            self.generate_fractional_diff_label(threshold).await?
        } else {
            None
        };

        // 计算置信度和质量评分
        let confidence = self.calculate_confidence(&barrier, &exit_type, holding_period_hours);
        let quality_score = self.calculate_quality_score(&barrier, return_pct, mfe, mae);
        let information_content = self.calculate_information_content(&barrier, return_pct);

        let label = TripleBarrierLabel {
            id: barrier.id,
            symbol: barrier.symbol,
            entry_time: barrier.entry_time,
            entry_price: barrier.entry_price,
            upper_barrier: barrier.upper_barrier,
            lower_barrier: barrier.lower_barrier,
            time_barrier: barrier.time_barrier,
            exit_type,
            exit_price,
            exit_time,
            return_pct,
            holding_period_hours,
            max_favorable_excursion: mfe,
            max_adverse_excursion: mae,
            volatility_regime,
            meta_label,
            fractional_diff_label,
            confidence,
            quality_score,
            information_content,
        };

        // 存储标签
        {
            let mut history = self.label_history.write().unwrap();
            history.push_back(label.clone());
            
            // 保持历史记录大小
            while history.len() > 10000 {
                history.pop_front();
            }
        }

        log::info!("Generated label for barrier {} with exit type {:?} and return {:.2}%",
            label.id, label.exit_type, return_pct);

        Ok(())
    }

    /// 计算动态障碍参数
    fn calculate_dynamic_parameters(
        &self,
        _symbol: &str,
        _entry_price: Decimal,
        volatility: f64,
        signal_strength: f64,
        regime: Option<&str>,
    ) -> BarrierParameters {
        let mut params = BarrierParameters::default();

        // 基于波动率调整
        let vol_multiplier = (volatility / 0.2).clamp(0.5, 2.0); // 基准20%波动率
        params.profit_threshold *= vol_multiplier;
        params.loss_threshold *= vol_multiplier;

        // 基于信号强度调整
        let strength_multiplier = signal_strength.abs().clamp(0.3, 1.5);
        params.profit_threshold *= strength_multiplier;
        params.loss_threshold *= strength_multiplier;

        // 基于体制调整
        if let Some(regime) = regime {
            match regime {
                "high_volatility" => {
                    params.profit_threshold *= 1.5;
                    params.loss_threshold *= 1.5;
                    params.time_horizon_hours = (params.time_horizon_hours as f64 * 0.7) as u32;
                }
                "low_volatility" => {
                    params.profit_threshold *= 0.8;
                    params.loss_threshold *= 0.8;
                    params.time_horizon_hours = (params.time_horizon_hours as f64 * 1.3) as u32;
                }
                "trending" => {
                    params.profit_threshold *= 1.2;
                    params.loss_threshold *= 0.9;
                }
                "mean_reverting" => {
                    params.profit_threshold *= 0.9;
                    params.loss_threshold *= 1.1;
                }
                _ => {}
            }
        }

        // 应用自适应因子
        params.profit_threshold *= self.config.adaptation_factor;
        params.loss_threshold *= self.config.adaptation_factor;

        params
    }

    /// 生成元标签
    async fn generate_meta_label(
        &self,
        barrier: &ActiveBarrier,
        exit_type: &BarrierExitType,
        return_pct: Decimal,
    ) -> Result<Option<MetaLabel>> {
        // 构建特征向量
        let features = self.extract_meta_features(barrier).await?;
        
        // 使用模型预测
        if let Some(meta_label) = self.meta_labeler.predict(&features).await? {
            Ok(Some(meta_label))
        } else {
            // 如果没有训练好的模型，使用规则生成
            let primary_label = match exit_type {
                BarrierExitType::TakeProfit => 1,
                BarrierExitType::StopLoss => -1,
                BarrierExitType::TimeExpiry => {
                    if return_pct > dec!(0.5) { 1 }
                    else if return_pct < dec!(-0.5) { -1 }
                    else { 0 }
                }
                BarrierExitType::Neutral => 0,
            };

            Ok(Some(MetaLabel {
                primary_label,
                meta_score: 0.5, // 默认置信度
                feature_importance: HashMap::new(),
                model_version: "rule_based_v1".to_string(),
            }))
        }
    }

    /// 生成分数差分标签
    async fn generate_fractional_diff_label(
        &self,
        _threshold: f64,
    ) -> Result<Option<FractionalDiffLabel>> {
        // 实现分数差分逻辑
        // 这里提供简化版本
        Ok(Some(FractionalDiffLabel {
            differencing_order: 0.5,
            stationarity_score: 0.8,
            adf_statistic: -3.5,
            adf_p_value: 0.01,
            memory_length: 100,
        }))
    }

    /// 提取元特征
    async fn extract_meta_features(&self, barrier: &ActiveBarrier) -> Result<FeatureVector> {
        let mut features = HashMap::new();
        
        // 价格相关特征
        let entry_price_f64: f64 = barrier.entry_price.try_into().unwrap_or(0.0);
        features.insert("entry_price".to_string(), entry_price_f64);
        let price_diff: f64 = (barrier.current_price - barrier.entry_price).try_into().unwrap_or(0.0);
        features.insert("price_position".to_string(), price_diff);
        
        // 波动率特征
        features.insert("volatility_at_entry".to_string(), 
            barrier.volatility_at_entry);
        
        // 信号强度
        if let Some(strength) = barrier.signal_strength {
            features.insert("signal_strength".to_string(), strength);
        }
        
        // 时间特征
        let time_elapsed = (Utc::now() - barrier.entry_time).num_seconds() as f64 / 3600.0;
        features.insert("time_elapsed_hours".to_string(), time_elapsed);
        
        // 障碍距离特征
        let upper_distance = (barrier.upper_barrier - barrier.current_price) / barrier.current_price;
        let lower_distance = (barrier.current_price - barrier.lower_barrier) / barrier.current_price;
        let upper_dist_f64: f64 = upper_distance.try_into().unwrap_or(0.0);
        let lower_dist_f64: f64 = lower_distance.try_into().unwrap_or(0.0);
        features.insert("upper_barrier_distance".to_string(), upper_dist_f64);
        features.insert("lower_barrier_distance".to_string(), lower_dist_f64);
        
        Ok(FeatureVector {
            timestamp: Utc::now(),
            features,
            regime: barrier.regime_at_entry.clone(),
        })
    }

    /// 计算置信度
    fn calculate_confidence(
        &self,
        barrier: &ActiveBarrier,
        exit_type: &BarrierExitType,
        holding_period_hours: f64,
    ) -> f64 {
        let mut confidence = 0.5; // 基础置信度
        
        // 基于退出类型调整
        match exit_type {
            BarrierExitType::TakeProfit | BarrierExitType::StopLoss => confidence += 0.3,
            BarrierExitType::TimeExpiry => confidence += 0.1,
            BarrierExitType::Neutral => confidence -= 0.1,
        }
        
        // 基于持有期调整
        let min_holding = self.config.min_holding_period_minutes as f64 / 60.0;
        if holding_period_hours >= min_holding {
            confidence += 0.2;
        }
        
        // 基于信号强度调整
        if let Some(strength) = barrier.signal_strength {
            confidence += strength.abs() * 0.2;
        }
        
        confidence.clamp(0.0, 1.0)
    }

    /// 计算质量评分
    fn calculate_quality_score(
        &self,
        barrier: &ActiveBarrier,
        return_pct: Decimal,
        mfe: Decimal,
        mae: Decimal,
    ) -> f64 {
        let mut score = 0.5; // 基础评分
        
        // 基于收益率调整
        let return_f64: f64 = return_pct.try_into().unwrap_or(0.0);
        if return_f64.abs() > 1.0 { // 大于1%的收益
            score += 0.2;
        }
        
        // 基于MFE/MAE比率调整
        if mae.is_zero() == false {
            let mfe_mae_ratio: f64 = (mfe / mae).try_into().unwrap_or(1.0);
            if mfe_mae_ratio > 1.5 {
                score += 0.2;
            }
        }
        
        // 基于波动率调整
        let vol_score = 1.0 - (barrier.volatility_at_entry - 0.2).abs().min(0.3) / 0.3;
        score += vol_score * 0.1;
        
        score.clamp(0.0, 1.0)
    }

    /// 计算信息含量
    fn calculate_information_content(&self, barrier: &ActiveBarrier, return_pct: Decimal) -> f64 {
        let return_f64_raw: f64 = return_pct.try_into().unwrap_or(0.0);
        let return_f64 = return_f64_raw.abs();
        let vol = barrier.volatility_at_entry;
        
        // 信息含量 = |收益率| / 波动率 * 信号强度
        let base_ic = return_f64 / (vol * 100.0); // 转换为百分比
        let signal_multiplier = barrier.signal_strength.unwrap_or(0.5);
        
        (base_ic * signal_multiplier).min(1.0)
    }

    /// 获取标签历史
    pub fn get_label_history(&self, limit: Option<usize>) -> Vec<TripleBarrierLabel> {
        let history = self.label_history.read().unwrap();
        
        if let Some(limit) = limit {
            history.iter().rev().take(limit).cloned().collect()
        } else {
            history.iter().cloned().collect()
        }
    }

    /// 获取标签统计
    pub fn get_label_statistics(&self) -> LabelStatistics {
        let history = self.label_history.read().unwrap();
        let total_labels = history.len();
        
        if total_labels == 0 {
            return LabelStatistics::default();
        }
        
        let mut exit_type_counts = HashMap::new();
        let mut total_return = Decimal::ZERO;
        let mut positive_returns = 0;
        let mut quality_scores = Vec::new();
        let mut information_contents = Vec::new();
        
        for label in history.iter() {
            *exit_type_counts.entry(label.exit_type.clone()).or_insert(0) += 1;
            total_return += label.return_pct;
            
            if label.return_pct > Decimal::ZERO {
                positive_returns += 1;
            }
            
            quality_scores.push(label.quality_score);
            information_contents.push(label.information_content);
        }
        
        let win_rate = positive_returns as f64 / total_labels as f64;
        let avg_return_decimal = total_return / Decimal::from(total_labels);
        let avg_return: f64 = avg_return_decimal.try_into().unwrap_or(0.0);
        let avg_quality_score = quality_scores.iter().sum::<f64>() / quality_scores.len() as f64;
        let avg_information_content = information_contents.iter().sum::<f64>() / information_contents.len() as f64;
        
        LabelStatistics {
            total_labels,
            exit_type_distribution: exit_type_counts,
            win_rate,
            average_return_pct: avg_return,
            average_quality_score: avg_quality_score,
            average_information_content: avg_information_content,
        }
    }
}

/// 障碍参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BarrierParameters {
    pub profit_threshold: f64,
    pub loss_threshold: f64,
    pub time_horizon_hours: u32,
}

impl Default for BarrierParameters {
    fn default() -> Self {
        Self {
            profit_threshold: 0.02,  // 2% 止盈
            loss_threshold: 0.01,    // 1% 止损
            time_horizon_hours: 24,  // 24小时时间障碍
        }
    }
}

/// 标签统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelStatistics {
    pub total_labels: usize,
    pub exit_type_distribution: HashMap<BarrierExitType, usize>,
    pub win_rate: f64,
    pub average_return_pct: f64,
    pub average_quality_score: f64,
    pub average_information_content: f64,
}

impl Default for LabelStatistics {
    fn default() -> Self {
        Self {
            total_labels: 0,
            exit_type_distribution: HashMap::new(),
            win_rate: 0.0,
            average_return_pct: 0.0,
            average_quality_score: 0.0,
            average_information_content: 0.0,
        }
    }
}

// 各个组件的实现
impl VolatilityEstimator {
    pub fn new(config: VolatilityConfig) -> Result<Self> {
        let mut models: HashMap<String, Box<dyn VolatilityModel>> = HashMap::new();
        
        for model_type in &config.models {
            let model: Box<dyn VolatilityModel> = match model_type {
                VolatilityModelType::RealizedVolatility => Box::new(RealizedVolatilityModel::new()),
                VolatilityModelType::EWMA => Box::new(EWMAVolatilityModel::new(0.94)),
                _ => continue, // 其他模型暂时跳过
            };
            models.insert(model.name().to_string(), model);
        }
        
        Ok(Self {
            config,
            price_history: Arc::new(RwLock::new(VecDeque::new())),
            volatility_history: Arc::new(RwLock::new(VecDeque::new())),
            models,
        })
    }

    pub async fn update(&self, price: f64) -> Result<f64> {
        // 更新价格历史
        {
            let mut history = self.price_history.write().unwrap();
            history.push_back(price);
            
            while history.len() > self.config.window_size {
                history.pop_front();
            }
        }
        
        // 计算波动率
        let volatility = self.calculate_volatility().await?;
        
        // 更新波动率历史
        {
            let mut vol_history = self.volatility_history.write().unwrap();
            vol_history.push_back(VolatilityPoint {
                timestamp: Utc::now(),
                realized_vol: volatility,
                ewma_vol: volatility,
                garch_vol: None,
                high_low_vol: None,
                ensemble_vol: volatility,
                regime_vol: None,
            });
            
            while vol_history.len() > 1000 {
                vol_history.pop_front();
            }
        }
        
        Ok(volatility)
    }

    async fn calculate_volatility(&self) -> Result<f64> {
        let prices = {
            let history = self.price_history.read().unwrap();
            history.iter().cloned().collect::<Vec<_>>()
        };
        
        if prices.len() < 2 {
            return Ok(0.0);
        }
        
        // 使用实现的波动率估计
        if let Some(model) = self.models.get("realized_volatility") {
            model.estimate(&prices)
        } else {
            // 简单实现
            let returns: Vec<f64> = prices.windows(2)
                .map(|w| (w[1] / w[0]).ln())
                .collect();
            
            let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>() / (returns.len() - 1) as f64;
            
            Ok(variance.sqrt() * (252.0_f64).sqrt()) // 年化波动率
        }
    }

    pub async fn get_current_volatility(&self) -> Result<f64> {
        let vol_history = self.volatility_history.read().unwrap();
        
        if let Some(latest) = vol_history.back() {
            Ok(latest.ensemble_vol)
        } else {
            Ok(0.2) // 默认20%波动率
        }
    }
}

impl RegimeDetector {
    pub fn new(config: RegimeConfig) -> Result<Self> {
        Ok(Self {
            config,
            hmm_model: None,
            clustering_model: None,
            current_regime: Arc::new(RwLock::new(None)),
            regime_history: Arc::new(RwLock::new(VecDeque::new())),
            feature_extractor: Arc::new(FeatureExtractor::new(FeatureConfig::default())?),
        })
    }

    pub async fn update(&self, _price_point: &PricePoint) -> Result<()> {
        // 简化实现：基于波动率的体制检测
        // 实际实现会更复杂
        Ok(())
    }

    pub async fn get_current_regime(&self) -> Result<Option<String>> {
        Ok(self.current_regime.read().unwrap().clone())
    }
}

impl FeatureExtractor {
    pub fn new(config: FeatureConfig) -> Result<Self> {
        Ok(Self {
            config,
            feature_history: Arc::new(RwLock::new(VecDeque::new())),
        })
    }
}

impl MetaLabeler {
    pub fn new(config: MetaLabelConfig) -> Result<Self> {
        Ok(Self {
            config,
            model: None,
            feature_importance: HashMap::new(),
        })
    }

    pub async fn predict(&self, _features: &FeatureVector) -> Result<Option<MetaLabel>> {
        // 简化实现
        Ok(None)
    }
}

// 波动率模型实现
#[derive(Debug)]
pub struct RealizedVolatilityModel {
    window: usize,
}

impl RealizedVolatilityModel {
    pub fn new() -> Self {
        Self { window: 20 }
    }
}

impl VolatilityModel for RealizedVolatilityModel {
    fn name(&self) -> &str {
        "realized_volatility"
    }

    fn estimate(&self, prices: &[f64]) -> Result<f64> {
        if prices.len() < 2 {
            return Ok(0.0);
        }
        
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
        
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / (returns.len() - 1) as f64;
        
        Ok(variance.sqrt() * (252.0_f64).sqrt())
    }

    fn update(&mut self, _price: f64) -> Result<f64> {
        Ok(0.0) // 简化实现
    }

    fn reset(&mut self) {
        // 重置模型状态
    }
}

#[derive(Debug)]
pub struct EWMAVolatilityModel {
    lambda: f64,
    current_vol: f64,
}

impl EWMAVolatilityModel {
    pub fn new(lambda: f64) -> Self {
        Self { lambda, current_vol: 0.0 }
    }
}

impl VolatilityModel for EWMAVolatilityModel {
    fn name(&self) -> &str {
        "ewma"
    }

    fn estimate(&self, _prices: &[f64]) -> Result<f64> {
        Ok(self.current_vol)
    }

    fn update(&mut self, price: f64) -> Result<f64> {
        // 简化EWMA实现
        self.current_vol = self.lambda * self.current_vol + (1.0 - self.lambda) * price * price;
        Ok(self.current_vol.sqrt())
    }

    fn reset(&mut self) {
        self.current_vol = 0.0;
    }
}

// 默认配置
impl Default for TripleBarrierConfig {
    fn default() -> Self {
        Self {
            default_time_horizon_hours: 24,
            default_profit_multiplier: 2.0,
            default_loss_multiplier: 1.0,
            volatility_window: 100,
            min_holding_period_minutes: 60,
            max_active_barriers: 100,
            enable_dynamic_adjustment: true,
            enable_meta_labeling: true,
            regime_aware: true,
            fractional_diff_threshold: Some(0.8),
            adaptation_factor: 1.0,
        }
    }
}

impl Default for VolatilityConfig {
    fn default() -> Self {
        Self {
            window_size: 20,
            update_frequency_seconds: 60,
            models: vec![
                VolatilityModelType::RealizedVolatility,
                VolatilityModelType::EWMA,
            ],
            ensemble_weights: {
                let mut weights = HashMap::new();
                weights.insert("realized_volatility".to_string(), 0.6);
                weights.insert("ewma".to_string(), 0.4);
                weights
            },
        }
    }
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            detection_method: RegimeDetectionMethod::HiddenMarkov,
            lookback_window: 100,
            min_regime_duration: Duration::hours(4),
            confidence_threshold: 0.7,
            features: vec![
                RegimeFeature::Volatility,
                RegimeFeature::Returns,
                RegimeFeature::Volume,
            ],
        }
    }
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            features: vec![
                FeatureType::Price,
                FeatureType::Returns,
                FeatureType::Volatility,
                FeatureType::Volume,
            ],
            lookback_periods: vec![5, 10, 20],
            normalization_method: NormalizationMethod::ZScore,
        }
    }
}

impl Default for MetaLabelConfig {
    fn default() -> Self {
        Self {
            model_type: MetaLabelModelType::LogisticRegression,
            training_window: 1000,
            retrain_frequency: 100,
            min_samples: 50,
            confidence_threshold: 0.6,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_triple_barrier_labeler_creation() {
        let config = TripleBarrierConfig::default();
        let labeler = TripleBarrierLabeler::new(config).unwrap();
        
        assert!(true); // 基本创建测试
    }
    
    #[tokio::test]
    async fn test_barrier_creation() {
        let config = TripleBarrierConfig::default();
        let labeler = TripleBarrierLabeler::new(config).unwrap();
        
        let barrier_id = labeler.create_barrier(
            "AAPL".to_string(),
            Decimal::from(150),
            Some(0.8),
            None,
        ).await.unwrap();
        
        assert!(!barrier_id.is_empty());
    }
    
    #[tokio::test]
    async fn test_price_update() {
        let config = TripleBarrierConfig::default();
        let labeler = TripleBarrierLabeler::new(config).unwrap();
        
        let price_point = PricePoint {
            timestamp: Utc::now(),
            symbol: "AAPL".to_string(),
            price: Decimal::from(150),
            bid: Some(Decimal::from_f64(149.95).unwrap()),
            ask: Some(Decimal::from_f64(150.05).unwrap()),
            volume: Some(Decimal::from(1000)),
        };
        
        labeler.update_price(price_point).await.unwrap();
    }
    
    #[test]
    fn test_volatility_model() {
        let mut model = RealizedVolatilityModel::new();
        let prices = vec![100.0, 101.0, 100.5, 102.0, 101.5];
        
        let vol = model.estimate(&prices).unwrap();
        assert!(vol > 0.0);
    }
}