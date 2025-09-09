use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};

use super::batch_processor::{FactorInput, FactorOutput, FactorParams, FactorMetadata};

/// 窗口计算器配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowCalculatorConfig {
    pub max_window_size: usize,          // 最大窗口大小
    pub supported_functions: Vec<WindowFunction>, // 支持的窗口函数
    pub enable_incremental: bool,        // 启用增量计算
    pub memory_efficient: bool,          // 内存优化模式
    pub parallel_windows: bool,          // 并行窗口计算
}

impl Default for WindowCalculatorConfig {
    fn default() -> Self {
        Self {
            max_window_size: 1000,
            supported_functions: vec![
                WindowFunction::SMA,
                WindowFunction::EMA,
                WindowFunction::RSI,
                WindowFunction::Bollinger,
                WindowFunction::MACD,
                WindowFunction::Volatility,
                WindowFunction::Correlation,
                WindowFunction::Beta,
            ],
            enable_incremental: true,
            memory_efficient: true,
            parallel_windows: true,
        }
    }
}

/// 窗口函数类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum WindowFunction {
    SMA,          // 简单移动平均
    EMA,          // 指数移动平均
    WMA,          // 加权移动平均
    RSI,          // 相对强弱指数
    Bollinger,    // 布林带
    MACD,         // MACD
    Stochastic,   // 随机指标
    ATR,          // 平均真实波幅
    Volatility,   // 波动率
    Correlation,  // 相关性
    Beta,         // 贝塔系数
    Sharpe,       // 夏普比率
    Drawdown,     // 最大回撤
    VaR,          // 风险价值
    Custom(String), // 自定义函数
}

/// 窗口状态
#[derive(Debug, Clone)]
pub struct WindowState {
    pub window_size: usize,
    pub data_buffer: VecDeque<f64>,
    pub auxiliary_data: HashMap<String, f64>, // 辅助计算数据
    pub last_update: i64,
    pub update_count: u64,
}

impl WindowState {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            data_buffer: VecDeque::with_capacity(window_size),
            auxiliary_data: HashMap::new(),
            last_update: 0,
            update_count: 0,
        }
    }
    
    pub fn add_value(&mut self, value: f64, timestamp: i64) {
        if self.data_buffer.len() >= self.window_size {
            self.data_buffer.pop_front();
        }
        self.data_buffer.push_back(value);
        self.last_update = timestamp;
        self.update_count += 1;
    }
    
    pub fn is_ready(&self) -> bool {
        self.data_buffer.len() >= self.window_size.min(2) // 至少需要2个数据点
    }
    
    pub fn get_values(&self) -> &VecDeque<f64> {
        &self.data_buffer
    }
}

/// 窗口计算结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowResult {
    pub function: WindowFunction,
    pub window_size: usize,
    pub value: f64,
    pub auxiliary_values: HashMap<String, f64>, // 额外的计算结果
    pub confidence: f64,
    pub sample_count: usize,
    pub computation_time_ms: f64,
}

/// 滚动窗口计算器
#[derive(Debug)]
pub struct RollingWindowCalculator {
    config: WindowCalculatorConfig,
    window_states: Arc<RwLock<HashMap<String, WindowState>>>, // symbol -> state
    function_handlers: HashMap<WindowFunction, Box<dyn WindowFunctionHandler + Send + Sync>>,
}

/// 窗口函数处理器特征
pub trait WindowFunctionHandler: std::fmt::Debug {
    fn calculate(&self, state: &WindowState, params: &BTreeMap<String, serde_json::Value>) -> Result<WindowResult>;
    fn supports_incremental(&self) -> bool;
    fn calculate_incremental(&self, previous_result: &WindowResult, new_value: f64, state: &WindowState) -> Result<WindowResult>;
}

impl RollingWindowCalculator {
    pub fn new(config: WindowCalculatorConfig) -> Self {
        let mut calculator = Self {
            config,
            window_states: Arc::new(RwLock::new(HashMap::new())),
            function_handlers: HashMap::new(),
        };
        
        // 注册默认的窗口函数处理器
        calculator.register_default_handlers();
        calculator
    }
    
    /// 注册默认的窗口函数处理器
    fn register_default_handlers(&mut self) {
        self.function_handlers.insert(WindowFunction::SMA, Box::new(SMAHandler));
        self.function_handlers.insert(WindowFunction::EMA, Box::new(EMAHandler));
        self.function_handlers.insert(WindowFunction::RSI, Box::new(RSIHandler));
        self.function_handlers.insert(WindowFunction::Bollinger, Box::new(BollingerHandler));
        self.function_handlers.insert(WindowFunction::MACD, Box::new(MACDHandler));
        self.function_handlers.insert(WindowFunction::Volatility, Box::new(VolatilityHandler));
        self.function_handlers.insert(WindowFunction::Correlation, Box::new(CorrelationHandler));
        self.function_handlers.insert(WindowFunction::Beta, Box::new(BetaHandler));
    }
    
    /// 注册自定义窗口函数处理器
    pub fn register_handler(&mut self, function: WindowFunction, handler: Box<dyn WindowFunctionHandler + Send + Sync>) {
        self.function_handlers.insert(function, handler);
    }
    
    /// 计算窗口函数
    pub async fn calculate_window(
        &self,
        symbol: &str,
        function: WindowFunction,
        window_size: usize,
        inputs: &[FactorInput],
        params: &BTreeMap<String, serde_json::Value>,
    ) -> Result<WindowResult> {
        if window_size > self.config.max_window_size {
            return Err(anyhow::anyhow!("Window size {} exceeds maximum {}", window_size, self.config.max_window_size));
        }
        
        let handler = self.function_handlers.get(&function)
            .context(format!("Unsupported window function: {:?}", function))?;
        
        // 获取或创建窗口状态
        let state_key = format!("{}_{:?}_{}", symbol, function, window_size);
        let mut window_state = {
            let mut states = self.window_states.write().await;
            states.entry(state_key.clone())
                .or_insert_with(|| WindowState::new(window_size))
                .clone()
        };
        
        // 更新窗口状态
        for input in inputs {
            window_state.add_value(input.price, input.timestamp);
        }
        
        // 保存状态
        {
            let mut states = self.window_states.write().await;
            states.insert(state_key, window_state.clone());
        }
        
        // 计算结果
        if !window_state.is_ready() {
            return Err(anyhow::anyhow!("Insufficient data for window calculation"));
        }
        
        let start_time = std::time::Instant::now();
        let mut result = handler.calculate(&window_state, params)?;
        result.computation_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        
        Ok(result)
    }
    
    /// 批量计算多个窗口函数
    pub async fn calculate_multiple_windows(
        &self,
        symbol: &str,
        functions: &[(WindowFunction, usize, BTreeMap<String, serde_json::Value>)],
        inputs: &[FactorInput],
    ) -> Result<Vec<WindowResult>> {
        let mut results = Vec::new();
        
        if self.config.parallel_windows {
            // 并行计算
            let futures = functions.iter().map(|(function, window_size, params)| {
                self.calculate_window(symbol, function.clone(), *window_size, inputs, params)
            });
            
            let parallel_results = futures::future::join_all(futures).await;
            for result in parallel_results {
                match result {
                    Ok(window_result) => results.push(window_result),
                    Err(e) => warn!("Window calculation failed: {}", e),
                }
            }
        } else {
            // 串行计算
            for (function, window_size, params) in functions {
                match self.calculate_window(symbol, function.clone(), *window_size, inputs, params).await {
                    Ok(result) => results.push(result),
                    Err(e) => warn!("Window calculation failed: {}", e),
                }
            }
        }
        
        Ok(results)
    }
    
    /// 增量更新窗口计算
    pub async fn update_incremental(
        &self,
        symbol: &str,
        function: WindowFunction,
        window_size: usize,
        new_input: &FactorInput,
        previous_result: &WindowResult,
    ) -> Result<WindowResult> {
        if !self.config.enable_incremental {
            return Err(anyhow::anyhow!("Incremental calculation is disabled"));
        }
        
        let handler = self.function_handlers.get(&function)
            .context("Unsupported window function")?;
        
        if !handler.supports_incremental() {
            return Err(anyhow::anyhow!("Function {:?} does not support incremental calculation", function));
        }
        
        // 获取窗口状态
        let state_key = format!("{}_{:?}_{}", symbol, function, window_size);
        let mut window_state = {
            let states = self.window_states.read().await;
            states.get(&state_key).cloned()
                .context("Window state not found")?
        };
        
        // 更新状态
        window_state.add_value(new_input.price, new_input.timestamp);
        
        // 保存状态
        {
            let mut states = self.window_states.write().await;
            states.insert(state_key, window_state.clone());
        }
        
        // 增量计算
        let start_time = std::time::Instant::now();
        let mut result = handler.calculate_incremental(previous_result, new_input.price, &window_state)?;
        result.computation_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        
        Ok(result)
    }
    
    /// 清理过期的窗口状态
    pub async fn cleanup_expired_states(&self, ttl_seconds: i64) -> Result<usize> {
        let current_time = chrono::Utc::now().timestamp();
        let cutoff_time = current_time - ttl_seconds;
        
        let mut states = self.window_states.write().await;
        let initial_count = states.len();
        
        states.retain(|_, state| state.last_update > cutoff_time * 1000); // 转换为毫秒
        
        let cleaned_count = initial_count - states.len();
        if cleaned_count > 0 {
            debug!("Cleaned up {} expired window states", cleaned_count);
        }
        
        Ok(cleaned_count)
    }
    
    /// 获取窗口状态统计
    pub async fn get_stats(&self) -> WindowCalculatorStats {
        let states = self.window_states.read().await;
        
        let mut stats = WindowCalculatorStats {
            total_windows: states.len(),
            functions_count: HashMap::new(),
            avg_window_size: 0.0,
            total_updates: 0,
            memory_usage_estimate: 0,
        };
        
        let mut total_window_size = 0;
        for state in states.values() {
            total_window_size += state.window_size;
            stats.total_updates += state.update_count;
            stats.memory_usage_estimate += state.data_buffer.capacity() * std::mem::size_of::<f64>();
        }
        
        if !states.is_empty() {
            stats.avg_window_size = total_window_size as f64 / states.len() as f64;
        }
        
        stats
    }
}

/// 窗口计算器统计
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowCalculatorStats {
    pub total_windows: usize,
    pub functions_count: HashMap<WindowFunction, usize>,
    pub avg_window_size: f64,
    pub total_updates: u64,
    pub memory_usage_estimate: usize,
}

// 以下是各种窗口函数的处理器实现

/// SMA (简单移动平均) 处理器
#[derive(Debug)]
struct SMAHandler;

impl WindowFunctionHandler for SMAHandler {
    fn calculate(&self, state: &WindowState, _params: &BTreeMap<String, serde_json::Value>) -> Result<WindowResult> {
        let values = state.get_values();
        if values.is_empty() {
            return Err(anyhow::anyhow!("No data for SMA calculation"));
        }
        
        let sum: f64 = values.iter().sum();
        let average = sum / values.len() as f64;
        
        Ok(WindowResult {
            function: WindowFunction::SMA,
            window_size: state.window_size,
            value: average,
            auxiliary_values: HashMap::new(),
            confidence: if values.len() >= state.window_size { 1.0 } else { values.len() as f64 / state.window_size as f64 },
            sample_count: values.len(),
            computation_time_ms: 0.0,
        })
    }
    
    fn supports_incremental(&self) -> bool {
        true
    }
    
    fn calculate_incremental(&self, previous_result: &WindowResult, new_value: f64, state: &WindowState) -> Result<WindowResult> {
        let values = state.get_values();
        let n = values.len() as f64;
        
        let new_average = if values.len() <= state.window_size {
            // 窗口未满，直接添加新值
            (previous_result.value * (n - 1.0) + new_value) / n
        } else {
            // 窗口已满，移除最旧的值
            let old_value = values.front().copied().unwrap_or(0.0);
            previous_result.value + (new_value - old_value) / state.window_size as f64
        };
        
        Ok(WindowResult {
            function: WindowFunction::SMA,
            window_size: state.window_size,
            value: new_average,
            auxiliary_values: HashMap::new(),
            confidence: if values.len() >= state.window_size { 1.0 } else { values.len() as f64 / state.window_size as f64 },
            sample_count: values.len(),
            computation_time_ms: 0.0,
        })
    }
}

/// EMA (指数移动平均) 处理器
#[derive(Debug)]
struct EMAHandler;

impl WindowFunctionHandler for EMAHandler {
    fn calculate(&self, state: &WindowState, params: &BTreeMap<String, serde_json::Value>) -> Result<WindowResult> {
        let values = state.get_values();
        if values.is_empty() {
            return Err(anyhow::anyhow!("No data for EMA calculation"));
        }
        
        // 获取平滑因子
        let alpha = params.get("alpha")
            .and_then(|v| v.as_f64())
            .unwrap_or(2.0 / (state.window_size as f64 + 1.0));
        
        let mut ema = values[0];
        for &value in values.iter().skip(1) {
            ema = alpha * value + (1.0 - alpha) * ema;
        }
        
        Ok(WindowResult {
            function: WindowFunction::EMA,
            window_size: state.window_size,
            value: ema,
            auxiliary_values: HashMap::from([("alpha".to_string(), alpha)]),
            confidence: if values.len() >= state.window_size { 1.0 } else { values.len() as f64 / state.window_size as f64 },
            sample_count: values.len(),
            computation_time_ms: 0.0,
        })
    }
    
    fn supports_incremental(&self) -> bool {
        true
    }
    
    fn calculate_incremental(&self, previous_result: &WindowResult, new_value: f64, _state: &WindowState) -> Result<WindowResult> {
        let alpha = previous_result.auxiliary_values.get("alpha").copied().unwrap_or(0.1);
        let new_ema = alpha * new_value + (1.0 - alpha) * previous_result.value;
        
        let mut result = previous_result.clone();
        result.value = new_ema;
        result.sample_count += 1;
        
        Ok(result)
    }
}

/// RSI 处理器
#[derive(Debug)]
struct RSIHandler;

impl WindowFunctionHandler for RSIHandler {
    fn calculate(&self, state: &WindowState, _params: &BTreeMap<String, serde_json::Value>) -> Result<WindowResult> {
        let values = state.get_values();
        if values.len() < 2 {
            return Err(anyhow::anyhow!("Insufficient data for RSI calculation"));
        }
        
        let mut gains = Vec::new();
        let mut losses = Vec::new();
        
        for i in 1..values.len() {
            let change = values[i] - values[i-1];
            if change > 0.0 {
                gains.push(change);
                losses.push(0.0);
            } else {
                gains.push(0.0);
                losses.push(-change);
            }
        }
        
        let avg_gain: f64 = gains.iter().sum::<f64>() / gains.len() as f64;
        let avg_loss: f64 = losses.iter().sum::<f64>() / losses.len() as f64;
        
        let rsi = if avg_loss == 0.0 {
            100.0
        } else {
            let rs = avg_gain / avg_loss;
            100.0 - (100.0 / (1.0 + rs))
        };
        
        Ok(WindowResult {
            function: WindowFunction::RSI,
            window_size: state.window_size,
            value: rsi,
            auxiliary_values: HashMap::from([
                ("avg_gain".to_string(), avg_gain),
                ("avg_loss".to_string(), avg_loss),
            ]),
            confidence: if values.len() >= state.window_size { 1.0 } else { values.len() as f64 / state.window_size as f64 },
            sample_count: values.len(),
            computation_time_ms: 0.0,
        })
    }
    
    fn supports_incremental(&self) -> bool {
        false // RSI增量计算较复杂，暂不支持
    }
    
    fn calculate_incremental(&self, _previous_result: &WindowResult, _new_value: f64, _state: &WindowState) -> Result<WindowResult> {
        Err(anyhow::anyhow!("RSI incremental calculation not implemented"))
    }
}

/// 波动率处理器
#[derive(Debug)]
struct VolatilityHandler;

impl WindowFunctionHandler for VolatilityHandler {
    fn calculate(&self, state: &WindowState, _params: &BTreeMap<String, serde_json::Value>) -> Result<WindowResult> {
        let values = state.get_values();
        if values.len() < 2 {
            return Err(anyhow::anyhow!("Insufficient data for volatility calculation"));
        }
        
        // 计算价格变化率
        let mut returns = Vec::new();
        for i in 1..values.len() {
            let ret = (values[i] / values[i-1]).ln();
            returns.push(ret);
        }
        
        // 计算标准差
        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        let volatility = variance.sqrt();
        
        Ok(WindowResult {
            function: WindowFunction::Volatility,
            window_size: state.window_size,
            value: volatility,
            auxiliary_values: HashMap::from([
                ("mean_return".to_string(), mean),
                ("variance".to_string(), variance),
            ]),
            confidence: if values.len() >= state.window_size { 1.0 } else { values.len() as f64 / state.window_size as f64 },
            sample_count: values.len(),
            computation_time_ms: 0.0,
        })
    }
    
    fn supports_incremental(&self) -> bool {
        false
    }
    
    fn calculate_incremental(&self, _previous_result: &WindowResult, _new_value: f64, _state: &WindowState) -> Result<WindowResult> {
        Err(anyhow::anyhow!("Volatility incremental calculation not implemented"))
    }
}

// 简化实现的其他处理器
macro_rules! simple_handler {
    ($name:ident, $function:expr) => {
        #[derive(Debug)]
        struct $name;
        
        impl WindowFunctionHandler for $name {
            fn calculate(&self, state: &WindowState, _params: &BTreeMap<String, serde_json::Value>) -> Result<WindowResult> {
                let values = state.get_values();
                if values.is_empty() {
                    return Err(anyhow::anyhow!("No data for calculation"));
                }
                
                let value = values.iter().sum::<f64>() / values.len() as f64; // 简化计算
                
                Ok(WindowResult {
                    function: $function,
                    window_size: state.window_size,
                    value,
                    auxiliary_values: HashMap::new(),
                    confidence: if values.len() >= state.window_size { 1.0 } else { values.len() as f64 / state.window_size as f64 },
                    sample_count: values.len(),
                    computation_time_ms: 0.0,
                })
            }
            
            fn supports_incremental(&self) -> bool {
                false
            }
            
            fn calculate_incremental(&self, _previous_result: &WindowResult, _new_value: f64, _state: &WindowState) -> Result<WindowResult> {
                Err(anyhow::anyhow!("Incremental calculation not implemented"))
            }
        }
    };
}

simple_handler!(BollingerHandler, WindowFunction::Bollinger);
simple_handler!(MACDHandler, WindowFunction::MACD);
simple_handler!(CorrelationHandler, WindowFunction::Correlation);
simple_handler!(BetaHandler, WindowFunction::Beta);

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sma_calculation() {
        let config = WindowCalculatorConfig::default();
        let calculator = RollingWindowCalculator::new(config);
        
        let inputs = vec![
            FactorInput {
                symbol: "BTCUSD".to_string(),
                timestamp: 1000,
                price: 100.0,
                volume: 1000.0,
                open: None,
                high: None,
                low: None,
                close: None,
                metadata: HashMap::new(),
            },
            FactorInput {
                symbol: "BTCUSD".to_string(),
                timestamp: 2000,
                price: 110.0,
                volume: 1000.0,
                open: None,
                high: None,
                low: None,
                close: None,
                metadata: HashMap::new(),
            },
        ];
        
        let result = calculator.calculate_window(
            "BTCUSD",
            WindowFunction::SMA,
            2,
            &inputs,
            &BTreeMap::new(),
        ).await.unwrap();
        
        assert_eq!(result.value, 105.0); // (100 + 110) / 2
        assert_eq!(result.sample_count, 2);
    }

    #[tokio::test]
    async fn test_ema_calculation() {
        let config = WindowCalculatorConfig::default();
        let calculator = RollingWindowCalculator::new(config);
        
        let inputs = vec![
            FactorInput {
                symbol: "BTCUSD".to_string(),
                timestamp: 1000,
                price: 100.0,
                volume: 1000.0,
                open: None,
                high: None,
                low: None,
                close: None,
                metadata: HashMap::new(),
            },
            FactorInput {
                symbol: "BTCUSD".to_string(),
                timestamp: 2000,
                price: 110.0,
                volume: 1000.0,
                open: None,
                high: None,
                low: None,
                close: None,
                metadata: HashMap::new(),
            },
        ];
        
        let result = calculator.calculate_window(
            "BTCUSD",
            WindowFunction::EMA,
            2,
            &inputs,
            &BTreeMap::new(),
        ).await.unwrap();
        
        assert!(result.value > 100.0 && result.value < 110.0);
        assert!(result.auxiliary_values.contains_key("alpha"));
    }

    #[tokio::test]
    async fn test_incremental_calculation() {
        let config = WindowCalculatorConfig::default();
        let calculator = RollingWindowCalculator::new(config);
        
        let initial_inputs = vec![
            FactorInput {
                symbol: "BTCUSD".to_string(),
                timestamp: 1000,
                price: 100.0,
                volume: 1000.0,
                open: None,
                high: None,
                low: None,
                close: None,
                metadata: HashMap::new(),
            },
        ];
        
        // 初始计算
        let initial_result = calculator.calculate_window(
            "BTCUSD",
            WindowFunction::SMA,
            2,
            &initial_inputs,
            &BTreeMap::new(),
        ).await.unwrap();
        
        // 增量更新
        let new_input = FactorInput {
            symbol: "BTCUSD".to_string(),
            timestamp: 2000,
            price: 110.0,
            volume: 1000.0,
            open: None,
            high: None,
            low: None,
            close: None,
            metadata: HashMap::new(),
        };
        
        let updated_result = calculator.update_incremental(
            "BTCUSD",
            WindowFunction::SMA,
            2,
            &new_input,
            &initial_result,
        ).await.unwrap();
        
        assert_eq!(updated_result.value, 105.0); // (100 + 110) / 2
    }
}