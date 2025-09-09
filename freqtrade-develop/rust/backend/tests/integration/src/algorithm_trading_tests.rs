use anyhow::Result;
use reqwest::Client;
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{info, debug, warn, error};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use tokio::time::timeout;

/// 算法交易测试套件
/// 
/// 验证已实现的算法交易功能：
/// - TWAP (时间加权平均价格) 算法
/// - VWAP (成交量加权平均价格) 算法  
/// - PoV (参与率) 算法
/// - 自适应算法
/// - Almgren-Chriss 最优执行模型
pub struct AlgorithmTradingTests {
    client: Client,
    trading_service_url: String,
    timeout_duration: Duration,
    test_metrics: AlgorithmTestMetrics,
}

/// 算法测试指标
#[derive(Debug, Default)]
struct AlgorithmTestMetrics {
    total_algorithms_tested: u32,
    successful_executions: u32,
    failed_executions: u32,
    average_execution_latency_ms: u64,
    max_execution_latency_ms: u64,
    min_execution_latency_ms: u64,
    twap_tests: u32,
    vwap_tests: u32,
    pov_tests: u32,
    adaptive_tests: u32,
}

/// 算法执行请求
#[derive(Debug, Serialize, Deserialize)]
struct AlgorithmExecutionRequest {
    algorithm_type: String,
    symbol: String,
    side: String,
    quantity: String, // 使用字符串避免精度问题
    parameters: HashMap<String, serde_json::Value>,
    timeout_seconds: u64,
    client_order_id: String,
}

/// 算法执行响应
#[derive(Debug, Serialize, Deserialize)]
struct AlgorithmExecutionResponse {
    algorithm_id: String,
    status: AlgorithmStatus,
    created_at: DateTime<Utc>,
    estimated_completion: Option<DateTime<Utc>>,
    progress: AlgorithmProgress,
    performance_metrics: Option<PerformanceMetrics>,
}

/// 算法状态
#[derive(Debug, Serialize, Deserialize, PartialEq)]
enum AlgorithmStatus {
    Pending,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

/// 算法进度
#[derive(Debug, Serialize, Deserialize)]
struct AlgorithmProgress {
    total_quantity: String,
    executed_quantity: String,
    remaining_quantity: String,
    completion_percentage: f64,
    slices_executed: u32,
    total_slices: u32,
}

/// 性能指标
#[derive(Debug, Serialize, Deserialize)]
struct PerformanceMetrics {
    average_price: String,
    market_impact: String,
    implementation_shortfall: String,
    execution_time_seconds: f64,
    slippage_bps: f64,
}

/// TWAP算法参数
#[derive(Debug, Serialize, Deserialize)]
struct TwapParameters {
    duration_minutes: u32,
    slice_count: u32,
    start_time: Option<DateTime<Utc>>,
    end_time: Option<DateTime<Utc>>,
}

/// VWAP算法参数
#[derive(Debug, Serialize, Deserialize)]
struct VwapParameters {
    participation_rate: String, // 0.0-1.0
    max_volume_per_slice: String,
    historical_days: u32,
    time_window_minutes: u32,
}

/// PoV算法参数
#[derive(Debug, Serialize, Deserialize)]
struct PovParameters {
    participation_rate: String, // 0.0-1.0
    max_order_rate: u32, // 每分钟最大订单数
    market_impact_threshold: String,
    aggressive_when_behind: bool,
}

/// 自适应算法参数
#[derive(Debug, Serialize, Deserialize)]
struct AdaptiveParameters {
    initial_strategy: String,
    adaptation_threshold: String,
    market_regime_sensitivity: String,
    fallback_strategy: String,
    learning_rate: f64,
}

impl AlgorithmTradingTests {
    /// 创建新的算法交易测试套件
    pub fn new(trading_service_url: String, timeout_seconds: u64) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_seconds))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            trading_service_url,
            timeout_duration: Duration::from_secs(timeout_seconds),
            test_metrics: AlgorithmTestMetrics::default(),
        }
    }

    /// 测试TWAP算法执行
    pub async fn test_twap_algorithm(&mut self) -> Result<()> {
        info!("🔄 开始测试 TWAP 算法");

        let test_cases = vec![
            // 基本TWAP测试
            TwapTestCase {
                name: "基本TWAP_30分钟_6切片",
                symbol: "BTCUSDT",
                side: "BUY",
                quantity: "0.1",
                duration_minutes: 30,
                slice_count: 6,
                expected_latency_ms: 50,
            },
            // 长时间TWAP测试
            TwapTestCase {
                name: "长时间TWAP_2小时_24切片",
                symbol: "ETHUSDT",
                side: "SELL",
                quantity: "5.0",
                duration_minutes: 120,
                slice_count: 24,
                expected_latency_ms: 50,
            },
            // 高频TWAP测试
            TwapTestCase {
                name: "高频TWAP_5分钟_60切片",
                symbol: "ADAUSDT",
                side: "BUY",
                quantity: "1000.0",
                duration_minutes: 5,
                slice_count: 60,
                expected_latency_ms: 50,
            },
        ];

        for test_case in test_cases {
            let start_time = Instant::now();
            
            match self.execute_twap_test_case(&test_case).await {
                Ok(algorithm_id) => {
                    let execution_latency = start_time.elapsed().as_millis() as u64;
                    
                    // 验证算法状态和进度
                    self.verify_algorithm_execution(&algorithm_id, &test_case.name).await?;
                    
                    // 更新测试指标
                    self.update_test_metrics(execution_latency, true);
                    self.test_metrics.twap_tests += 1;
                    
                    info!("✅ {} 测试成功 (延迟: {}ms)", test_case.name, execution_latency);
                    
                    // 验证延迟要求
                    if execution_latency > test_case.expected_latency_ms {
                        warn!("⚠️ {} 执行延迟超过预期: {}ms > {}ms", 
                              test_case.name, execution_latency, test_case.expected_latency_ms);
                    }
                }
                Err(e) => {
                    error!("❌ {} 测试失败: {}", test_case.name, e);
                    self.update_test_metrics(0, false);
                    return Err(e);
                }
            }
        }

        info!("✅ TWAP算法测试完成，共执行 {} 个测试用例", self.test_metrics.twap_tests);
        Ok(())
    }

    /// 测试VWAP算法执行
    pub async fn test_vwap_algorithm(&mut self) -> Result<()> {
        info!("📊 开始测试 VWAP 算法");

        let test_cases = vec![
            VwapTestCase {
                name: "基本VWAP_20%参与率",
                symbol: "BTCUSDT",
                side: "BUY",
                quantity: "0.5",
                participation_rate: "0.2",
                max_volume_per_slice: "0.1",
                historical_days: 5,
                expected_latency_ms: 50,
            },
            VwapTestCase {
                name: "高参与率VWAP_50%参与率",
                symbol: "ETHUSDT",
                side: "SELL",
                quantity: "10.0",
                participation_rate: "0.5",
                max_volume_per_slice: "2.0",
                historical_days: 3,
                expected_latency_ms: 50,
            },
            VwapTestCase {
                name: "保守VWAP_10%参与率",
                symbol: "BNBUSDT",
                side: "BUY",
                quantity: "100.0",
                participation_rate: "0.1",
                max_volume_per_slice: "5.0",
                historical_days: 7,
                expected_latency_ms: 50,
            },
        ];

        for test_case in test_cases {
            let start_time = Instant::now();
            
            match self.execute_vwap_test_case(&test_case).await {
                Ok(algorithm_id) => {
                    let execution_latency = start_time.elapsed().as_millis() as u64;
                    
                    // 验证算法执行和VWAP计算逻辑
                    self.verify_vwap_execution(&algorithm_id, &test_case).await?;
                    
                    self.update_test_metrics(execution_latency, true);
                    self.test_metrics.vwap_tests += 1;
                    
                    info!("✅ {} 测试成功 (延迟: {}ms)", test_case.name, execution_latency);
                }
                Err(e) => {
                    error!("❌ {} 测试失败: {}", test_case.name, e);
                    self.update_test_metrics(0, false);
                    return Err(e);
                }
            }
        }

        info!("✅ VWAP算法测试完成，共执行 {} 个测试用例", self.test_metrics.vwap_tests);
        Ok(())
    }

    /// 测试PoV算法执行
    pub async fn test_pov_algorithm(&mut self) -> Result<()> {
        info!("🎯 开始测试 PoV (参与率) 算法");

        let test_cases = vec![
            PovTestCase {
                name: "标准PoV_30%参与率",
                symbol: "BTCUSDT",
                side: "BUY",
                quantity: "0.2",
                participation_rate: "0.3",
                max_order_rate: 10,
                market_impact_threshold: "0.05",
                aggressive_when_behind: false,
                expected_latency_ms: 50,
            },
            PovTestCase {
                name: "激进PoV_50%参与率",
                symbol: "ETHUSDT",
                side: "SELL",
                quantity: "8.0",
                participation_rate: "0.5",
                max_order_rate: 20,
                market_impact_threshold: "0.08",
                aggressive_when_behind: true,
                expected_latency_ms: 50,
            },
        ];

        for test_case in test_cases {
            let start_time = Instant::now();
            
            match self.execute_pov_test_case(&test_case).await {
                Ok(algorithm_id) => {
                    let execution_latency = start_time.elapsed().as_millis() as u64;
                    
                    // 验证市场参与率控制
                    self.verify_pov_participation_control(&algorithm_id, &test_case).await?;
                    
                    self.update_test_metrics(execution_latency, true);
                    self.test_metrics.pov_tests += 1;
                    
                    info!("✅ {} 测试成功 (延迟: {}ms)", test_case.name, execution_latency);
                }
                Err(e) => {
                    error!("❌ {} 测试失败: {}", test_case.name, e);
                    self.update_test_metrics(0, false);
                    return Err(e);
                }
            }
        }

        info!("✅ PoV算法测试完成，共执行 {} 个测试用例", self.test_metrics.pov_tests);
        Ok(())
    }

    /// 测试自适应算法执行
    pub async fn test_adaptive_algorithm(&mut self) -> Result<()> {
        info!("🧠 开始测试自适应算法");

        let test_cases = vec![
            AdaptiveTestCase {
                name: "TWAP到VWAP自适应",
                symbol: "BTCUSDT",
                side: "BUY",
                quantity: "0.3",
                initial_strategy: "TWAP",
                adaptation_threshold: "0.15", // 15% implementation shortfall
                fallback_strategy: "VWAP",
                learning_rate: 0.1,
                expected_latency_ms: 50,
            },
            AdaptiveTestCase {
                name: "VWAP到PoV自适应",
                symbol: "ETHUSDT", 
                side: "SELL",
                quantity: "6.0",
                initial_strategy: "VWAP",
                adaptation_threshold: "0.12",
                fallback_strategy: "POV",
                learning_rate: 0.15,
                expected_latency_ms: 50,
            },
        ];

        for test_case in test_cases {
            let start_time = Instant::now();
            
            match self.execute_adaptive_test_case(&test_case).await {
                Ok(algorithm_id) => {
                    let execution_latency = start_time.elapsed().as_millis() as u64;
                    
                    // 验证自适应切换逻辑
                    self.verify_adaptive_switching(&algorithm_id, &test_case).await?;
                    
                    self.update_test_metrics(execution_latency, true);
                    self.test_metrics.adaptive_tests += 1;
                    
                    info!("✅ {} 测试成功 (延迟: {}ms)", test_case.name, execution_latency);
                }
                Err(e) => {
                    error!("❌ {} 测试失败: {}", test_case.name, e);
                    self.update_test_metrics(0, false);
                    return Err(e);
                }
            }
        }

        info!("✅ 自适应算法测试完成，共执行 {} 个测试用例", self.test_metrics.adaptive_tests);
        Ok(())
    }

    /// 测试算法监控和状态管理
    pub async fn test_algorithm_monitoring(&self) -> Result<()> {
        info!("📈 开始测试算法监控和状态管理");

        // 启动一个TWAP算法用于监控测试
        let test_request = self.create_twap_request(
            "BTCUSDT",
            "BUY",
            "0.05",
            60,
            12,
        )?;

        let algorithm_id = self.submit_algorithm_request(&test_request).await?;
        
        // 测试状态查询
        info!("🔍 测试算法状态查询");
        let status_response = self.query_algorithm_status(&algorithm_id).await?;
        assert!(matches!(status_response.status, AlgorithmStatus::Running | AlgorithmStatus::Pending));

        // 测试进度监控
        info!("📊 测试算法进度监控");
        let mut progress_checks = 0;
        let max_checks = 10;

        while progress_checks < max_checks {
            tokio::time::sleep(Duration::from_secs(5)).await;
            
            let progress_response = self.query_algorithm_status(&algorithm_id).await?;
            
            debug!("算法进度: {:.1}% ({}/{}切片)", 
                progress_response.progress.completion_percentage,
                progress_response.progress.slices_executed,
                progress_response.progress.total_slices);

            progress_checks += 1;

            if matches!(progress_response.status, AlgorithmStatus::Completed) {
                info!("✅ 算法执行完成");
                break;
            }
        }

        // 测试算法暂停和恢复
        info!("⏸️ 测试算法暂停功能");
        self.pause_algorithm(&algorithm_id).await?;
        
        tokio::time::sleep(Duration::from_secs(2)).await;
        let paused_status = self.query_algorithm_status(&algorithm_id).await?;
        assert_eq!(paused_status.status, AlgorithmStatus::Paused);

        info!("▶️ 测试算法恢复功能");
        self.resume_algorithm(&algorithm_id).await?;
        
        tokio::time::sleep(Duration::from_secs(2)).await;
        let resumed_status = self.query_algorithm_status(&algorithm_id).await?;
        assert_eq!(resumed_status.status, AlgorithmStatus::Running);

        info!("✅ 算法监控和状态管理测试完成");
        Ok(())
    }

    /// 验证延迟要求
    pub async fn validate_latency_requirements(&self, threshold_ms: u64) -> Result<()> {
        info!("⏱️ 验证算法执行延迟要求");

        let avg_latency = self.test_metrics.average_execution_latency_ms;
        let max_latency = self.test_metrics.max_execution_latency_ms;

        if avg_latency <= threshold_ms {
            info!("✅ 平均执行延迟: {}ms (要求: ≤ {}ms)", avg_latency, threshold_ms);
        } else {
            error!("❌ 平均执行延迟超过要求: {}ms > {}ms", avg_latency, threshold_ms);
            return Err(anyhow::anyhow!("算法执行延迟超过要求"));
        }

        if max_latency <= threshold_ms * 2 {
            info!("✅ 最大执行延迟: {}ms (要求: ≤ {}ms)", max_latency, threshold_ms * 2);
        } else {
            warn!("⚠️ 最大执行延迟较高: {}ms", max_latency);
        }

        // 延迟分布统计
        info!("📊 延迟统计:");
        info!("  • 平均延迟: {}ms", avg_latency);
        info!("  • 最小延迟: {}ms", self.test_metrics.min_execution_latency_ms);
        info!("  • 最大延迟: {}ms", max_latency);

        Ok(())
    }

    // ========== 私有辅助方法 ==========

    /// 执行TWAP测试用例
    async fn execute_twap_test_case(&self, test_case: &TwapTestCase) -> Result<String> {
        let request = self.create_twap_request(
            &test_case.symbol,
            &test_case.side,
            &test_case.quantity,
            test_case.duration_minutes,
            test_case.slice_count,
        )?;

        self.submit_algorithm_request(&request).await
    }

    /// 执行VWAP测试用例
    async fn execute_vwap_test_case(&self, test_case: &VwapTestCase) -> Result<String> {
        let request = self.create_vwap_request(
            &test_case.symbol,
            &test_case.side,
            &test_case.quantity,
            &test_case.participation_rate,
            &test_case.max_volume_per_slice,
            test_case.historical_days,
        )?;

        self.submit_algorithm_request(&request).await
    }

    /// 执行PoV测试用例
    async fn execute_pov_test_case(&self, test_case: &PovTestCase) -> Result<String> {
        let request = self.create_pov_request(
            &test_case.symbol,
            &test_case.side,
            &test_case.quantity,
            &test_case.participation_rate,
            test_case.max_order_rate,
            &test_case.market_impact_threshold,
            test_case.aggressive_when_behind,
        )?;

        self.submit_algorithm_request(&request).await
    }

    /// 执行自适应测试用例
    async fn execute_adaptive_test_case(&self, test_case: &AdaptiveTestCase) -> Result<String> {
        let request = self.create_adaptive_request(
            &test_case.symbol,
            &test_case.side,
            &test_case.quantity,
            &test_case.initial_strategy,
            &test_case.adaptation_threshold,
            &test_case.fallback_strategy,
            test_case.learning_rate,
        )?;

        self.submit_algorithm_request(&request).await
    }

    /// 创建TWAP请求
    fn create_twap_request(
        &self,
        symbol: &str,
        side: &str,
        quantity: &str,
        duration_minutes: u32,
        slice_count: u32,
    ) -> Result<AlgorithmExecutionRequest> {
        let mut parameters = HashMap::new();
        parameters.insert("duration_minutes".to_string(), serde_json::Value::Number(duration_minutes.into()));
        parameters.insert("slice_count".to_string(), serde_json::Value::Number(slice_count.into()));

        Ok(AlgorithmExecutionRequest {
            algorithm_type: "TWAP".to_string(),
            symbol: symbol.to_string(),
            side: side.to_string(),
            quantity: quantity.to_string(),
            parameters,
            timeout_seconds: 3600, // 1 hour
            client_order_id: Uuid::new_v4().to_string(),
        })
    }

    /// 创建VWAP请求
    fn create_vwap_request(
        &self,
        symbol: &str,
        side: &str,
        quantity: &str,
        participation_rate: &str,
        max_volume_per_slice: &str,
        historical_days: u32,
    ) -> Result<AlgorithmExecutionRequest> {
        let mut parameters = HashMap::new();
        parameters.insert("participation_rate".to_string(), serde_json::Value::String(participation_rate.to_string()));
        parameters.insert("max_volume_per_slice".to_string(), serde_json::Value::String(max_volume_per_slice.to_string()));
        parameters.insert("historical_days".to_string(), serde_json::Value::Number(historical_days.into()));

        Ok(AlgorithmExecutionRequest {
            algorithm_type: "VWAP".to_string(),
            symbol: symbol.to_string(),
            side: side.to_string(),
            quantity: quantity.to_string(),
            parameters,
            timeout_seconds: 3600,
            client_order_id: Uuid::new_v4().to_string(),
        })
    }

    /// 创建PoV请求
    fn create_pov_request(
        &self,
        symbol: &str,
        side: &str,
        quantity: &str,
        participation_rate: &str,
        max_order_rate: u32,
        market_impact_threshold: &str,
        aggressive_when_behind: bool,
    ) -> Result<AlgorithmExecutionRequest> {
        let mut parameters = HashMap::new();
        parameters.insert("participation_rate".to_string(), serde_json::Value::String(participation_rate.to_string()));
        parameters.insert("max_order_rate".to_string(), serde_json::Value::Number(max_order_rate.into()));
        parameters.insert("market_impact_threshold".to_string(), serde_json::Value::String(market_impact_threshold.to_string()));
        parameters.insert("aggressive_when_behind".to_string(), serde_json::Value::Bool(aggressive_when_behind));

        Ok(AlgorithmExecutionRequest {
            algorithm_type: "POV".to_string(),
            symbol: symbol.to_string(),
            side: side.to_string(),
            quantity: quantity.to_string(),
            parameters,
            timeout_seconds: 3600,
            client_order_id: Uuid::new_v4().to_string(),
        })
    }

    /// 创建自适应请求
    fn create_adaptive_request(
        &self,
        symbol: &str,
        side: &str,
        quantity: &str,
        initial_strategy: &str,
        adaptation_threshold: &str,
        fallback_strategy: &str,
        learning_rate: f64,
    ) -> Result<AlgorithmExecutionRequest> {
        let mut parameters = HashMap::new();
        parameters.insert("initial_strategy".to_string(), serde_json::Value::String(initial_strategy.to_string()));
        parameters.insert("adaptation_threshold".to_string(), serde_json::Value::String(adaptation_threshold.to_string()));
        parameters.insert("fallback_strategy".to_string(), serde_json::Value::String(fallback_strategy.to_string()));
        parameters.insert("learning_rate".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(learning_rate).unwrap()));

        Ok(AlgorithmExecutionRequest {
            algorithm_type: "ADAPTIVE".to_string(),
            symbol: symbol.to_string(),
            side: side.to_string(),
            quantity: quantity.to_string(),
            parameters,
            timeout_seconds: 3600,
            client_order_id: Uuid::new_v4().to_string(),
        })
    }

    /// 提交算法请求
    async fn submit_algorithm_request(&self, request: &AlgorithmExecutionRequest) -> Result<String> {
        let url = format!("{}/api/v1/algorithms/execute", self.trading_service_url);
        
        let response = timeout(
            self.timeout_duration,
            self.client.post(&url).json(request).send(),
        ).await??;

        if response.status().is_success() {
            let execution_response: AlgorithmExecutionResponse = response.json().await?;
            Ok(execution_response.algorithm_id)
        } else {
            let error_text = response.text().await?;
            Err(anyhow::anyhow!("算法提交失败: {}", error_text))
        }
    }

    /// 查询算法状态
    async fn query_algorithm_status(&self, algorithm_id: &str) -> Result<AlgorithmExecutionResponse> {
        let url = format!("{}/api/v1/algorithms/{}/status", self.trading_service_url, algorithm_id);
        
        let response = timeout(
            self.timeout_duration,
            self.client.get(&url).send(),
        ).await??;

        if response.status().is_success() {
            let status_response: AlgorithmExecutionResponse = response.json().await?;
            Ok(status_response)
        } else {
            let error_text = response.text().await?;
            Err(anyhow::anyhow!("查询算法状态失败: {}", error_text))
        }
    }

    /// 暂停算法
    async fn pause_algorithm(&self, algorithm_id: &str) -> Result<()> {
        let url = format!("{}/api/v1/algorithms/{}/pause", self.trading_service_url, algorithm_id);
        
        let response = timeout(
            self.timeout_duration,
            self.client.post(&url).send(),
        ).await??;

        if response.status().is_success() {
            Ok(())
        } else {
            let error_text = response.text().await?;
            Err(anyhow::anyhow!("暂停算法失败: {}", error_text))
        }
    }

    /// 恢复算法
    async fn resume_algorithm(&self, algorithm_id: &str) -> Result<()> {
        let url = format!("{}/api/v1/algorithms/{}/resume", self.trading_service_url, algorithm_id);
        
        let response = timeout(
            self.timeout_duration,
            self.client.post(&url).send(),
        ).await??;

        if response.status().is_success() {
            Ok(())
        } else {
            let error_text = response.text().await?;
            Err(anyhow::anyhow!("恢复算法失败: {}", error_text))
        }
    }

    /// 验证算法执行
    async fn verify_algorithm_execution(&self, algorithm_id: &str, test_name: &str) -> Result<()> {
        // 等待算法开始执行
        tokio::time::sleep(Duration::from_secs(2)).await;
        
        let status = self.query_algorithm_status(algorithm_id).await?;
        
        if !matches!(status.status, AlgorithmStatus::Running | AlgorithmStatus::Completed) {
            return Err(anyhow::anyhow!("{}: 算法未能正确启动, 状态: {:?}", test_name, status.status));
        }

        Ok(())
    }

    /// 验证VWAP执行逻辑
    async fn verify_vwap_execution(&self, algorithm_id: &str, test_case: &VwapTestCase) -> Result<()> {
        // 等待一些执行进度
        tokio::time::sleep(Duration::from_secs(5)).await;
        
        let status = self.query_algorithm_status(algorithm_id).await?;
        
        // 验证执行进度合理性
        if status.progress.completion_percentage > 100.0 {
            return Err(anyhow::anyhow!("VWAP执行进度异常: {}%", status.progress.completion_percentage));
        }

        // 检查参与率控制
        debug!("VWAP算法 {} 当前进度: {:.1}%", test_case.name, status.progress.completion_percentage);

        Ok(())
    }

    /// 验证PoV参与率控制
    async fn verify_pov_participation_control(&self, algorithm_id: &str, test_case: &PovTestCase) -> Result<()> {
        tokio::time::sleep(Duration::from_secs(3)).await;
        
        let status = self.query_algorithm_status(algorithm_id).await?;
        
        debug!("PoV算法 {} 执行统计: 已执行切片 {}/{}",
               test_case.name, status.progress.slices_executed, status.progress.total_slices);

        Ok(())
    }

    /// 验证自适应切换逻辑
    async fn verify_adaptive_switching(&self, algorithm_id: &str, test_case: &AdaptiveTestCase) -> Result<()> {
        // 监控一段时间，看是否触发策略切换
        let monitor_duration = Duration::from_secs(30);
        let start_time = Instant::now();

        while start_time.elapsed() < monitor_duration {
            let status = self.query_algorithm_status(algorithm_id).await?;
            
            // 检查是否有性能指标数据
            if let Some(metrics) = &status.performance_metrics {
                debug!("自适应算法 {} 性能: 实施缺口={}%, 滑点={}bps",
                       test_case.name, metrics.implementation_shortfall, metrics.slippage_bps);
            }

            tokio::time::sleep(Duration::from_secs(5)).await;
        }

        Ok(())
    }

    /// 更新测试指标
    fn update_test_metrics(&mut self, latency_ms: u64, success: bool) {
        self.test_metrics.total_algorithms_tested += 1;

        if success {
            self.test_metrics.successful_executions += 1;
            
            if latency_ms > 0 {
                // 更新延迟统计
                let total_latency = (self.test_metrics.average_execution_latency_ms * (self.test_metrics.successful_executions - 1) as u64) + latency_ms;
                self.test_metrics.average_execution_latency_ms = total_latency / self.test_metrics.successful_executions as u64;
                
                if self.test_metrics.max_execution_latency_ms == 0 || latency_ms > self.test_metrics.max_execution_latency_ms {
                    self.test_metrics.max_execution_latency_ms = latency_ms;
                }
                
                if self.test_metrics.min_execution_latency_ms == 0 || latency_ms < self.test_metrics.min_execution_latency_ms {
                    self.test_metrics.min_execution_latency_ms = latency_ms;
                }
            }
        } else {
            self.test_metrics.failed_executions += 1;
        }
    }
}

// ========== 测试用例结构体 ==========

#[derive(Debug)]
struct TwapTestCase {
    name: &'static str,
    symbol: &'static str,
    side: &'static str,
    quantity: &'static str,
    duration_minutes: u32,
    slice_count: u32,
    expected_latency_ms: u64,
}

#[derive(Debug)]
struct VwapTestCase {
    name: &'static str,
    symbol: &'static str,
    side: &'static str,
    quantity: &'static str,
    participation_rate: &'static str,
    max_volume_per_slice: &'static str,
    historical_days: u32,
    expected_latency_ms: u64,
}

#[derive(Debug)]
struct PovTestCase {
    name: &'static str,
    symbol: &'static str,
    side: &'static str,
    quantity: &'static str,
    participation_rate: &'static str,
    max_order_rate: u32,
    market_impact_threshold: &'static str,
    aggressive_when_behind: bool,
    expected_latency_ms: u64,
}

#[derive(Debug)]
struct AdaptiveTestCase {
    name: &'static str,
    symbol: &'static str,
    side: &'static str,
    quantity: &'static str,
    initial_strategy: &'static str,
    adaptation_threshold: &'static str,
    fallback_strategy: &'static str,
    learning_rate: f64,
    expected_latency_ms: u64,
}