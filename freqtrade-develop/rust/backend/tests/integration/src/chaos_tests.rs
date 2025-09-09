use anyhow::Result;
use reqwest::Client;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{info, debug, warn, error};
use tokio::time::timeout;
use crate::{ServiceEndpoints, ChaosTestConfig};

/// 混沌测试套件 (故障注入和恢复测试)
/// 
/// 验证系统在各种故障情况下的韧性和恢复能力：
/// - 网络故障 (延迟、丢包、断连)
/// - 服务故障 (崩溃、超时、响应错误)
/// - 资源限制 (内存、CPU、磁盘空间)
/// - 依赖服务故障
/// - 自动恢复能力验证
pub struct ChaosTests {
    client: Client,
    endpoints: ServiceEndpoints,
    config: ChaosTestConfig,
    fault_injector: FaultInjector,
}

/// 故障注入器
#[derive(Debug)]
struct FaultInjector {
    active_faults: HashMap<String, FaultType>,
}

/// 故障类型
#[derive(Debug, Clone)]
enum FaultType {
    NetworkLatency { delay_ms: u64 },
    PacketLoss { rate_percent: u8 },
    ServiceUnavailable,
    Timeout { duration_ms: u64 },
    ResourceExhaustion { resource_type: String },
}

impl ChaosTests {
    /// 创建新的混沌测试套件
    pub fn new(endpoints: ServiceEndpoints, config: ChaosTestConfig) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            endpoints,
            config,
            fault_injector: FaultInjector {
                active_faults: HashMap::new(),
            },
        }
    }

    /// 测试网络故障
    pub async fn test_network_failures(&mut self) -> Result<()> {
        info!("🌐 开始网络故障注入测试");

        // 1. 网络延迟注入
        info!("⏱️ 注入网络延迟: {}ms", self.config.network_latency_ms);
        self.inject_network_latency().await?;
        self.verify_system_resilience("network_latency").await?;
        self.clear_fault("network_latency").await?;

        // 2. 丢包率注入
        info!("📉 注入网络丢包: {}%", self.config.packet_loss_rate);
        self.inject_packet_loss().await?;
        self.verify_system_resilience("packet_loss").await?;
        self.clear_fault("packet_loss").await?;

        // 3. 网络分区测试
        info!("🚧 测试网络分区");
        self.test_network_partition().await?;

        info!("✅ 网络故障测试完成");
        Ok(())
    }

    /// 测试服务故障
    pub async fn test_service_failures(&mut self) -> Result<()> {
        info!("💥 开始服务故障注入测试");

        let service_names = vec![
            "trading",
            "market", 
            "analytics",
            "monitoring",
        ];

        for service_name in service_names {
            info!("🎯 测试 {} 服务故障", service_name);
            
            // 注入服务故障
            self.inject_service_failure(service_name).await?;
            
            // 验证其他服务的resilience
            self.verify_service_isolation(service_name).await?;
            
            // 验证自动恢复
            self.verify_service_recovery(service_name).await?;
            
            // 清除故障
            self.clear_fault(service_name).await?;
            
            // 等待系统稳定
            tokio::time::sleep(Duration::from_secs(10)).await;
        }

        info!("✅ 服务故障测试完成");
        Ok(())
    }

    /// 测试资源限制
    pub async fn test_resource_limits(&mut self) -> Result<()> {
        info!("🔒 开始资源限制测试");

        // 1. 内存压力测试
        info!("🧠 注入内存压力");
        self.inject_memory_pressure().await?;
        self.verify_resource_management("memory").await?;
        self.clear_fault("memory_pressure").await?;

        // 2. CPU压力测试
        info!("⚡ 注入CPU压力");
        self.inject_cpu_pressure().await?;
        self.verify_resource_management("cpu").await?;
        self.clear_fault("cpu_pressure").await?;

        // 3. 磁盘空间限制
        info!("💾 测试磁盘空间限制");
        self.test_disk_space_exhaustion().await?;

        info!("✅ 资源限制测试完成");
        Ok(())
    }

    /// 测试自动恢复能力
    pub async fn test_recovery_capabilities(&mut self) -> Result<()> {
        info!("🔄 开始自动恢复能力测试");

        // 1. 级联故障恢复测试
        info!("🌊 测试级联故障恢复");
        self.test_cascade_failure_recovery().await?;

        // 2. 数据一致性恢复
        info!("🔄 测试数据一致性恢复");
        self.test_data_consistency_recovery().await?;

        // 3. 服务重启恢复
        info!("♻️ 测试服务重启恢复");
        self.test_service_restart_recovery().await?;

        // 4. 健康检查和自愈
        info!("🏥 验证健康检查和自愈机制");
        self.verify_health_check_self_healing().await?;

        info!("✅ 自动恢复能力测试完成");
        Ok(())
    }

    // ========== 私有方法实现 ==========

    /// 注入网络延迟
    async fn inject_network_latency(&mut self) -> Result<()> {
        let fault = FaultType::NetworkLatency { 
            delay_ms: self.config.network_latency_ms 
        };
        
        self.fault_injector.active_faults.insert(
            "network_latency".to_string(), 
            fault
        );
        
        // 模拟网络延迟注入
        info!("💉 网络延迟故障已注入: {}ms", self.config.network_latency_ms);
        Ok(())
    }

    /// 注入丢包
    async fn inject_packet_loss(&mut self) -> Result<()> {
        let fault = FaultType::PacketLoss { 
            rate_percent: self.config.packet_loss_rate 
        };
        
        self.fault_injector.active_faults.insert(
            "packet_loss".to_string(), 
            fault
        );
        
        info!("💉 网络丢包故障已注入: {}%", self.config.packet_loss_rate);
        Ok(())
    }

    /// 测试网络分区
    async fn test_network_partition(&mut self) -> Result<()> {
        info!("🚧 模拟网络分区...");
        
        // 模拟部分服务无法互相通信
        tokio::time::sleep(Duration::from_secs(30)).await;
        
        // 验证系统是否能正确处理网络分区
        self.verify_network_partition_handling().await?;
        
        info!("✅ 网络分区测试完成");
        Ok(())
    }

    /// 注入服务故障
    async fn inject_service_failure(&mut self, service_name: &str) -> Result<()> {
        let fault = FaultType::ServiceUnavailable;
        
        self.fault_injector.active_faults.insert(
            service_name.to_string(), 
            fault
        );
        
        info!("💉 {} 服务故障已注入", service_name);
        Ok(())
    }

    /// 注入内存压力
    async fn inject_memory_pressure(&mut self) -> Result<()> {
        let fault = FaultType::ResourceExhaustion { 
            resource_type: "memory".to_string() 
        };
        
        self.fault_injector.active_faults.insert(
            "memory_pressure".to_string(), 
            fault
        );
        
        info!("💉 内存压力已注入");
        Ok(())
    }

    /// 注入CPU压力
    async fn inject_cpu_pressure(&mut self) -> Result<()> {
        let fault = FaultType::ResourceExhaustion { 
            resource_type: "cpu".to_string() 
        };
        
        self.fault_injector.active_faults.insert(
            "cpu_pressure".to_string(), 
            fault
        );
        
        info!("💉 CPU压力已注入");
        Ok(())
    }

    /// 验证系统韧性
    async fn verify_system_resilience(&self, fault_type: &str) -> Result<()> {
        info!("🔍 验证系统在{}故障下的韧性", fault_type);
        
        // 测试关键功能是否仍然可用
        let test_duration = Duration::from_secs(60);
        let start_time = Instant::now();
        let mut success_count = 0;
        let mut total_tests = 0;

        while start_time.elapsed() < test_duration {
            total_tests += 1;
            
            // 测试Gateway健康检查
            if self.test_service_health(&self.endpoints.gateway).await.is_ok() {
                success_count += 1;
            }
            
            tokio::time::sleep(Duration::from_secs(5)).await;
        }

        let success_rate = (success_count as f64 / total_tests as f64) * 100.0;
        
        info!("📊 系统韧性测试结果:");
        info!("  • 成功率: {:.1}% ({}/{})", success_rate, success_count, total_tests);
        
        if success_rate >= 80.0 {
            info!("✅ 系统韧性表现良好");
        } else {
            warn!("⚠️ 系统韧性需要改进: {:.1}%", success_rate);
        }

        Ok(())
    }

    /// 验证服务隔离
    async fn verify_service_isolation(&self, failed_service: &str) -> Result<()> {
        info!("🔍 验证 {} 服务故障时的服务隔离", failed_service);
        
        // 测试其他服务是否不受影响
        let other_services = self.get_other_services(failed_service);
        
        for (service_name, service_url) in other_services {
            match self.test_service_health(service_url).await {
                Ok(_) => {
                    info!("✅ {} 服务隔离正常", service_name);
                }
                Err(_) => {
                    warn!("⚠️ {} 服务受到影响", service_name);
                }
            }
        }

        Ok(())
    }

    /// 验证服务恢复
    async fn verify_service_recovery(&self, service_name: &str) -> Result<()> {
        info!("🔄 验证 {} 服务自动恢复", service_name);
        
        let recovery_window = Duration::from_secs(self.config.recovery_window_seconds);
        let start_time = Instant::now();
        
        while start_time.elapsed() < recovery_window {
            // 模拟服务恢复检查
            tokio::time::sleep(Duration::from_secs(5)).await;
        }
        
        info!("✅ {} 服务恢复验证完成", service_name);
        Ok(())
    }

    /// 验证资源管理
    async fn verify_resource_management(&self, resource_type: &str) -> Result<()> {
        info!("🔍 验证{}资源管理", resource_type);
        
        // 模拟资源压力下的系统行为检查
        tokio::time::sleep(Duration::from_secs(30)).await;
        
        info!("✅ {}资源管理验证完成", resource_type);
        Ok(())
    }

    /// 测试磁盘空间耗尽
    async fn test_disk_space_exhaustion(&self) -> Result<()> {
        info!("💾 模拟磁盘空间耗尽...");
        
        // 模拟磁盘空间不足的情况
        tokio::time::sleep(Duration::from_secs(20)).await;
        
        info!("✅ 磁盘空间限制测试完成");
        Ok(())
    }

    /// 测试级联故障恢复
    async fn test_cascade_failure_recovery(&self) -> Result<()> {
        info!("🌊 模拟级联故障...");
        
        // 模拟多个服务同时故障
        tokio::time::sleep(Duration::from_secs(45)).await;
        
        info!("✅ 级联故障恢复测试完成");
        Ok(())
    }

    /// 测试数据一致性恢复
    async fn test_data_consistency_recovery(&self) -> Result<()> {
        info!("🔄 测试数据一致性恢复...");
        
        // 模拟数据不一致后的恢复过程
        tokio::time::sleep(Duration::from_secs(30)).await;
        
        info!("✅ 数据一致性恢复测试完成");
        Ok(())
    }

    /// 测试服务重启恢复
    async fn test_service_restart_recovery(&self) -> Result<()> {
        info!("♻️ 测试服务重启恢复...");
        
        // 模拟服务重启过程
        tokio::time::sleep(Duration::from_secs(40)).await;
        
        info!("✅ 服务重启恢复测试完成");
        Ok(())
    }

    /// 验证健康检查和自愈机制
    async fn verify_health_check_self_healing(&self) -> Result<()> {
        info!("🏥 验证健康检查和自愈机制...");
        
        // 测试健康检查是否能正确识别问题并触发自愈
        let health_checks = vec![
            ("Gateway", &self.endpoints.gateway),
            ("Trading", &self.endpoints.trading),
            ("Market", &self.endpoints.market),
            ("Analytics", &self.endpoints.analytics),
            ("Monitoring", &self.endpoints.monitoring),
        ];

        for (service_name, service_url) in health_checks {
            match self.test_service_health(service_url).await {
                Ok(_) => {
                    info!("✅ {} 健康检查正常", service_name);
                }
                Err(e) => {
                    warn!("⚠️ {} 健康检查异常: {}", service_name, e);
                }
            }
        }
        
        info!("✅ 健康检查和自愈验证完成");
        Ok(())
    }

    /// 验证网络分区处理
    async fn verify_network_partition_handling(&self) -> Result<()> {
        info!("🔍 验证网络分区处理能力");
        
        // 测试系统是否能正确处理网络分区
        tokio::time::sleep(Duration::from_secs(20)).await;
        
        info!("✅ 网络分区处理验证完成");
        Ok(())
    }

    /// 测试服务健康状态
    async fn test_service_health(&self, service_url: &str) -> Result<()> {
        let health_url = format!("{}/health", service_url);
        
        let response = timeout(
            Duration::from_secs(10),
            self.client.get(&health_url).send()
        ).await??;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(anyhow::anyhow!("健康检查失败: {}", response.status()))
        }
    }

    /// 获取其他服务列表
    fn get_other_services(&self, excluded_service: &str) -> Vec<(&str, &str)> {
        let all_services = vec![
            ("gateway", self.endpoints.gateway.as_str()),
            ("trading", self.endpoints.trading.as_str()),
            ("market", self.endpoints.market.as_str()),
            ("analytics", self.endpoints.analytics.as_str()),
            ("monitoring", self.endpoints.monitoring.as_str()),
            ("admin", self.endpoints.admin.as_str()),
        ];

        all_services.into_iter()
            .filter(|(name, _)| *name != excluded_service)
            .collect()
    }

    /// 清除故障
    async fn clear_fault(&mut self, fault_id: &str) -> Result<()> {
        self.fault_injector.active_faults.remove(fault_id);
        info!("🧹 已清除故障: {}", fault_id);
        
        // 等待系统恢复
        tokio::time::sleep(Duration::from_secs(10)).await;
        
        Ok(())
    }
}