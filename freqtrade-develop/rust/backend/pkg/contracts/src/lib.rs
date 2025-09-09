// Sprint 8: Generated gRPC service definitions
// 包含所有proto生成的代码
tonic::include_proto!("trading.v1");

// Include events protocol definitions
tonic::include_proto!("events.v1");

// 为了方便使用，重新导出各个服务的类型
pub mod execution {
    pub use crate::{
        execution_service_client::ExecutionServiceClient,
        execution_service_server::{ExecutionService, ExecutionServiceServer},
        PlaceOrderRequest, PlaceOrderResponse, 
        OrderRequest, OrderStatus, ExecutionReport,
        CancelOrderRequest, CancelResponse,
        PositionQuery, PositionInfo,
        AccountQuery, AccountBalance,
        BatchOrderRequest, BatchOrderResponse,
        ExecutionStats, HealthCheckRequest, HealthCheckResponse
    };
}

pub mod backtest {
    pub use crate::{
        backtest_service_client::BacktestServiceClient,
        backtest_service_server::{BacktestService, BacktestServiceServer},
        BacktestRequest, BacktestResult
    };
}

pub mod factor {
    pub use crate::{
        factor_service_client::FactorServiceClient,
        factor_service_server::{FactorService, FactorServiceServer},
        FactorBatchRequest, FactorBatchResult
    };
}

pub mod risk {
    pub use crate::{
        risk_service_client::RiskServiceClient,
        risk_service_server::{RiskService, RiskServiceServer},
        PretradeRequest, PretradeDecision
    };
}

pub mod markets {
    pub use crate::{
        market_data_service_client::MarketDataServiceClient,
        market_data_service_server::{MarketDataService, MarketDataServiceServer},
        data_management_service_client::DataManagementServiceClient,
        data_management_service_server::{DataManagementService, DataManagementServiceServer},
        SymbolSearchRequest, CandleRequest, CandleResponse,
        OrderBookRequest, TradeRequest, TradeResponse,
        StatsRequest, StatsResponse,
        DataImportRequest, DataBackfillRequest, DataGapResponse, JobResponse
    };
}

pub mod common {
    pub use crate::{
        JobAck, JobId, ProgressEvent, ErrorDetails,
        PageRequest, PageResponse, TimeRange, Status
    };
}

pub mod events {
    pub use crate::{
        event_service_client::EventServiceClient,
        event_service_server::{EventService, EventServiceServer},
        BaseEvent, MarketDataEvent, TradingEvent, SystemEvent,
        PriceUpdateEvent, OrderBookUpdateEvent, KlineUpdateEvent, TradeUpdateEvent,
        PriceLevel, StrategySignalEvent, OrderEvent, PositionUpdateEvent, RiskEvent,
        ServiceHealthEvent, PerformanceMetricsEvent, UserSessionEvent, ConfigChangeEvent,
        PublishEventRequest, PublishEventResponse, SubscribeEventsRequest,
        GetEventHistoryRequest, GetEventHistoryResponse,
        SignalType, OrderSide, EventOrderStatus, RiskType, RiskSeverity,
        HealthStatus, SessionAction
    };
}

// Legacy support
pub mod generated {
    tonic::include_proto!("trading");
}

pub mod types;
pub mod validation;
pub mod client;
pub mod unified_contract_manager;

// Export common types for convenience
// pub use common::*; // 暂时注释掉，避免未使用警告
pub use types::*;
pub use client::{TradingClient, ClientConfig, ServiceHealthStatus};
pub use unified_contract_manager::{UnifiedContractManager, UnifiedContract, ContractType, ContractStatus, ContractManagerConfig};

use anyhow::Result;

/// AG3合约验证接口
pub trait ContractValidator {
    fn validate(&self) -> Result<()>;
}

/// 统一错误类型
#[derive(Debug, thiserror::Error)]
pub enum ContractError {
    #[error("Validation error: {0}")]
    Validation(String),
    
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    #[error("gRPC error: {0}")]
    Grpc(Box<tonic::Status>),
    
    #[error("Other error: {0}")]
    Other(Box<anyhow::Error>),
}

impl From<tonic::Status> for ContractError {
    fn from(status: tonic::Status) -> Self {
        Self::Grpc(Box::new(status))
    }
}

impl From<anyhow::Error> for ContractError {
    fn from(error: anyhow::Error) -> Self {
        Self::Other(Box::new(error))
    }
}

pub type ContractResult<T> = Result<T, ContractError>;

#[cfg(test)]
mod tests {
    use crate::unified_contract_manager::ContractStore;

    #[test]
    fn test_contract_compilation() {
        // Test that contracts can be instantiated
        let _store = ContractStore::new();
    }
}
