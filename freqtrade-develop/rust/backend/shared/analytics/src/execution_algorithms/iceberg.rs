use super::*;
use anyhow::Result;
use std::collections::HashMap;

/// Iceberg 执行算法
#[derive(Debug)]
pub struct IcebergAlgorithm {
    statistics: AlgorithmStatistics,
}

impl IcebergAlgorithm {
    pub fn new() -> Result<Self> {
        Ok(Self {
            statistics: AlgorithmStatistics {
                algorithm_name: "ICEBERG".to_string(),
                ..Default::default()
            },
        })
    }
}

impl ExecutionAlgorithm for IcebergAlgorithm {
    fn name(&self) -> &str {
        "ICEBERG"
    }

    fn calculate_child_orders(
        &self,
        parent_order: &ParentOrder,
        _market_conditions: &MarketConditions,
        execution_params: &ExecutionParams,
    ) -> Result<Vec<ChildOrder>> {
        let display_size = parent_order.total_quantity * execution_params.iceberg_size_ratio;
        
        let child_order = ChildOrder {
            id: format!("{}_iceberg_0", parent_order.id),
            parent_id: parent_order.id.clone(),
            sequence_number: 0,
            quantity: parent_order.total_quantity,
            price: parent_order.limit_price,
            venue: "PRIMARY".to_string(),
            order_type: OrderType::IcebergLimit,
            time_in_force: TimeInForce::GoodTillCancel,
            scheduled_time: parent_order.created_at,
            execution_window: parent_order.time_horizon,
            is_hidden: true,
            display_quantity: Some(display_size),
            post_only: true,
            reduce_only: false,
        };

        Ok(vec![child_order])
    }

    fn adapt_parameters(&mut self, _execution_state: &ExecutionState, _market_update: &MarketUpdate) -> Result<()> {
        Ok(())
    }

    fn get_statistics(&self) -> AlgorithmStatistics {
        self.statistics.clone()
    }

    fn validate_parameters(&self, _params: &HashMap<String, f64>) -> Result<()> {
        Ok(())
    }
}