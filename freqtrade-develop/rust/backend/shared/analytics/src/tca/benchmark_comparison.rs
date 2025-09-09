use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// 基准比较引擎
#[derive(Debug, Clone)]
pub struct BenchmarkEngine {
    benchmarks: HashMap<String, BenchmarkDefinition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkDefinition {
    pub name: String,
    pub benchmark_type: BenchmarkType,
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BenchmarkType {
    VWAP,
    TWAP,
    Close,
    Open,
    Arrival,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub benchmark_name: String,
    pub execution_price: f64,
    pub benchmark_price: f64,
    pub slippage_bps: f64,
    pub performance_ratio: f64,
}

impl BenchmarkEngine {
    pub fn new() -> Self {
        Self {
            benchmarks: HashMap::new(),
        }
    }

    pub fn add_benchmark(&mut self, name: String, definition: BenchmarkDefinition) {
        self.benchmarks.insert(name, definition);
    }

    pub fn compare_execution(
        &self,
        execution_price: f64,
        benchmark_name: &str,
        market_data: &HashMap<String, f64>,
    ) -> Result<BenchmarkResult> {
        let benchmark = self.benchmarks.get(benchmark_name)
            .ok_or_else(|| anyhow::anyhow!("Benchmark not found: {}", benchmark_name))?;

        let benchmark_price = self.calculate_benchmark_price(benchmark, market_data)?;
        let slippage_bps = ((execution_price - benchmark_price) / benchmark_price) * 10000.0;
        let performance_ratio = benchmark_price / execution_price;

        Ok(BenchmarkResult {
            benchmark_name: benchmark_name.to_string(),
            execution_price,
            benchmark_price,
            slippage_bps,
            performance_ratio,
        })
    }

    fn calculate_benchmark_price(
        &self,
        benchmark: &BenchmarkDefinition,
        market_data: &HashMap<String, f64>,
    ) -> Result<f64> {
        match &benchmark.benchmark_type {
            BenchmarkType::VWAP => {
                market_data.get("vwap")
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("VWAP data not available"))
            }
            BenchmarkType::TWAP => {
                market_data.get("twap")
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("TWAP data not available"))
            }
            BenchmarkType::Close => {
                market_data.get("close")
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("Close price not available"))
            }
            BenchmarkType::Open => {
                market_data.get("open")
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("Open price not available"))
            }
            BenchmarkType::Arrival => {
                market_data.get("arrival")
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("Arrival price not available"))
            }
            BenchmarkType::Custom(name) => {
                market_data.get(name)
                    .copied()
                    .ok_or_else(|| anyhow::anyhow!("Custom benchmark {} not available", name))
            }
        }
    }

    pub fn calculate_multiple_benchmarks(
        &self,
        execution_price: f64,
        market_data: &HashMap<String, f64>,
    ) -> Vec<BenchmarkResult> {
        let mut results = Vec::new();
        
        for (name, _) in &self.benchmarks {
            if let Ok(result) = self.compare_execution(execution_price, name, market_data) {
                results.push(result);
            }
        }
        
        results
    }
}

impl Default for BenchmarkEngine {
    fn default() -> Self {
        Self::new()
    }
}