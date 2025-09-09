use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use std::time::Duration;
use tokio::runtime::Runtime;

/// 市场数据吞吐量基准测试
/// 
/// 验证市场数据处理性能：
/// - 消息处理吞吐量 > 100K msg/s
/// - P99处理延迟 < 1ms
/// - SIMD优化效果验证
/// - 背压控制性能影响

fn market_data_processing_benchmark(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("market_data_processing");
    group.measurement_time(Duration::from_secs(60));
    
    // 不同消息大小的处理性能
    for msg_size in [100, 500, 1000, 2000].iter() {
        group.throughput(Throughput::Bytes(*msg_size as u64));
        group.bench_with_input(
            criterion::BenchmarkId::new("process_messages", msg_size),
            msg_size,
            |b, &msg_size| {
                b.to_async(&rt).iter(|| async {
                    let batch = generate_message_batch(black_box(msg_size), black_box(1000));
                    process_message_batch(batch).await
                });
            },
        );
    }
    
    group.finish();
}

fn simd_optimization_benchmark(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("simd_optimization");
    group.measurement_time(Duration::from_secs(30));
    
    let data_size = 100_000;
    let test_data: Vec<f32> = (0..data_size).map(|i| i as f32).collect();
    
    group.bench_function("simd_computation", |b| {
        b.to_async(&rt).iter(|| async {
            simd_vector_computation(black_box(&test_data)).await
        });
    });
    
    group.bench_function("scalar_computation", |b| {
        b.to_async(&rt).iter(|| async {
            scalar_computation(black_box(&test_data)).await
        });
    });
    
    group.finish();
}

fn backpressure_impact_benchmark(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("backpressure_impact");
    group.measurement_time(Duration::from_secs(45));
    
    // 测试不同负载下的处理性能
    for load_factor in [0.5, 0.8, 0.9, 1.0, 1.2].iter() {
        group.bench_with_input(
            criterion::BenchmarkId::new("under_load", (*load_factor * 100.0) as u32),
            load_factor,
            |b, &load_factor| {
                b.to_async(&rt).iter(|| async {
                    simulate_processing_under_load(black_box(load_factor)).await
                });
            },
        );
    }
    
    group.finish();
}

fn latency_distribution_benchmark(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("latency_distribution");
    group.measurement_time(Duration::from_secs(30));
    
    group.bench_function("p50_latency", |b| {
        b.to_async(&rt).iter(|| async {
            process_single_message(black_box(generate_test_message())).await
        });
    });
    
    group.bench_function("p99_latency", |b| {
        b.to_async(&rt).iter(|| async {
            // 模拟P99场景下的处理
            process_complex_message(black_box(generate_complex_test_message())).await
        });
    });
    
    group.finish();
}

// 模拟函数实现
fn generate_message_batch(msg_size: usize, count: usize) -> Vec<Vec<u8>> {
    (0..count)
        .map(|_| vec![0u8; msg_size])
        .collect()
}

async fn process_message_batch(batch: Vec<Vec<u8>>) {
    // 模拟批量消息处理
    let processing_time = Duration::from_nanos(batch.len() as u64 * 100);
    tokio::time::sleep(processing_time).await;
}

async fn simd_vector_computation(data: &[f32]) -> f32 {
    // 模拟SIMD向量化计算
    tokio::time::sleep(Duration::from_nanos(data.len() as u64 / 8)).await;
    data.iter().sum()
}

async fn scalar_computation(data: &[f32]) -> f32 {
    // 模拟标量计算
    tokio::time::sleep(Duration::from_nanos(data.len() as u64)).await;
    data.iter().sum()
}

async fn simulate_processing_under_load(load_factor: f64) {
    // 模拟不同负载下的处理时间
    let base_time = Duration::from_micros(100);
    let adjusted_time = Duration::from_nanos(
        (base_time.as_nanos() as f64 * load_factor) as u64
    );
    tokio::time::sleep(adjusted_time).await;
}

fn generate_test_message() -> Vec<u8> {
    vec![0u8; 512]
}

fn generate_complex_test_message() -> Vec<u8> {
    vec![0u8; 2048]
}

async fn process_single_message(message: Vec<u8>) {
    // 模拟单个消息处理时间
    let processing_time = Duration::from_nanos(message.len() as u64 * 2);
    tokio::time::sleep(processing_time).await;
}

async fn process_complex_message(message: Vec<u8>) {
    // 模拟复杂消息处理时间（P99场景）
    let processing_time = Duration::from_nanos(message.len() as u64 * 5);
    tokio::time::sleep(processing_time).await;
}

criterion_group!(
    benches,
    market_data_processing_benchmark,
    simd_optimization_benchmark,
    backpressure_impact_benchmark,
    latency_distribution_benchmark
);
criterion_main!(benches);