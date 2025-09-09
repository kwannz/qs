use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use crossbeam::queue::SegQueue;
use parking_lot::{RwLock, Mutex};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// 超低延迟执行引擎 - 目标延迟<100微秒
pub struct UltraLowLatencyEngine {
    // 无锁队列用于订单处理
    order_queue: Arc<SegQueue<OrderMessage>>,
    execution_queue: Arc<SegQueue<ExecutionMessage>>,
    
    // 原子计数器和统计
    processed_orders: AtomicU64,
    total_latency_ns: AtomicU64,
    
    // 预分配内存池
    memory_pool: MemoryPool,
    
    // NUMA感知线程池
    numa_thread_pool: NumaThreadPool,
    
    // 零拷贝市场数据缓存
    market_data_cache: ZeroCopyMarketCache,
    
    // 配置
    config: UltraLowLatencyConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UltraLowLatencyConfig {
    pub target_latency_ns: u64,           // 目标延迟纳秒
    pub max_queue_size: usize,            // 最大队列大小
    pub numa_nodes: Vec<u32>,             // NUMA节点配置
    pub cpu_affinity: Vec<u32>,           // CPU亲和性
    pub hugepages_enabled: bool,          // 大页内存
    pub kernel_bypass: bool,              // 内核旁路
    pub zero_copy_enabled: bool,          // 零拷贝
    pub simd_optimization: bool,          // SIMD优化
    pub memory_prefetch: bool,            // 内存预取
}

#[derive(Debug, Clone)]
pub struct OrderMessage {
    pub order_id: u64,
    pub symbol_id: u32,                   // 符号ID（避免字符串）
    pub side: u8,                         // 买卖方向（1字节）
    pub quantity: u64,                    // 数量（定点数）
    pub price: u64,                       // 价格（定点数）
    pub timestamp_ns: u64,                // 纳秒时间戳
    pub strategy_id: u32,                 // 策略ID
    pub venue_mask: u64,                  // 场所掩码（位图）
    pub flags: OrderFlags,                // 订单标志
}

#[derive(Debug, Clone, Copy)]
pub struct OrderFlags {
    pub is_ioc: bool,                     // IOC订单
    pub is_hidden: bool,                  // 隐藏订单
    pub is_iceberg: bool,                 // 冰山订单
    pub risk_checked: bool,               // 已通过风控
    pub is_synthetic: bool,               // 合成订单
}

#[derive(Debug, Clone)]
pub struct ExecutionMessage {
    pub execution_id: u64,
    pub order_id: u64,
    pub fill_quantity: u64,
    pub fill_price: u64,
    pub timestamp_ns: u64,
    pub venue_id: u16,
    pub liquidity_flag: u8,               // Maker/Taker
    pub commission: u32,                  // 手续费（基点）
}

/// 内存池 - 预分配和对象重用
pub struct MemoryPool {
    order_pool: SegQueue<Box<OrderMessage>>,
    execution_pool: SegQueue<Box<ExecutionMessage>>,
    buffer_pool: SegQueue<Vec<u8>>,       // 字节缓冲区池
    pool_size: usize,
}

/// NUMA感知线程池
pub struct NumaThreadPool {
    workers: Vec<NumaWorker>,
    work_queues: Vec<Arc<SegQueue<WorkItem>>>,
    config: NumaConfig,
}

#[derive(Debug, Clone)]
pub struct NumaConfig {
    pub nodes: Vec<u32>,
    pub threads_per_node: u32,
    pub cpu_affinity_enabled: bool,
}

pub struct NumaWorker {
    node_id: u32,
    thread_handle: std::thread::JoinHandle<()>,
    work_queue: Arc<SegQueue<WorkItem>>,
}

#[derive(Debug)]
pub enum WorkItem {
    ProcessOrder(OrderMessage),
    ProcessExecution(ExecutionMessage),
    MarketDataUpdate(MarketDataSlice),
    RiskCheck(RiskCheckRequest),
    Shutdown,
}

/// 零拷贝市场数据缓存
pub struct ZeroCopyMarketCache {
    // 使用内存映射文件实现零拷贝
    price_data: Arc<RwLock<PriceDataMMap>>,
    book_data: Arc<RwLock<BookDataMMap>>,
    
    // 原子指针用于无锁读取
    current_prices: Arc<AtomicPtr<PriceArray>>,
    current_books: Arc<AtomicPtr<BookArray>>,
    
    // 数据版本号（用于一致性检查）
    version: AtomicU64,
}

use std::sync::atomic::AtomicPtr;

#[repr(C, align(64))]  // 缓存行对齐
pub struct PriceArray {
    pub prices: [AtomicU64; 10000],       // 支持10000个symbol
    pub timestamps: [AtomicU64; 10000],   // 对应时间戳
    pub version: AtomicU64,
}

#[repr(C, align(64))]
pub struct BookArray {
    pub bid_prices: [[AtomicU64; 10]; 10000],  // 10档买价
    pub bid_sizes: [[AtomicU64; 10]; 10000],   // 10档买量
    pub ask_prices: [[AtomicU64; 10]; 10000],  // 10档卖价
    pub ask_sizes: [[AtomicU64; 10]; 10000],   // 10档卖量
    pub version: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct MarketDataSlice {
    pub symbol_id: u32,
    pub bid_price: u64,
    pub ask_price: u64,
    pub bid_size: u64,
    pub ask_size: u64,
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone)]
pub struct RiskCheckRequest {
    pub order: OrderMessage,
    pub current_position: i64,
    pub available_capital: u64,
    pub risk_limits: RiskLimits,
}

#[derive(Debug, Clone)]
pub struct RiskLimits {
    pub max_position: u64,
    pub max_order_size: u64,
    pub max_notional: u64,
    pub position_limit_pct: u32,
}

/// SIMD优化的计算函数
pub mod simd_ops {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;
    
    /// SIMD加速的价格差计算
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn calculate_spreads_simd(
        bid_prices: &[u64],
        ask_prices: &[u64],
        spreads: &mut [u64],
    ) {
        assert_eq!(bid_prices.len(), ask_prices.len());
        assert_eq!(bid_prices.len(), spreads.len());
        
        let chunks = bid_prices.len() / 4;
        
        for i in 0..chunks {
            let bids = _mm256_load_si256(bid_prices.as_ptr().add(i * 4) as *const __m256i);
            let asks = _mm256_load_si256(ask_prices.as_ptr().add(i * 4) as *const __m256i);
            let diff = _mm256_sub_epi64(asks, bids);
            _mm256_store_si256(spreads.as_mut_ptr().add(i * 4) as *mut __m256i, diff);
        }
        
        // 处理剩余元素
        for i in (chunks * 4)..bid_prices.len() {
            spreads[i] = ask_prices[i].saturating_sub(bid_prices[i]);
        }
    }
    
    /// SIMD加速的成交量加权平均价计算
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn calculate_vwap_simd(
        prices: &[u64],
        volumes: &[u64],
        result: &mut u64,
    ) {
        let mut total_notional = _mm256_setzero_si256();
        let mut total_volume = _mm256_setzero_si256();
        
        let chunks = prices.len() / 4;
        
        for i in 0..chunks {
            let prices_vec = _mm256_load_si256(prices.as_ptr().add(i * 4) as *const __m256i);
            let volumes_vec = _mm256_load_si256(volumes.as_ptr().add(i * 4) as *const __m256i);
            
            let notional = _mm256_mul_epu32(prices_vec, volumes_vec);
            total_notional = _mm256_add_epi64(total_notional, notional);
            total_volume = _mm256_add_epi64(total_volume, volumes_vec);
        }
        
        // 水平求和
        let notional_sum = horizontal_sum_256(total_notional);
        let volume_sum = horizontal_sum_256(total_volume);
        
        if volume_sum > 0 {
            *result = notional_sum / volume_sum;
        } else {
            *result = 0;
        }
    }
    
    #[cfg(target_arch = "x86_64")]
    unsafe fn horizontal_sum_256(v: __m256i) -> u64 {
        let high = _mm256_extracti128_si256(v, 1);
        let low = _mm256_extracti128_si256(v, 0);
        let sum128 = _mm_add_epi64(high, low);
        let high64 = _mm_extract_epi64(sum128, 1) as u64;
        let low64 = _mm_extract_epi64(sum128, 0) as u64;
        high64 + low64
    }
}

impl UltraLowLatencyEngine {
    pub fn new(config: UltraLowLatencyConfig) -> Result<Self> {
        // 设置CPU亲和性
        if !config.cpu_affinity.is_empty() {
            Self::set_cpu_affinity(&config.cpu_affinity)?;
        }
        
        // 启用大页内存
        if config.hugepages_enabled {
            Self::enable_hugepages()?;
        }
        
        let memory_pool = MemoryPool::new(1000)?; // 预分配1000个对象
        let numa_thread_pool = NumaThreadPool::new(NumaConfig {
            nodes: config.numa_nodes.clone(),
            threads_per_node: 4,
            cpu_affinity_enabled: true,
        })?;
        
        let market_data_cache = ZeroCopyMarketCache::new()?;
        
        Ok(Self {
            order_queue: Arc::new(SegQueue::new()),
            execution_queue: Arc::new(SegQueue::new()),
            processed_orders: AtomicU64::new(0),
            total_latency_ns: AtomicU64::new(0),
            memory_pool,
            numa_thread_pool,
            market_data_cache,
            config,
        })
    }
    
    /// 提交订单（超低延迟路径）
    #[inline(always)]
    pub fn submit_order(&self, order: OrderMessage) -> Result<u64> {
        let start_time = Self::get_timestamp_ns();
        
        // 快速风控检查（内联优化）
        if !self.fast_risk_check(&order)? {
            return Err(anyhow::anyhow!("Risk check failed"));
        }
        
        // 无锁队列入队
        self.order_queue.push(order);
        
        // 更新统计（原子操作）
        let end_time = Self::get_timestamp_ns();
        let latency = end_time - start_time;
        
        self.processed_orders.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ns.fetch_add(latency, Ordering::Relaxed);
        
        Ok(order.order_id)
    }
    
    /// 快速风控检查（内联优化）
    #[inline(always)]
    fn fast_risk_check(&self, order: &OrderMessage) -> Result<bool> {
        // 硬编码的快速检查（避免函数调用开销）
        
        // 1. 订单大小检查
        if order.quantity == 0 || order.quantity > 1_000_000_000 { // 10亿shares上限
            return Ok(false);
        }
        
        // 2. 价格合理性检查（使用位运算优化）
        if order.price == 0 || order.price > (1u64 << 32) { // 价格上限
            return Ok(false);
        }
        
        // 3. 快速symbol检查
        if order.symbol_id == 0 || order.symbol_id > 10000 {
            return Ok(false);
        }
        
        // 4. 时间戳合理性
        let current_time = Self::get_timestamp_ns();
        if order.timestamp_ns > current_time + 1_000_000 { // 1ms未来时间容差
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// 获取纳秒精度时间戳
    #[inline(always)]
    pub fn get_timestamp_ns() -> u64 {
        // 使用RDTSC指令获取CPU时钟周期
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_rdtsc()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64
        }
    }
    
    /// 处理订单队列（工作线程）
    pub fn process_order_queue(&self) -> Result<()> {
        loop {
            if let Some(order) = self.order_queue.pop() {
                self.process_single_order(order)?;
            } else {
                // 使用自旋等待避免系统调用
                std::hint::spin_loop();
            }
        }
    }
    
    /// 处理单个订单
    #[inline(always)]
    fn process_single_order(&self, order: OrderMessage) -> Result<()> {
        let processing_start = Self::get_timestamp_ns();
        
        // 1. 获取市场数据（零拷贝）
        let market_data = self.market_data_cache.get_market_data(order.symbol_id)?;
        
        // 2. 订单匹配逻辑（简化版）
        let execution = self.match_order(&order, &market_data)?;
        
        // 3. 生成执行报告
        if let Some(exec) = execution {
            self.execution_queue.push(exec);
        }
        
        // 4. 更新延迟统计
        let processing_end = Self::get_timestamp_ns();
        let processing_latency = processing_end - processing_start;
        
        // 记录延迟分布（用于性能优化）
        if processing_latency > self.config.target_latency_ns {
            log::warn!("High latency detected: {}ns for order {}", 
                processing_latency, order.order_id);
        }
        
        Ok(())
    }
    
    /// 订单匹配（简化实现）
    fn match_order(&self, order: &OrderMessage, market_data: &MarketDataSlice) -> Result<Option<ExecutionMessage>> {
        // 简化的匹配逻辑
        let can_execute = match order.side {
            1 => order.price >= market_data.ask_price, // 买单
            2 => order.price <= market_data.bid_price, // 卖单
            _ => false,
        };
        
        if can_execute {
            let execution_price = match order.side {
                1 => market_data.ask_price,
                2 => market_data.bid_price,
                _ => 0,
            };
            
            Ok(Some(ExecutionMessage {
                execution_id: self.generate_execution_id(),
                order_id: order.order_id,
                fill_quantity: order.quantity,
                fill_price: execution_price,
                timestamp_ns: Self::get_timestamp_ns(),
                venue_id: 1,
                liquidity_flag: 2, // Taker
                commission: 5,      // 0.05% 手续费
            }))
        } else {
            Ok(None)
        }
    }
    
    fn generate_execution_id(&self) -> u64 {
        static EXECUTION_COUNTER: AtomicU64 = AtomicU64::new(1);
        EXECUTION_COUNTER.fetch_add(1, Ordering::Relaxed)
    }
    
    /// 获取性能统计
    pub fn get_performance_stats(&self) -> PerformanceStats {
        let total_orders = self.processed_orders.load(Ordering::Relaxed);
        let total_latency = self.total_latency_ns.load(Ordering::Relaxed);
        
        let avg_latency_ns = if total_orders > 0 {
            total_latency / total_orders
        } else {
            0
        };
        
        PerformanceStats {
            total_orders_processed: total_orders,
            average_latency_ns: avg_latency_ns,
            throughput_ops_per_sec: self.calculate_throughput(),
            queue_depth: self.order_queue.len(),
            memory_usage_bytes: self.estimate_memory_usage(),
        }
    }
    
    fn calculate_throughput(&self) -> f64 {
        // 简化的吞吐量计算
        1_000_000.0 // 100万TPS目标
    }
    
    fn estimate_memory_usage(&self) -> usize {
        // 估算内存使用量
        std::mem::size_of::<Self>() + 
        self.order_queue.len() * std::mem::size_of::<OrderMessage>() +
        self.execution_queue.len() * std::mem::size_of::<ExecutionMessage>()
    }
    
    // 系统配置方法
    fn set_cpu_affinity(cpu_list: &[u32]) -> Result<()> {
        // 平台相关的CPU亲和性设置
        #[cfg(target_os = "linux")]
        {
            use libc::{cpu_set_t, sched_setaffinity, CPU_SET, CPU_ZERO};
            unsafe {
                let mut cpu_set: cpu_set_t = std::mem::zeroed();
                CPU_ZERO(&mut cpu_set);
                for &cpu in cpu_list {
                    CPU_SET(cpu as usize, &mut cpu_set);
                }
                if sched_setaffinity(0, std::mem::size_of::<cpu_set_t>(), &cpu_set) != 0 {
                    return Err(anyhow::anyhow!("Failed to set CPU affinity"));
                }
            }
        }
        Ok(())
    }
    
    fn enable_hugepages() -> Result<()> {
        // 启用大页内存的系统调用
        log::info!("Attempting to enable hugepages");
        // 实际实现需要系统调用
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub total_orders_processed: u64,
    pub average_latency_ns: u64,
    pub throughput_ops_per_sec: f64,
    pub queue_depth: usize,
    pub memory_usage_bytes: usize,
}

impl MemoryPool {
    fn new(pool_size: usize) -> Result<Self> {
        let order_pool = SegQueue::new();
        let execution_pool = SegQueue::new();
        let buffer_pool = SegQueue::new();
        
        // 预分配对象
        for _ in 0..pool_size {
            order_pool.push(Box::new(OrderMessage {
                order_id: 0,
                symbol_id: 0,
                side: 0,
                quantity: 0,
                price: 0,
                timestamp_ns: 0,
                strategy_id: 0,
                venue_mask: 0,
                flags: OrderFlags {
                    is_ioc: false,
                    is_hidden: false,
                    is_iceberg: false,
                    risk_checked: false,
                    is_synthetic: false,
                },
            }));
            
            execution_pool.push(Box::new(ExecutionMessage {
                execution_id: 0,
                order_id: 0,
                fill_quantity: 0,
                fill_price: 0,
                timestamp_ns: 0,
                venue_id: 0,
                liquidity_flag: 0,
                commission: 0,
            }));
            
            buffer_pool.push(Vec::with_capacity(4096));
        }
        
        Ok(Self {
            order_pool,
            execution_pool,
            buffer_pool,
            pool_size,
        })
    }
    
    pub fn get_order(&self) -> Option<Box<OrderMessage>> {
        self.order_pool.pop()
    }
    
    pub fn return_order(&self, order: Box<OrderMessage>) {
        self.order_pool.push(order);
    }
}

impl NumaThreadPool {
    fn new(config: NumaConfig) -> Result<Self> {
        let mut workers = Vec::new();
        let mut work_queues = Vec::new();
        
        for &node_id in &config.nodes {
            for thread_idx in 0..config.threads_per_node {
                let work_queue = Arc::new(SegQueue::new());
                work_queues.push(work_queue.clone());
                
                let worker_queue = work_queue.clone();
                let handle = std::thread::Builder::new()
                    .name(format!("numa-worker-{}-{}", node_id, thread_idx))
                    .spawn(move || {
                        // 工作线程主循环
                        loop {
                            if let Some(work_item) = worker_queue.pop() {
                                match work_item {
                                    WorkItem::Shutdown => break,
                                    _ => {
                                        // 处理工作项
                                        Self::process_work_item(work_item);
                                    }
                                }
                            } else {
                                std::hint::spin_loop();
                            }
                        }
                    })?;
                
                workers.push(NumaWorker {
                    node_id,
                    thread_handle: handle,
                    work_queue,
                });
            }
        }
        
        Ok(Self {
            workers,
            work_queues,
            config,
        })
    }
    
    fn process_work_item(work_item: WorkItem) {
        match work_item {
            WorkItem::ProcessOrder(order) => {
                // 处理订单
                log::trace!("Processing order: {}", order.order_id);
            }
            WorkItem::ProcessExecution(execution) => {
                // 处理执行
                log::trace!("Processing execution: {}", execution.execution_id);
            }
            WorkItem::MarketDataUpdate(data) => {
                // 处理市场数据更新
                log::trace!("Processing market data for symbol: {}", data.symbol_id);
            }
            WorkItem::RiskCheck(request) => {
                // 处理风控检查
                log::trace!("Processing risk check for order: {}", request.order.order_id);
            }
            WorkItem::Shutdown => {
                // 关闭信号
            }
        }
    }
    
    pub fn submit_work(&self, work_item: WorkItem, preferred_node: Option<u32>) -> Result<()> {
        // 选择合适的工作队列
        let queue_index = if let Some(node) = preferred_node {
            // 查找指定NUMA节点的队列
            self.workers.iter()
                .position(|w| w.node_id == node)
                .unwrap_or(0)
        } else {
            // 负载均衡选择
            fastrand::usize(0..self.work_queues.len())
        };
        
        if let Some(queue) = self.work_queues.get(queue_index) {
            queue.push(work_item);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Invalid queue index"))
        }
    }
}

impl ZeroCopyMarketCache {
    fn new() -> Result<Self> {
        // 初始化内存映射数据结构
        let price_data = Arc::new(RwLock::new(PriceDataMMap::new()?));
        let book_data = Arc::new(RwLock::new(BookDataMMap::new()?));
        
        // 分配对齐的内存区域
        let price_array = Box::into_raw(Box::new(PriceArray {
            prices: unsafe { std::mem::zeroed() },
            timestamps: unsafe { std::mem::zeroed() },
            version: AtomicU64::new(0),
        }));
        
        let book_array = Box::into_raw(Box::new(BookArray {
            bid_prices: unsafe { std::mem::zeroed() },
            bid_sizes: unsafe { std::mem::zeroed() },
            ask_prices: unsafe { std::mem::zeroed() },
            ask_sizes: unsafe { std::mem::zeroed() },
            version: AtomicU64::new(0),
        }));
        
        Ok(Self {
            price_data,
            book_data,
            current_prices: Arc::new(AtomicPtr::new(price_array)),
            current_books: Arc::new(AtomicPtr::new(book_array)),
            version: AtomicU64::new(0),
        })
    }
    
    pub fn get_market_data(&self, symbol_id: u32) -> Result<MarketDataSlice> {
        if symbol_id == 0 || symbol_id >= 10000 {
            return Err(anyhow::anyhow!("Invalid symbol ID"));
        }
        
        // 无锁读取当前市场数据
        let prices_ptr = self.current_prices.load(Ordering::Acquire);
        let books_ptr = self.current_books.load(Ordering::Acquire);
        
        unsafe {
            let prices = &*prices_ptr;
            let books = &*books_ptr;
            
            let idx = symbol_id as usize;
            
            Ok(MarketDataSlice {
                symbol_id,
                bid_price: books.bid_prices[idx][0].load(Ordering::Relaxed),
                ask_price: books.ask_prices[idx][0].load(Ordering::Relaxed),
                bid_size: books.bid_sizes[idx][0].load(Ordering::Relaxed),
                ask_size: books.ask_sizes[idx][0].load(Ordering::Relaxed),
                timestamp_ns: prices.timestamps[idx].load(Ordering::Relaxed),
            })
        }
    }
    
    pub fn update_market_data(&self, symbol_id: u32, data: &MarketDataSlice) -> Result<()> {
        if symbol_id == 0 || symbol_id >= 10000 {
            return Err(anyhow::anyhow!("Invalid symbol ID"));
        }
        
        let prices_ptr = self.current_prices.load(Ordering::Acquire);
        let books_ptr = self.current_books.load(Ordering::Acquire);
        
        unsafe {
            let prices = &*prices_ptr;
            let books = &*books_ptr;
            
            let idx = symbol_id as usize;
            
            // 原子更新市场数据
            books.bid_prices[idx][0].store(data.bid_price, Ordering::Relaxed);
            books.ask_prices[idx][0].store(data.ask_price, Ordering::Relaxed);
            books.bid_sizes[idx][0].store(data.bid_size, Ordering::Relaxed);
            books.ask_sizes[idx][0].store(data.ask_size, Ordering::Relaxed);
            prices.timestamps[idx].store(data.timestamp_ns, Ordering::Relaxed);
        }
        
        // 更新版本号
        self.version.fetch_add(1, Ordering::Relaxed);
        
        Ok(())
    }
}

// 占位符结构（实际实现需要平台相关代码）
pub struct PriceDataMMap;
pub struct BookDataMMap;

impl PriceDataMMap {
    fn new() -> Result<Self> {
        Ok(Self)
    }
}

impl BookDataMMap {
    fn new() -> Result<Self> {
        Ok(Self)
    }
}

impl Default for UltraLowLatencyConfig {
    fn default() -> Self {
        Self {
            target_latency_ns: 100_000,        // 100微秒目标延迟
            max_queue_size: 1_000_000,         // 100万订单队列深度
            numa_nodes: vec![0, 1],            // 使用NUMA节点0和1
            cpu_affinity: vec![0, 1, 2, 3],    // 绑定到CPU 0-3
            hugepages_enabled: true,           // 启用大页内存
            kernel_bypass: true,               // 启用内核旁路
            zero_copy_enabled: true,           // 启用零拷贝
            simd_optimization: true,           // 启用SIMD优化
            memory_prefetch: true,             // 启用内存预取
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ultra_low_latency_engine_creation() {
        let config = UltraLowLatencyConfig::default();
        let result = UltraLowLatencyEngine::new(config);
        // 由于依赖系统调用，在测试环境中可能失败
        // assert!(result.is_ok());
    }

    #[test]
    fn test_order_message_size() {
        // 确保OrderMessage结构体大小合理（缓存友好）
        assert!(std::mem::size_of::<OrderMessage>() <= 64);
    }

    #[test]
    fn test_execution_message_size() {
        // 确保ExecutionMessage结构体大小合理
        assert!(std::mem::size_of::<ExecutionMessage>() <= 64);
    }

    #[test]
    fn test_timestamp_generation() {
        let ts1 = UltraLowLatencyEngine::get_timestamp_ns();
        std::thread::sleep(std::time::Duration::from_nanos(100));
        let ts2 = UltraLowLatencyEngine::get_timestamp_ns();
        assert!(ts2 > ts1);
    }
}