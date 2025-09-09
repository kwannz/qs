use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use chrono::{DateTime, Utc, NaiveDateTime};
use anyhow::{Result, anyhow};
use std::fs;
use tracing::{info, warn, error};

/// 数据类型枚举
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataType {
    FuturesPriceHistory,
    LiquidationAdvanced,
    LongShortRatioAdvanced,
    OrderbookAdvanced,
    NetPosition,
    SpotOrderbook,
    SpotTakerBuySell,
    EtfData,
    IndexData,
    HyperliquidWhale,
}

impl DataType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DataType::FuturesPriceHistory => "futures_price_history",
            DataType::LiquidationAdvanced => "liquidation_advanced",
            DataType::LongShortRatioAdvanced => "long_short_ratio_advanced",
            DataType::OrderbookAdvanced => "orderbook_advanced",
            DataType::NetPosition => "net_position",
            DataType::SpotOrderbook => "spot_orderbook",
            DataType::SpotTakerBuySell => "spot_taker_buy_sell",
            DataType::EtfData => "etf_data",
            DataType::IndexData => "index_data",
            DataType::HyperliquidWhale => "hyperliquid_whale",
        }
    }
    
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "futures_price_history" => Some(DataType::FuturesPriceHistory),
            "liquidation_advanced" => Some(DataType::LiquidationAdvanced),
            "long_short_ratio_advanced" => Some(DataType::LongShortRatioAdvanced),
            "orderbook_advanced" => Some(DataType::OrderbookAdvanced),
            "net_position" => Some(DataType::NetPosition),
            "spot_orderbook" => Some(DataType::SpotOrderbook),
            "spot_taker_buy_sell" => Some(DataType::SpotTakerBuySell),
            "etf_data" => Some(DataType::EtfData),
            "index_data" => Some(DataType::IndexData),
            "hyperliquid_whale" => Some(DataType::HyperliquidWhale),
            _ => None,
        }
    }
}

/// 数据查询参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQuery {
    pub data_type: DataType,
    pub exchange: Option<String>,
    pub symbol: Option<String>,
    pub interval: Option<String>,
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub limit: Option<usize>,
}

impl Default for DataQuery {
    fn default() -> Self {
        Self {
            data_type: DataType::FuturesPriceHistory,
            exchange: None,
            symbol: None,
            interval: Some("1h".to_string()),
            start_time: None,
            end_time: None,
            limit: Some(1000),
        }
    }
}

/// 数据信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataInfo {
    pub data_type: DataType,
    pub exchange: Option<String>,
    pub symbol: Option<String>,
    pub interval: Option<String>,
    pub record_count: usize,
    pub file_size_bytes: u64,
    pub date_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    pub last_updated: DateTime<Utc>,
}

/// 数据读取器缓存
pub struct DataCache {
    cache: HashMap<String, DataFrame>,
    max_size: usize,
}

impl DataCache {
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
        }
    }
    
    pub fn get(&self, key: &str) -> Option<&DataFrame> {
        self.cache.get(key)
    }
    
    pub fn insert(&mut self, key: String, df: DataFrame) {
        if self.cache.len() >= self.max_size {
            // 简单的LRU策略：删除第一个元素
            if let Some(first_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&first_key);
            }
        }
        self.cache.insert(key, df);
    }
    
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}

/// 主要的数据读取器
pub struct DataReader {
    hot_data_path: PathBuf,
    warm_data_path: PathBuf,
    cold_archive_path: PathBuf,
    cache: DataCache,
}

impl DataReader {
    /// 创建新的数据读取器
    pub fn new(base_path: impl AsRef<Path>) -> Self {
        let base = base_path.as_ref();
        
        Self {
            hot_data_path: base.join("crypto"),
            warm_data_path: base.join("historical data"),
            cold_archive_path: base.join("archive"),
            cache: DataCache::new(64),
        }
    }
    
    /// 使用自定义缓存大小创建数据读取器
    pub fn with_cache_size(base_path: impl AsRef<Path>, cache_size: usize) -> Self {
        let base = base_path.as_ref();
        
        Self {
            hot_data_path: base.join("crypto"),
            warm_data_path: base.join("historical data"),
            cold_archive_path: base.join("archive"),
            cache: DataCache::new(cache_size),
        }
    }
    
    /// 获取数据文件路径
    fn get_data_file_path(&self, query: &DataQuery) -> Result<PathBuf> {
        let data_type_str = query.data_type.as_str();
        
        // 根据数据类型确定基础路径
        let base_path = match query.data_type {
            DataType::FuturesPriceHistory 
            | DataType::LiquidationAdvanced 
            | DataType::LongShortRatioAdvanced 
            | DataType::OrderbookAdvanced 
            | DataType::NetPosition => {
                self.hot_data_path.join("futures").join(data_type_str)
            }
            DataType::SpotOrderbook 
            | DataType::SpotTakerBuySell => {
                self.hot_data_path.join("spot").join(data_type_str)
            }
            DataType::EtfData => {
                self.hot_data_path.join("etf").join(data_type_str)
            }
            DataType::IndexData 
            | DataType::HyperliquidWhale => {
                self.hot_data_path.join("indicators").join("market").join(data_type_str)
            }
        };
        
        if !base_path.exists() {
            return Err(anyhow!("数据路径不存在: {:?}", base_path));
        }
        
        // 构建文件名模式
        let mut filename = data_type_str.to_string();
        
        if let Some(exchange) = &query.exchange {
            filename.push('_');
            filename.push_str(exchange);
        }
        
        if let Some(symbol) = &query.symbol {
            filename.push('_');
            filename.push_str(symbol);
        }
        
        if let Some(interval) = &query.interval {
            filename.push('_');
            filename.push_str(interval);
        }
        
        filename.push_str(".parquet");
        
        let file_path = base_path.join(&filename);
        
        if file_path.exists() {
            Ok(file_path)
        } else {
            // 如果精确匹配不存在，尝试模糊匹配
            self.find_matching_file(&base_path, &query)
        }
    }
    
    /// 查找匹配的文件
    fn find_matching_file(&self, base_path: &Path, query: &DataQuery) -> Result<PathBuf> {
        let entries = fs::read_dir(base_path)?;
        
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            
            if path.extension().map_or(false, |ext| ext == "parquet") {
                let filename = path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("");
                
                // 检查是否匹配查询条件
                if self.matches_query(filename, query) {
                    return Ok(path);
                }
            }
        }
        
        Err(anyhow!("未找到匹配的数据文件"))
    }
    
    /// 检查文件名是否匹配查询条件
    fn matches_query(&self, filename: &str, query: &DataQuery) -> bool {
        let parts: Vec<&str> = filename.split('_').collect();
        
        if parts.is_empty() {
            return false;
        }
        
        // 检查数据类型
        if !filename.starts_with(query.data_type.as_str()) {
            return false;
        }
        
        // 检查交易所
        if let Some(exchange) = &query.exchange {
            if !filename.contains(exchange) {
                return false;
            }
        }
        
        // 检查交易对
        if let Some(symbol) = &query.symbol {
            if !filename.contains(symbol) {
                return false;
            }
        }
        
        // 检查时间间隔
        if let Some(interval) = &query.interval {
            if !filename.contains(interval) {
                return false;
            }
        }
        
        true
    }
    
    /// 读取数据
    pub fn read_data(&mut self, query: &DataQuery) -> Result<DataFrame> {
        let cache_key = format!(
            "{}_{}_{}_{}", 
            query.data_type.as_str(),
            query.exchange.as_deref().unwrap_or(""),
            query.symbol.as_deref().unwrap_or(""),
            query.interval.as_deref().unwrap_or("")
        );
        
        // 检查缓存
        if let Some(cached_df) = self.cache.get(&cache_key) {
            info!("从缓存获取数据: {}", cache_key);
            return Ok(cached_df.clone());
        }
        
        // 获取文件路径
        let file_path = self.get_data_file_path(query)?;
        
        info!("读取数据文件: {:?}", file_path);
        
        // 读取Parquet文件
        let mut df = LazyFrame::scan_parquet(&file_path, ScanArgsParquet::default())?
            .collect()?;
            
        // 应用时间过滤
        if query.start_time.is_some() || query.end_time.is_some() {
            df = self.apply_time_filter(df, query)?;
        }
        
        // 应用限制
        if let Some(limit) = query.limit {
            df = df.head(Some(limit));
        }
        
        // 缓存结果
        self.cache.insert(cache_key, df.clone());
        
        Ok(df)
    }
    
    /// 应用时间过滤
    fn apply_time_filter(&self, mut df: DataFrame, query: &DataQuery) -> Result<DataFrame> {
        if !df.get_column_names().contains(&"timestamp") {
            return Ok(df);
        }
        
        let timestamp_col = df.column("timestamp")?;
        
        // 转换为DateTime类型（如果需要）
        let timestamp_col = if timestamp_col.dtype() == &DataType::Datetime(TimeUnit::Milliseconds, None) {
            timestamp_col.clone()
        } else {
            // 尝试转换
            timestamp_col.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))?
        };
        
        // 构建过滤条件
        let mut mask = BooleanChunked::from_iter(
            std::iter::repeat(true).take(df.height())
        );
        
        if let Some(start_time) = query.start_time {
            let start_timestamp = start_time.timestamp_millis();
            let start_mask = timestamp_col.gt_eq(AnyValue::Int64(start_timestamp))?;
            mask = mask.bitand(&start_mask.bool()?);
        }
        
        if let Some(end_time) = query.end_time {
            let end_timestamp = end_time.timestamp_millis();
            let end_mask = timestamp_col.lt_eq(AnyValue::Int64(end_timestamp))?;
            mask = mask.bitand(&end_mask.bool()?);
        }
        
        df = df.filter(&mask)?;
        Ok(df)
    }
    
    /// 获取最新数据
    pub fn read_latest(&mut self, data_type: DataType, exchange: &str, symbol: &str, limit: Option<usize>) -> Result<DataFrame> {
        let query = DataQuery {
            data_type,
            exchange: Some(exchange.to_string()),
            symbol: Some(symbol.to_string()),
            interval: Some("1h".to_string()),
            start_time: None,
            end_time: None,
            limit,
        };
        
        let mut df = self.read_data(&query)?;
        
        // 如果有时间戳列，按时间倒序排序
        if df.get_column_names().contains(&"timestamp") {
            df = df.sort(["timestamp"], false)?;
        }
        
        Ok(df)
    }
    
    /// 获取历史数据范围
    pub fn read_range(&mut self, 
                     data_type: DataType,
                     exchange: &str, 
                     symbol: &str,
                     start: DateTime<Utc>, 
                     end: DateTime<Utc>) -> Result<DataFrame> {
        let query = DataQuery {
            data_type,
            exchange: Some(exchange.to_string()),
            symbol: Some(symbol.to_string()),
            interval: Some("1h".to_string()),
            start_time: Some(start),
            end_time: Some(end),
            limit: None,
        };
        
        self.read_data(&query)
    }
    
    /// 获取可用的数据类型
    pub fn get_available_data_types(&self) -> Vec<DataType> {
        let mut data_types = Vec::new();
        
        // 扫描futures目录
        if let Ok(entries) = fs::read_dir(&self.hot_data_path.join("futures")) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if let Some(data_type) = DataType::from_str(name) {
                        data_types.push(data_type);
                    }
                }
            }
        }
        
        // 扫描spot目录
        if let Ok(entries) = fs::read_dir(&self.hot_data_path.join("spot")) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if let Some(data_type) = DataType::from_str(name) {
                        data_types.push(data_type);
                    }
                }
            }
        }
        
        // 扫描etf目录
        if let Ok(entries) = fs::read_dir(&self.hot_data_path.join("etf")) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if let Some(data_type) = DataType::from_str(name) {
                        data_types.push(data_type);
                    }
                }
            }
        }
        
        // 扫描indicators目录
        if let Ok(entries) = fs::read_dir(&self.hot_data_path.join("indicators").join("market")) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if let Some(data_type) = DataType::from_str(name) {
                        data_types.push(data_type);
                    }
                }
            }
        }
        
        data_types.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        data_types.dedup();
        data_types
    }
    
    /// 获取数据信息
    pub fn get_data_info(&mut self, query: &DataQuery) -> Result<DataInfo> {
        let file_path = self.get_data_file_path(query)?;
        
        let metadata = fs::metadata(&file_path)?;
        let file_size = metadata.len();
        let last_updated = DateTime::<Utc>::from(metadata.modified()?);
        
        // 读取数据获取更多信息
        let df = self.read_data(query)?;
        let record_count = df.height();
        
        // 获取日期范围
        let date_range = if df.get_column_names().contains(&"timestamp") {
            let timestamp_col = df.column("timestamp")?;
            if let (Some(min_val), Some(max_val)) = (timestamp_col.min(), timestamp_col.max()) {
                // 转换时间戳
                if let (Ok(min_ts), Ok(max_ts)) = (
                    min_val.extract::<i64>(),
                    max_val.extract::<i64>()
                ) {
                    let min_dt = DateTime::<Utc>::from_utc(
                        NaiveDateTime::from_timestamp_millis(min_ts).unwrap_or_default(),
                        Utc
                    );
                    let max_dt = DateTime::<Utc>::from_utc(
                        NaiveDateTime::from_timestamp_millis(max_ts).unwrap_or_default(),
                        Utc
                    );
                    Some((min_dt, max_dt))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };
        
        Ok(DataInfo {
            data_type: query.data_type.clone(),
            exchange: query.exchange.clone(),
            symbol: query.symbol.clone(),
            interval: query.interval.clone(),
            record_count,
            file_size_bytes: file_size,
            date_range,
            last_updated,
        })
    }
    
    /// 清除缓存
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        info!("数据缓存已清除");
    }
}

// 便捷函数
pub fn create_data_reader(base_path: impl AsRef<Path>) -> DataReader {
    DataReader::new(base_path)
}

pub fn read_btc_price_data(base_path: impl AsRef<Path>, exchange: &str, days: u64) -> Result<DataFrame> {
    let mut reader = DataReader::new(base_path);
    let end_time = Utc::now();
    let start_time = end_time - chrono::Duration::days(days as i64);
    
    reader.read_range(
        DataType::FuturesPriceHistory,
        exchange,
        "BTCUSDT",
        start_time,
        end_time
    )
}

pub fn read_eth_price_data(base_path: impl AsRef<Path>, exchange: &str, days: u64) -> Result<DataFrame> {
    let mut reader = DataReader::new(base_path);
    let end_time = Utc::now();
    let start_time = end_time - chrono::Duration::days(days as i64);
    
    reader.read_range(
        DataType::FuturesPriceHistory,
        exchange,
        "ETHUSDT",
        start_time,
        end_time
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_data_type_conversion() {
        assert_eq!(DataType::FuturesPriceHistory.as_str(), "futures_price_history");
        assert_eq!(DataType::from_str("futures_price_history"), Some(DataType::FuturesPriceHistory));
        assert_eq!(DataType::from_str("invalid"), None);
    }
    
    #[test]
    fn test_data_query_default() {
        let query = DataQuery::default();
        assert_eq!(query.data_type, DataType::FuturesPriceHistory);
        assert_eq!(query.interval, Some("1h".to_string()));
        assert_eq!(query.limit, Some(1000));
    }
    
    #[test]
    fn test_data_cache() {
        let mut cache = DataCache::new(2);
        let df1 = DataFrame::empty();
        let df2 = DataFrame::empty();
        
        cache.insert("test1".to_string(), df1);
        cache.insert("test2".to_string(), df2);
        
        assert!(cache.get("test1").is_some());
        assert!(cache.get("test2").is_some());
        
        // 添加第三个应该删除第一个
        let df3 = DataFrame::empty();
        cache.insert("test3".to_string(), df3);
        
        assert!(cache.get("test1").is_none());
        assert!(cache.get("test2").is_some());
        assert!(cache.get("test3").is_some());
    }
}