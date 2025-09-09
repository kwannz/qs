//! 日志存储模块

use std::path::Path;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use tokio::sync::mpsc;
use anyhow::Result;
use chrono::{DateTime, Utc};

use crate::formatter::LogEntry;

/// 日志存储后端接口
#[async_trait::async_trait]
pub trait LogStorage: Send + Sync {
    /// 写入日志条目
    async fn write_log(&mut self, entry: &LogEntry) -> Result<()>;
    
    /// 批量写入日志条目
    async fn write_logs(&mut self, entries: &[LogEntry]) -> Result<()> {
        for entry in entries {
            self.write_log(entry).await?;
        }
        Ok(())
    }
    
    /// 刷新缓存
    async fn flush(&mut self) -> Result<()>;
    
    /// 关闭存储
    async fn close(&mut self) -> Result<()>;
}

/// 文件存储后端
pub struct FileStorage {
    writer: BufWriter<File>,
    #[allow(dead_code)]
    path: String,
}

impl FileStorage {
    /// 创建文件存储
    pub fn new(path: &str) -> Result<Self> {
        // 确保目录存在
        if let Some(parent) = Path::new(path).parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        
        let writer = BufWriter::new(file);

        Ok(Self {
            writer,
            path: path.to_string(),
        })
    }
}

#[async_trait::async_trait]
impl LogStorage for FileStorage {
    async fn write_log(&mut self, entry: &LogEntry) -> Result<()> {
        let json = serde_json::to_string(entry)?;
        writeln!(self.writer, "{json}")?;
        Ok(())
    }

    async fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

/// 内存存储后端 (测试用)
pub struct MemoryStorage {
    entries: Vec<LogEntry>,
    max_entries: usize,
}

impl MemoryStorage {
    /// 创建内存存储
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
        }
    }

    /// 获取所有日志条目
    pub fn get_entries(&self) -> &[LogEntry] {
        &self.entries
    }

    /// 清空日志条目
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[async_trait::async_trait]
impl LogStorage for MemoryStorage {
    async fn write_log(&mut self, entry: &LogEntry) -> Result<()> {
        self.entries.push(entry.clone());
        
        // 保持最大条目数限制
        if self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }
        
        Ok(())
    }

    async fn flush(&mut self) -> Result<()> {
        // 内存存储无需刷新
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        // 内存存储无需关闭操作
        Ok(())
    }
}

/// 存储后端枚举
pub enum StorageBackend {
    File(FileStorage),
    Memory(MemoryStorage),
}

#[async_trait::async_trait]
impl LogStorage for StorageBackend {
    async fn write_log(&mut self, entry: &LogEntry) -> Result<()> {
        match self {
            StorageBackend::File(storage) => storage.write_log(entry).await,
            StorageBackend::Memory(storage) => storage.write_log(entry).await,
        }
    }

    async fn write_logs(&mut self, entries: &[LogEntry]) -> Result<()> {
        match self {
            StorageBackend::File(storage) => storage.write_logs(entries).await,
            StorageBackend::Memory(storage) => storage.write_logs(entries).await,
        }
    }

    async fn flush(&mut self) -> Result<()> {
        match self {
            StorageBackend::File(storage) => storage.flush().await,
            StorageBackend::Memory(storage) => storage.flush().await,
        }
    }

    async fn close(&mut self) -> Result<()> {
        match self {
            StorageBackend::File(storage) => storage.close().await,
            StorageBackend::Memory(storage) => storage.close().await,
        }
    }
}

/// 多重存储后端
pub struct MultiStorage {
    storages: Vec<StorageBackend>,
}

impl Default for MultiStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiStorage {
    /// 创建多重存储
    pub fn new() -> Self {
        Self {
            storages: Vec::new(),
        }
    }

    /// 添加存储后端
    pub fn add_storage(mut self, storage: StorageBackend) -> Self {
        self.storages.push(storage);
        self
    }
}

#[async_trait::async_trait]
impl LogStorage for MultiStorage {
    async fn write_log(&mut self, entry: &LogEntry) -> Result<()> {
        for storage in &mut self.storages {
            if let Err(e) = storage.write_log(entry).await {
                tracing::error!(error = %e, "存储后端写入失败");
            }
        }
        Ok(())
    }

    async fn write_logs(&mut self, entries: &[LogEntry]) -> Result<()> {
        for storage in &mut self.storages {
            if let Err(e) = storage.write_logs(entries).await {
                tracing::error!(error = %e, "存储后端批量写入失败");
            }
        }
        Ok(())
    }

    async fn flush(&mut self) -> Result<()> {
        for storage in &mut self.storages {
            if let Err(e) = storage.flush().await {
                tracing::error!(error = %e, "存储后端刷新失败");
            }
        }
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        for storage in &mut self.storages {
            if let Err(e) = storage.close().await {
                tracing::error!(error = %e, "存储后端关闭失败");
            }
        }
        Ok(())
    }
}

/// 异步日志写入器
pub struct AsyncLogWriter {
    tx: mpsc::UnboundedSender<LogWriterMessage>,
}

/// 日志写入器消息
enum LogWriterMessage {
    WriteLog(Box<LogEntry>),
    WriteLogs(Vec<LogEntry>),
    Flush,
    Close,
}

/// 异步日志写入器句柄
pub struct AsyncLogWriterHandle {
    rx: mpsc::UnboundedReceiver<LogWriterMessage>,
    storage: StorageBackend,
}

impl AsyncLogWriter {
    /// 创建异步日志写入器
    pub fn new(storage: StorageBackend) -> (Self, AsyncLogWriterHandle) {
        let (tx, rx) = mpsc::unbounded_channel();
        
        let writer = Self { tx };
        let handle = AsyncLogWriterHandle { rx, storage };
        
        (writer, handle)
    }

    /// 写入日志条目
    pub fn write_log(&self, entry: LogEntry) -> Result<()> {
        self.tx.send(LogWriterMessage::WriteLog(Box::new(entry)))
            .map_err(|e| anyhow::anyhow!("发送日志失败: {}", e))
    }

    /// 批量写入日志条目
    pub fn write_logs(&self, entries: Vec<LogEntry>) -> Result<()> {
        self.tx.send(LogWriterMessage::WriteLogs(entries))
            .map_err(|e| anyhow::anyhow!("发送批量日志失败: {}", e))
    }

    /// 刷新缓存
    pub fn flush(&self) -> Result<()> {
        self.tx.send(LogWriterMessage::Flush)
            .map_err(|e| anyhow::anyhow!("发送刷新命令失败: {}", e))
    }

    /// 关闭写入器
    pub fn close(&self) -> Result<()> {
        self.tx.send(LogWriterMessage::Close)
            .map_err(|e| anyhow::anyhow!("发送关闭命令失败: {}", e))
    }
}

impl AsyncLogWriterHandle {
    /// 运行异步写入器
    pub async fn run(mut self) {
        tracing::info!("启动异步日志写入器");

        while let Some(message) = self.rx.recv().await {
            match message {
                LogWriterMessage::WriteLog(entry) => {
                    if let Err(e) = self.storage.write_log(&entry).await {
                        tracing::error!(error = %e, "异步写入日志失败");
                    }
                }
                LogWriterMessage::WriteLogs(entries) => {
                    if let Err(e) = self.storage.write_logs(&entries).await {
                        tracing::error!(error = %e, "异步批量写入日志失败");
                    }
                }
                LogWriterMessage::Flush => {
                    if let Err(e) = self.storage.flush().await {
                        tracing::error!(error = %e, "异步刷新存储失败");
                    }
                }
                LogWriterMessage::Close => {
                    if let Err(e) = self.storage.close().await {
                        tracing::error!(error = %e, "异步关闭存储失败");
                    }
                    break;
                }
            }
        }

        tracing::info!("异步日志写入器已停止");
    }
}

/// 日志轮转管理器
pub struct LogRotator {
    base_path: String,
    max_file_size: u64,
    max_files: usize,
    current_file_index: usize,
}

impl LogRotator {
    /// 创建日志轮转管理器
    pub fn new(base_path: String, max_file_size: u64, max_files: usize) -> Self {
        Self {
            base_path,
            max_file_size,
            max_files,
            current_file_index: 0,
        }
    }

    /// 检查是否需要轮转
    pub fn should_rotate(&self, current_size: u64) -> bool {
        current_size >= self.max_file_size
    }

    /// 执行轮转
    pub fn rotate(&mut self) -> Result<String> {
        self.current_file_index = (self.current_file_index + 1) % self.max_files;
        
        let new_path = format!("{}.{}", self.base_path, self.current_file_index);
        
        // 如果文件存在，先删除
        if Path::new(&new_path).exists() {
            std::fs::remove_file(&new_path)?;
        }
        
        tracing::info!(
            old_index = (self.current_file_index + self.max_files - 1) % self.max_files,
            new_index = self.current_file_index,
            new_path = %new_path,
            "执行日志轮转"
        );
        
        Ok(new_path)
    }

    /// 获取当前文件路径
    pub fn current_path(&self) -> String {
        if self.current_file_index == 0 {
            self.base_path.clone()
        } else {
            format!("{}.{}", self.base_path, self.current_file_index)
        }
    }
}

/// 结构化日志查询器
pub struct LogQuery {
    /// 时间范围开始
    pub start_time: Option<DateTime<Utc>>,
    
    /// 时间范围结束
    pub end_time: Option<DateTime<Utc>>,
    
    /// 日志级别过滤
    pub level: Option<String>,
    
    /// 服务名称过滤
    pub service: Option<String>,
    
    /// 操作名称过滤
    pub action: Option<String>,
    
    /// 用户ID过滤
    pub user_id: Option<String>,
    
    /// 追踪ID过滤
    pub trace_id: Option<String>,
    
    /// 自由文本搜索
    pub text_search: Option<String>,
    
    /// 结果限制
    pub limit: usize,
    
    /// 偏移量
    pub offset: usize,
}

impl LogQuery {
    /// 创建新的日志查询
    pub fn new() -> Self {
        Self {
            start_time: None,
            end_time: None,
            level: None,
            service: None,
            action: None,
            user_id: None,
            trace_id: None,
            text_search: None,
            limit: 100,
            offset: 0,
        }
    }

    /// 设置时间范围
    pub fn with_time_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.start_time = Some(start);
        self.end_time = Some(end);
        self
    }

    /// 设置日志级别过滤
    pub fn with_level(mut self, level: String) -> Self {
        self.level = Some(level);
        self
    }

    /// 设置分页
    pub fn with_pagination(mut self, limit: usize, offset: usize) -> Self {
        self.limit = limit;
        self.offset = offset;
        self
    }

    /// 匹配日志条目
    pub fn matches(&self, entry: &LogEntry) -> bool {
        // 检查时间范围
        if let Some(start) = &self.start_time {
            if entry.timestamp < *start {
                return false;
            }
        }
        
        if let Some(end) = &self.end_time {
            if entry.timestamp > *end {
                return false;
            }
        }

        // 检查日志级别
        if let Some(level) = &self.level {
            let entry_level = format!("{:?}", entry.level).to_uppercase();
            if entry_level != level.to_uppercase() {
                return false;
            }
        }

        // 检查服务名称
        if let Some(service) = &self.service {
            if entry.service != *service {
                return false;
            }
        }

        // 检查操作名称
        if let Some(action) = &self.action {
            if entry.action != *action {
                return false;
            }
        }

        // 检查用户ID
        if let Some(user_id) = &self.user_id {
            if entry.user_id.as_ref() != Some(user_id) {
                return false;
            }
        }

        // 检查追踪ID
        if let Some(trace_id) = &self.trace_id {
            if entry.trace_id != *trace_id {
                return false;
            }
        }

        // 检查自由文本搜索
        if let Some(text) = &self.text_search {
            let entry_json = serde_json::to_string(entry).unwrap_or_default();
            if !entry_json.to_lowercase().contains(&text.to_lowercase()) {
                return false;
            }
        }

        true
    }
}

impl Default for LogQuery {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_file_storage() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_str().unwrap();
        
        let mut storage = FileStorage::new(path).unwrap();
        let entry = LogEntry::new("test-service".to_string(), "test-action".to_string());
        
        storage.write_log(&entry).await.unwrap();
        storage.flush().await.unwrap();
        
        // 验证文件内容
        let content = std::fs::read_to_string(path).unwrap();
        assert!(content.contains("test-service"));
        assert!(content.contains("test-action"));
    }

    #[tokio::test]
    async fn test_memory_storage() {
        let mut storage = MemoryStorage::new(10);
        let entry = LogEntry::new("test-service".to_string(), "test-action".to_string());
        
        storage.write_log(&entry).await.unwrap();
        
        let entries = storage.get_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].service, "test-service");
    }

    #[tokio::test]
    async fn test_log_query() {
        let query = LogQuery::new()
            .with_level("INFO".to_string());
        
        let entry = LogEntry::new("test-service".to_string(), "test-action".to_string());
        assert!(query.matches(&entry));
    }

    #[test]
    fn test_log_rotator() {
        let mut rotator = LogRotator::new("test.log".to_string(), 1024, 3);
        
        assert!(!rotator.should_rotate(512));
        assert!(rotator.should_rotate(1024));
        
        let new_path = rotator.rotate().unwrap();
        assert_eq!(new_path, "test.log.1");
    }
}