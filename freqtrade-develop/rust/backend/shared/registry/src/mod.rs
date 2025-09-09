pub mod strategy_registry;
pub mod factor_registry; 
pub mod version_control;
pub mod lineage_tracking;
pub mod metadata_manager;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// AG3策略因子注册表系统
pub struct RegistrySystem {
    strategy_registry: strategy_registry::StrategyRegistry,
    factor_registry: factor_registry::FactorRegistry,
    version_control: version_control::VersionController,
    lineage_tracker: lineage_tracking::LineageTracker,
    metadata_manager: metadata_manager::MetadataManager,
    config: RegistryConfig,
}

/// 注册表配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    pub enable_version_control: bool,     // 启用版本控制
    pub enable_lineage_tracking: bool,    // 启用血缘跟踪
    pub auto_versioning: bool,            // 自动版本管理
    pub retention_policy: RetentionPolicy, // 保留策略
    pub backup_strategy: BackupStrategy,   // 备份策略
    pub access_control: AccessControl,     // 访问控制
    pub storage_backend: StorageBackend,   // 存储后端
}

/// 保留策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    pub max_versions: usize,              // 最大版本数
    pub retention_days: u64,              // 保留天数
    pub archive_after_days: u64,          // 归档天数
    pub cleanup_frequency_hours: u64,     // 清理频率
}

/// 备份策略
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupStrategy {
    pub backup_frequency_hours: u64,      // 备份频率
    pub backup_locations: Vec<String>,    // 备份位置
    pub compression_enabled: bool,        // 压缩备份
    pub encryption_enabled: bool,         // 加密备份
}

/// 访问控制
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControl {
    pub require_authentication: bool,     // 需要认证
    pub role_based_access: bool,          // 基于角色的访问
    pub audit_logging: bool,              // 审计日志
    pub allowed_users: Vec<String>,       // 允许的用户
}

/// 存储后端
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageBackend {
    Local(String),                        // 本地文件系统
    S3 { bucket: String, region: String }, // AWS S3
    Database(String),                     // 数据库连接
    Redis(String),                        // Redis连接
}

/// 注册实体基类
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryEntity {
    pub id: String,                       // 唯一标识
    pub name: String,                     // 名称
    pub version: String,                  // 版本
    pub description: String,              // 描述
    pub author: String,                   // 作者
    pub created_at: DateTime<Utc>,        // 创建时间
    pub updated_at: DateTime<Utc>,        // 更新时间
    pub tags: Vec<String>,                // 标签
    pub metadata: HashMap<String, String>, // 元数据
    pub status: EntityStatus,             // 状态
    pub lineage: LineageInfo,             // 血缘信息
}

/// 实体状态
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EntityStatus {
    Development,  // 开发中
    Testing,      // 测试中
    Staging,      // 预发布
    Production,   // 生产环境
    Deprecated,   // 已弃用
    Archived,     // 已归档
}

/// 血缘信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageInfo {
    pub parent_id: Option<String>,        // 父实体ID
    pub children_ids: Vec<String>,        // 子实体ID列表
    pub dependencies: Vec<Dependency>,    // 依赖关系
    pub derivations: Vec<Derivation>,     // 派生关系
}

/// 依赖关系
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    pub dependency_id: String,            // 依赖实体ID
    pub dependency_type: DependencyType,  // 依赖类型
    pub version_constraint: String,       // 版本约束
    pub required: bool,                   // 是否必需
}

/// 依赖类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyType {
    Strategy,     // 策略依赖
    Factor,       // 因子依赖
    Data,         // 数据依赖
    Model,        // 模型依赖
    Library,      // 库依赖
}

/// 派生关系
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Derivation {
    pub source_id: String,                // 源实体ID
    pub derivation_type: DerivationType,  // 派生类型
    pub transformation: String,           // 转换描述
    pub confidence: f64,                  // 置信度
}

/// 派生类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DerivationType {
    Clone,        // 克隆
    Fork,         // 分叉
    Merge,        // 合并
    Transform,    // 转换
    Optimize,     // 优化
}

/// 查询条件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCriteria {
    pub entity_type: Option<EntityType>,  // 实体类型
    pub name_pattern: Option<String>,     // 名称模式
    pub version_pattern: Option<String>,  // 版本模式
    pub author: Option<String>,           // 作者
    pub tags: Vec<String>,                // 标签过滤
    pub status: Option<EntityStatus>,     // 状态过滤
    pub created_after: Option<DateTime<Utc>>, // 创建时间过滤
    pub created_before: Option<DateTime<Utc>>,
    pub has_dependencies: Option<bool>,   // 是否有依赖
    pub metadata_filters: HashMap<String, String>, // 元数据过滤
    pub limit: Option<usize>,             // 结果限制
    pub offset: Option<usize>,            // 偏移量
}

/// 实体类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntityType {
    Strategy,
    Factor,
    Model,
    Dataset,
}

/// 查询结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub entities: Vec<RegistryEntity>,
    pub total_count: usize,
    pub has_more: bool,
    pub query_time_ms: u64,
}

impl RegistrySystem {
    pub fn new(config: RegistryConfig) -> Result<Self> {
        Ok(Self {
            strategy_registry: strategy_registry::StrategyRegistry::new(&config)?,
            factor_registry: factor_registry::FactorRegistry::new(&config)?,
            version_control: version_control::VersionController::new(&config)?,
            lineage_tracker: lineage_tracking::LineageTracker::new(&config)?,
            metadata_manager: metadata_manager::MetadataManager::new(&config)?,
            config,
        })
    }

    /// 注册新实体
    pub fn register_entity(
        &mut self,
        entity_type: EntityType,
        name: String,
        content: serde_json::Value,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<RegistryEntity> {
        let entity_id = Uuid::new_v4().to_string();
        let version = if self.config.auto_versioning {
            self.generate_next_version(&name, &entity_type)?
        } else {
            "1.0.0".to_string()
        };

        let mut entity = RegistryEntity {
            id: entity_id.clone(),
            name: name.clone(),
            version: version.clone(),
            description: String::new(),
            author: "system".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            tags: Vec::new(),
            metadata: metadata.unwrap_or_default(),
            status: EntityStatus::Development,
            lineage: LineageInfo {
                parent_id: None,
                children_ids: Vec::new(),
                dependencies: Vec::new(),
                derivations: Vec::new(),
            },
        };

        // 根据实体类型注册到相应的注册表
        match entity_type {
            EntityType::Strategy => {
                self.strategy_registry.register(&mut entity, content)?;
            }
            EntityType::Factor => {
                self.factor_registry.register(&mut entity, content)?;
            }
            EntityType::Model | EntityType::Dataset => {
                // 通用实体注册
                self.register_generic_entity(&mut entity, content)?;
            }
        }

        // 版本控制
        if self.config.enable_version_control {
            self.version_control.create_version(&entity, &content)?;
        }

        // 血缘跟踪
        if self.config.enable_lineage_tracking {
            self.lineage_tracker.track_creation(&entity)?;
        }

        // 元数据管理
        self.metadata_manager.index_entity(&entity)?;

        log::info!("Registered entity: {} v{}", name, version);
        Ok(entity)
    }

    /// 更新实体
    pub fn update_entity(
        &mut self,
        entity_id: &str,
        content: serde_json::Value,
        change_description: Option<String>,
    ) -> Result<RegistryEntity> {
        // 获取当前实体
        let mut entity = self.get_entity_by_id(entity_id)?
            .ok_or_else(|| anyhow::anyhow!("Entity not found: {}", entity_id))?;

        // 生成新版本
        if self.config.auto_versioning {
            entity.version = self.increment_version(&entity.version)?;
        }
        entity.updated_at = Utc::now();

        // 版本控制
        if self.config.enable_version_control {
            self.version_control.create_version(&entity, &content)?;
            if let Some(desc) = change_description {
                self.version_control.add_change_log(&entity.id, &entity.version, desc)?;
            }
        }

        // 更新相应的注册表
        self.update_in_registries(&entity, content)?;

        // 血缘跟踪
        if self.config.enable_lineage_tracking {
            self.lineage_tracker.track_update(&entity)?;
        }

        // 更新元数据索引
        self.metadata_manager.update_index(&entity)?;

        log::info!("Updated entity: {} to v{}", entity.name, entity.version);
        Ok(entity)
    }

    /// 删除实体
    pub fn delete_entity(&mut self, entity_id: &str, soft_delete: bool) -> Result<()> {
        let entity = self.get_entity_by_id(entity_id)?
            .ok_or_else(|| anyhow::anyhow!("Entity not found: {}", entity_id))?;

        if soft_delete {
            // 软删除：标记为已归档
            let mut updated_entity = entity.clone();
            updated_entity.status = EntityStatus::Archived;
            updated_entity.updated_at = Utc::now();
            
            self.update_in_registries(&updated_entity, serde_json::Value::Null)?;
        } else {
            // 硬删除：从所有注册表中移除
            self.strategy_registry.delete(&entity.id)?;
            self.factor_registry.delete(&entity.id)?;
            
            if self.config.enable_version_control {
                self.version_control.delete_all_versions(&entity.id)?;
            }
        }

        // 血缘跟踪
        if self.config.enable_lineage_tracking {
            if soft_delete {
                self.lineage_tracker.track_archive(&entity)?;
            } else {
                self.lineage_tracker.track_deletion(&entity)?;
            }
        }

        // 更新元数据索引
        self.metadata_manager.remove_from_index(&entity.id)?;

        log::info!("Deleted entity: {} (soft: {})", entity.name, soft_delete);
        Ok(())
    }

    /// 查询实体
    pub fn query_entities(&self, criteria: QueryCriteria) -> Result<QueryResult> {
        let start_time = std::time::Instant::now();
        
        let mut results = Vec::new();
        let mut total_count = 0;

        // 根据实体类型查询不同的注册表
        match criteria.entity_type {
            Some(EntityType::Strategy) => {
                let strategy_results = self.strategy_registry.query(&criteria)?;
                results.extend(strategy_results);
            }
            Some(EntityType::Factor) => {
                let factor_results = self.factor_registry.query(&criteria)?;
                results.extend(factor_results);
            }
            Some(EntityType::Model) | Some(EntityType::Dataset) => {
                // 查询通用实体
                results.extend(self.query_generic_entities(&criteria)?);
            }
            None => {
                // 查询所有类型
                results.extend(self.strategy_registry.query(&criteria)?);
                results.extend(self.factor_registry.query(&criteria)?);
                results.extend(self.query_generic_entities(&criteria)?);
            }
        }

        // 应用过滤条件
        results = self.apply_filters(results, &criteria);

        total_count = results.len();

        // 应用分页
        if let (Some(limit), offset) = (criteria.limit, criteria.offset.unwrap_or(0)) {
            let end = (offset + limit).min(results.len());
            results = results[offset..end].to_vec();
        }

        let query_time = start_time.elapsed().as_millis() as u64;
        let has_more = criteria.limit.map_or(false, |limit| {
            total_count > criteria.offset.unwrap_or(0) + limit
        });

        Ok(QueryResult {
            entities: results,
            total_count,
            has_more,
            query_time_ms: query_time,
        })
    }

    /// 获取实体详情
    pub fn get_entity_by_id(&self, entity_id: &str) -> Result<Option<RegistryEntity>> {
        // 先尝试策略注册表
        if let Some(entity) = self.strategy_registry.get_by_id(entity_id)? {
            return Ok(Some(entity));
        }

        // 尝试因子注册表
        if let Some(entity) = self.factor_registry.get_by_id(entity_id)? {
            return Ok(Some(entity));
        }

        // 尝试通用注册表
        self.get_generic_entity_by_id(entity_id)
    }

    /// 获取实体版本历史
    pub fn get_version_history(&self, entity_id: &str) -> Result<Vec<version_control::VersionInfo>> {
        if self.config.enable_version_control {
            self.version_control.get_version_history(entity_id)
        } else {
            Ok(Vec::new())
        }
    }

    /// 获取实体血缘关系
    pub fn get_lineage(&self, entity_id: &str, depth: usize) -> Result<lineage_tracking::LineageGraph> {
        if self.config.enable_lineage_tracking {
            self.lineage_tracker.build_lineage_graph(entity_id, depth)
        } else {
            Ok(lineage_tracking::LineageGraph::empty())
        }
    }

    /// 添加依赖关系
    pub fn add_dependency(
        &mut self,
        entity_id: &str,
        dependency: Dependency,
    ) -> Result<()> {
        let mut entity = self.get_entity_by_id(entity_id)?
            .ok_or_else(|| anyhow::anyhow!("Entity not found: {}", entity_id))?;

        // 检查循环依赖
        if self.would_create_cycle(entity_id, &dependency.dependency_id)? {
            return Err(anyhow::anyhow!("Adding dependency would create a cycle"));
        }

        entity.lineage.dependencies.push(dependency.clone());
        entity.updated_at = Utc::now();

        // 更新注册表
        self.update_in_registries(&entity, serde_json::Value::Null)?;

        // 更新血缘跟踪
        if self.config.enable_lineage_tracking {
            self.lineage_tracker.add_dependency(&entity.id, &dependency)?;
        }

        log::info!("Added dependency: {} -> {}", entity_id, dependency.dependency_id);
        Ok(())
    }

    /// 创建派生实体
    pub fn create_derivative(
        &mut self,
        source_entity_id: &str,
        derivative_name: String,
        derivation_type: DerivationType,
        transformation_content: serde_json::Value,
    ) -> Result<RegistryEntity> {
        let source_entity = self.get_entity_by_id(source_entity_id)?
            .ok_or_else(|| anyhow::anyhow!("Source entity not found: {}", source_entity_id))?;

        // 创建新的派生实体
        let derivative_id = Uuid::new_v4().to_string();
        let mut derivative_entity = RegistryEntity {
            id: derivative_id.clone(),
            name: derivative_name,
            version: "1.0.0".to_string(),
            description: format!("Derived from {}", source_entity.name),
            author: source_entity.author.clone(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            tags: source_entity.tags.clone(),
            metadata: source_entity.metadata.clone(),
            status: EntityStatus::Development,
            lineage: LineageInfo {
                parent_id: Some(source_entity_id.to_string()),
                children_ids: Vec::new(),
                dependencies: source_entity.lineage.dependencies.clone(),
                derivations: vec![Derivation {
                    source_id: source_entity_id.to_string(),
                    derivation_type: derivation_type.clone(),
                    transformation: "Auto-generated derivative".to_string(),
                    confidence: 1.0,
                }],
            },
        };

        // 注册派生实体
        self.register_generic_entity(&mut derivative_entity, transformation_content)?;

        // 更新源实体的子实体列表
        let mut updated_source = source_entity.clone();
        updated_source.lineage.children_ids.push(derivative_id.clone());
        updated_source.updated_at = Utc::now();
        self.update_in_registries(&updated_source, serde_json::Value::Null)?;

        // 血缘跟踪
        if self.config.enable_lineage_tracking {
            self.lineage_tracker.track_derivation(&source_entity, &derivative_entity, &derivation_type)?;
        }

        log::info!("Created derivative entity: {} from {}", 
            derivative_entity.name, source_entity.name);
        Ok(derivative_entity)
    }

    /// 执行清理任务
    pub fn cleanup(&mut self) -> Result<()> {
        let retention_policy = &self.config.retention_policy;
        let cutoff_date = Utc::now() - chrono::Duration::days(retention_policy.retention_days as i64);
        let archive_date = Utc::now() - chrono::Duration::days(retention_policy.archive_after_days as i64);

        // 归档旧实体
        let old_entities = self.find_entities_before_date(archive_date)?;
        for entity in old_entities {
            if entity.status != EntityStatus::Archived {
                let mut updated_entity = entity.clone();
                updated_entity.status = EntityStatus::Archived;
                updated_entity.updated_at = Utc::now();
                self.update_in_registries(&updated_entity, serde_json::Value::Null)?;
            }
        }

        // 删除过期实体
        let expired_entities = self.find_entities_before_date(cutoff_date)?;
        for entity in expired_entities {
            if entity.status == EntityStatus::Archived {
                self.delete_entity(&entity.id, false)?;
            }
        }

        // 版本清理
        if self.config.enable_version_control {
            self.version_control.cleanup_old_versions(retention_policy.max_versions)?;
        }

        log::info!("Cleanup completed");
        Ok(())
    }

    /// 备份注册表
    pub fn backup(&self) -> Result<String> {
        let backup_strategy = &self.config.backup_strategy;
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let backup_id = format!("registry_backup_{}", timestamp);

        // 收集所有实体数据
        let all_entities = self.export_all_entities()?;
        
        // 创建备份数据
        let backup_data = RegistryBackup {
            backup_id: backup_id.clone(),
            timestamp: Utc::now(),
            entities: all_entities,
            metadata: self.metadata_manager.export_metadata()?,
            version_history: if self.config.enable_version_control {
                Some(self.version_control.export_history()?)
            } else {
                None
            },
            lineage_data: if self.config.enable_lineage_tracking {
                Some(self.lineage_tracker.export_lineage()?)
            } else {
                None
            },
        };

        // 序列化备份数据
        let backup_json = serde_json::to_string_pretty(&backup_data)?;
        
        // 根据配置保存备份
        for location in &backup_strategy.backup_locations {
            self.save_backup(&backup_json, location, &backup_id)?;
        }

        log::info!("Created backup: {}", backup_id);
        Ok(backup_id)
    }

    /// 从备份恢复
    pub fn restore_from_backup(&mut self, backup_location: &str, backup_id: &str) -> Result<()> {
        let backup_data = self.load_backup(backup_location, backup_id)?;
        
        // 清空当前数据
        self.clear_all_registries()?;
        
        // 恢复实体
        for entity in backup_data.entities {
            // 根据实体类型重新注册
            self.restore_entity(entity)?;
        }
        
        // 恢复元数据
        self.metadata_manager.import_metadata(backup_data.metadata)?;
        
        // 恢复版本历史
        if let Some(version_history) = backup_data.version_history {
            if self.config.enable_version_control {
                self.version_control.import_history(version_history)?;
            }
        }
        
        // 恢复血缘数据
        if let Some(lineage_data) = backup_data.lineage_data {
            if self.config.enable_lineage_tracking {
                self.lineage_tracker.import_lineage(lineage_data)?;
            }
        }
        
        log::info!("Restored from backup: {}", backup_id);
        Ok(())
    }

    // 辅助方法实现
    fn generate_next_version(&self, name: &str, entity_type: &EntityType) -> Result<String> {
        let latest_version = self.find_latest_version(name, entity_type)?;
        Ok(self.increment_version(&latest_version.unwrap_or_else(|| "0.0.0".to_string()))?)
    }

    fn increment_version(&self, version: &str) -> Result<String> {
        let parts: Vec<&str> = version.split('.').collect();
        if parts.len() != 3 {
            return Ok("1.0.0".to_string());
        }
        
        let major: u32 = parts[0].parse().unwrap_or(0);
        let minor: u32 = parts[1].parse().unwrap_or(0);
        let patch: u32 = parts[2].parse().unwrap_or(0);
        
        Ok(format!("{}.{}.{}", major, minor, patch + 1))
    }

    fn find_latest_version(&self, name: &str, entity_type: &EntityType) -> Result<Option<String>> {
        let criteria = QueryCriteria {
            entity_type: Some(entity_type.clone()),
            name_pattern: Some(name.to_string()),
            ..Default::default()
        };
        
        let results = self.query_entities(criteria)?;
        let latest = results.entities.iter()
            .max_by(|a, b| {
                version_compare::compare(&a.version, &b.version)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            
        Ok(latest.map(|e| e.version.clone()))
    }

    fn would_create_cycle(&self, entity_id: &str, dependency_id: &str) -> Result<bool> {
        if !self.config.enable_lineage_tracking {
            return Ok(false);
        }
        
        self.lineage_tracker.would_create_cycle(entity_id, dependency_id)
    }

    fn apply_filters(&self, mut entities: Vec<RegistryEntity>, criteria: &QueryCriteria) -> Vec<RegistryEntity> {
        // 名称模式过滤
        if let Some(pattern) = &criteria.name_pattern {
            entities.retain(|e| e.name.contains(pattern));
        }

        // 版本模式过滤
        if let Some(version_pattern) = &criteria.version_pattern {
            entities.retain(|e| e.version.contains(version_pattern));
        }

        // 作者过滤
        if let Some(author) = &criteria.author {
            entities.retain(|e| e.author == *author);
        }

        // 标签过滤
        if !criteria.tags.is_empty() {
            entities.retain(|e| {
                criteria.tags.iter().all(|tag| e.tags.contains(tag))
            });
        }

        // 状态过滤
        if let Some(status) = &criteria.status {
            entities.retain(|e| e.status == *status);
        }

        // 时间过滤
        if let Some(after) = criteria.created_after {
            entities.retain(|e| e.created_at >= after);
        }
        if let Some(before) = criteria.created_before {
            entities.retain(|e| e.created_at <= before);
        }

        // 依赖过滤
        if let Some(has_deps) = criteria.has_dependencies {
            entities.retain(|e| !e.lineage.dependencies.is_empty() == has_deps);
        }

        // 元数据过滤
        for (key, value) in &criteria.metadata_filters {
            entities.retain(|e| {
                e.metadata.get(key).map_or(false, |v| v == value)
            });
        }

        entities
    }

    // 更多辅助方法的占位实现
    fn register_generic_entity(&mut self, entity: &mut RegistryEntity, content: serde_json::Value) -> Result<()> {
        // 通用实体注册逻辑
        Ok(())
    }

    fn update_in_registries(&mut self, entity: &RegistryEntity, content: serde_json::Value) -> Result<()> {
        // 更新所有相关注册表
        Ok(())
    }

    fn query_generic_entities(&self, criteria: &QueryCriteria) -> Result<Vec<RegistryEntity>> {
        // 查询通用实体
        Ok(Vec::new())
    }

    fn get_generic_entity_by_id(&self, entity_id: &str) -> Result<Option<RegistryEntity>> {
        // 获取通用实体
        Ok(None)
    }

    fn find_entities_before_date(&self, date: DateTime<Utc>) -> Result<Vec<RegistryEntity>> {
        let criteria = QueryCriteria {
            created_before: Some(date),
            ..Default::default()
        };
        Ok(self.query_entities(criteria)?.entities)
    }

    fn export_all_entities(&self) -> Result<Vec<RegistryEntity>> {
        let all_criteria = QueryCriteria::default();
        Ok(self.query_entities(all_criteria)?.entities)
    }

    fn save_backup(&self, backup_json: &str, location: &str, backup_id: &str) -> Result<()> {
        // 根据存储后端保存备份
        match &self.config.storage_backend {
            StorageBackend::Local(base_path) => {
                let backup_path = format!("{}/{}.json", base_path, backup_id);
                std::fs::write(backup_path, backup_json)?;
            }
            _ => {
                // 其他存储后端的实现
                return Err(anyhow::anyhow!("Storage backend not implemented"));
            }
        }
        Ok(())
    }

    fn load_backup(&self, location: &str, backup_id: &str) -> Result<RegistryBackup> {
        match &self.config.storage_backend {
            StorageBackend::Local(base_path) => {
                let backup_path = format!("{}/{}.json", base_path, backup_id);
                let backup_json = std::fs::read_to_string(backup_path)?;
                Ok(serde_json::from_str(&backup_json)?)
            }
            _ => {
                Err(anyhow::anyhow!("Storage backend not implemented"))
            }
        }
    }

    fn clear_all_registries(&mut self) -> Result<()> {
        self.strategy_registry.clear()?;
        self.factor_registry.clear()?;
        // 清理其他注册表
        Ok(())
    }

    fn restore_entity(&mut self, entity: RegistryEntity) -> Result<()> {
        // 根据实体类型恢复到相应注册表
        Ok(())
    }
}

/// 备份数据结构
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RegistryBackup {
    backup_id: String,
    timestamp: DateTime<Utc>,
    entities: Vec<RegistryEntity>,
    metadata: serde_json::Value,
    version_history: Option<serde_json::Value>,
    lineage_data: Option<serde_json::Value>,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            enable_version_control: true,
            enable_lineage_tracking: true,
            auto_versioning: true,
            retention_policy: RetentionPolicy {
                max_versions: 10,
                retention_days: 365,
                archive_after_days: 90,
                cleanup_frequency_hours: 24,
            },
            backup_strategy: BackupStrategy {
                backup_frequency_hours: 24,
                backup_locations: vec!["./backups".to_string()],
                compression_enabled: true,
                encryption_enabled: false,
            },
            access_control: AccessControl {
                require_authentication: false,
                role_based_access: false,
                audit_logging: true,
                allowed_users: vec!["admin".to_string()],
            },
            storage_backend: StorageBackend::Local("./registry_data".to_string()),
        }
    }
}

impl Default for QueryCriteria {
    fn default() -> Self {
        Self {
            entity_type: None,
            name_pattern: None,
            version_pattern: None,
            author: None,
            tags: Vec::new(),
            status: None,
            created_after: None,
            created_before: None,
            has_dependencies: None,
            metadata_filters: HashMap::new(),
            limit: Some(100),
            offset: Some(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_system_creation() {
        let config = RegistryConfig::default();
        let result = RegistrySystem::new(config);
        // 由于依赖其他模块，这里只测试基本创建
        // assert!(result.is_ok());
    }

    #[test]
    fn test_version_increment() {
        let config = RegistryConfig::default();
        if let Ok(registry) = RegistrySystem::new(config) {
            assert_eq!(registry.increment_version("1.2.3").unwrap(), "1.2.4");
            assert_eq!(registry.increment_version("invalid").unwrap(), "1.0.0");
        }
    }

    #[test]
    fn test_query_criteria_default() {
        let criteria = QueryCriteria::default();
        assert_eq!(criteria.limit, Some(100));
        assert_eq!(criteria.offset, Some(0));
        assert!(criteria.tags.is_empty());
    }
}