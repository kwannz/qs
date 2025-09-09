pub mod strategy_repository;
pub mod strategy_instance_repository;
pub mod strategy_execution_repository;
pub mod strategy_backtest_repository;

pub use strategy_repository::StrategyRepository;
pub use strategy_instance_repository::StrategyInstanceRepository;
pub use strategy_execution_repository::StrategyExecutionRepository;
pub use strategy_backtest_repository::StrategyBacktestRepository;

use sqlx::{Pool, Postgres, Transaction};
use anyhow::Result;

#[allow(dead_code)]
pub trait Repository<T> {
    async fn create(&self, entity: &T) -> Result<T>;
    async fn find_by_id(&self, id: uuid::Uuid) -> Result<Option<T>>;
    async fn update(&self, id: uuid::Uuid, entity: &T) -> Result<Option<T>>;
    async fn delete(&self, id: uuid::Uuid) -> Result<bool>;
    async fn list(&self, limit: Option<u32>, offset: Option<u32>) -> Result<Vec<T>>;
}

// Transaction Manager for complex operations
pub struct TransactionManager {
    pool: Pool<Postgres>,
}

impl TransactionManager {
    pub fn new(pool: Pool<Postgres>) -> Self {
        Self { pool }
    }

    pub async fn begin_transaction(&self) -> Result<Transaction<'_, Postgres>> {
        Ok(self.pool.begin().await?)
    }

    pub async fn commit_transaction(tx: Transaction<'_, Postgres>) -> Result<()> {
        tx.commit().await?;
        Ok(())
    }

    pub async fn rollback_transaction(tx: Transaction<'_, Postgres>) -> Result<()> {
        tx.rollback().await?;
        Ok(())
    }

    // Get a reference to the pool for manual transaction management
    pub fn pool(&self) -> &Pool<Postgres> {
        &self.pool
    }
}

#[allow(dead_code)]
pub fn get_database_pool() -> Option<&'static Pool<Postgres>> {
    static POOL: std::sync::OnceLock<Pool<Postgres>> = std::sync::OnceLock::new();
    POOL.get()
}

#[allow(dead_code)]
pub async fn initialize_database_pool(database_url: &str) -> Result<Pool<Postgres>> {
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(10)
        .connect(database_url)
        .await?;
    
    Ok(pool)
}