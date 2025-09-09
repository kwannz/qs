use sqlx::{Pool, Postgres, Row};
use uuid::Uuid;
use anyhow::Result;
use chrono::Utc;
use crate::models::{Strategy, StrategyType, StrategyStatus, CreateStrategyRequest, UpdateStrategyRequest};
use super::Repository;

pub struct StrategyRepository {
    pool: Pool<Postgres>,
}

impl StrategyRepository {
    pub fn new(pool: Pool<Postgres>) -> Self {
        Self { pool }
    }

    pub async fn find_by_type(&self, strategy_type: StrategyType) -> Result<Vec<Strategy>> {
        let type_str = strategy_type.to_string();
        let rows = sqlx::query(
            r#"
            SELECT id, name, description, strategy_type, code_version, strategy_code,
                   config_schema, default_params, max_position_size, max_daily_loss,
                   max_drawdown_percent, allowed_symbols, allowed_exchanges,
                   expected_return, max_leverage, min_capital, created_by,
                   is_active, is_public, tags, created_at, updated_at
            FROM strategies 
            WHERE strategy_type = $1 AND is_active = true
            ORDER BY created_at DESC
            "#,
        )
        .bind(&type_str)
        .fetch_all(&self.pool)
        .await?;

        let strategies: Result<Vec<Strategy>> = rows.into_iter().map(|row| {
            Ok(Strategy {
                id: row.get("id"),
                name: row.get("name"),
                description: row.get("description"),
                strategy_type: parse_strategy_type(row.get("strategy_type"))?,
                status: StrategyStatus::Draft, // Default status, will be updated from instances
                parameters: serde_json::from_value(row.get::<serde_json::Value, _>("default_params"))?,
                symbols: row.get::<Option<Vec<String>>, _>("allowed_symbols").unwrap_or_default(),
                exchanges: row.get::<Option<Vec<String>>, _>("allowed_exchanges").unwrap_or_default(),
                created_by: row.get("created_by"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
                last_execution: None, // Will be populated from execution data if needed
            })
        }).collect();

        strategies
    }

    pub async fn find_by_status(&self, _status: StrategyStatus) -> Result<Vec<Strategy>> {
        // This requires joining with strategy_instances table to get actual status
        let rows = sqlx::query(
            r#"
            SELECT DISTINCT s.id, s.name, s.description, s.strategy_type, s.code_version,
                   s.strategy_code, s.config_schema, s.default_params, s.max_position_size,
                   s.max_daily_loss, s.max_drawdown_percent, s.allowed_symbols,
                   s.allowed_exchanges, s.expected_return, s.max_leverage, s.min_capital,
                   s.created_by, s.is_active, s.is_public, s.tags, s.created_at, s.updated_at,
                   si.status as instance_status, si.last_execution
            FROM strategies s
            LEFT JOIN strategy_instances si ON s.id = si.strategy_id
            WHERE s.is_active = true
            ORDER BY s.created_at DESC
            "#,
        )
        .fetch_all(&self.pool)
        .await?;

        let strategies: Result<Vec<Strategy>> = rows.into_iter().map(|row| {
            Ok(Strategy {
                id: row.get("id"),
                name: row.get("name"),
                description: row.get("description"),
                strategy_type: parse_strategy_type(row.get("strategy_type"))?,
                status: parse_strategy_status(
                    row.get::<Option<String>, _>("instance_status")
                        .unwrap_or_else(|| "draft".to_string())
                )?,
                parameters: serde_json::from_value(row.get::<serde_json::Value, _>("default_params"))?,
                symbols: row.get::<Option<Vec<String>>, _>("allowed_symbols").unwrap_or_default(),
                exchanges: row.get::<Option<Vec<String>>, _>("allowed_exchanges").unwrap_or_default(),
                created_by: row.get("created_by"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
                last_execution: row.get("last_execution"),
            })
        }).collect();

        strategies
    }

    pub async fn search(&self, query: &str, limit: Option<u32>, offset: Option<u32>) -> Result<Vec<Strategy>> {
        let limit = limit.unwrap_or(50).min(100) as i64;
        let offset = offset.unwrap_or(0) as i64;

        let rows = sqlx::query(
            r#"
            SELECT id, name, description, strategy_type, code_version, strategy_code,
                   config_schema, default_params, max_position_size, max_daily_loss,
                   max_drawdown_percent, allowed_symbols, allowed_exchanges,
                   expected_return, max_leverage, min_capital, created_by,
                   is_active, is_public, tags, created_at, updated_at,
                   ts_rank(search_vector, plainto_tsquery('english', $1)) as rank
            FROM strategies 
            WHERE search_vector @@ plainto_tsquery('english', $1)
              AND is_active = true
            ORDER BY rank DESC, created_at DESC
            LIMIT $2 OFFSET $3
            "#,
        )
        .bind(query)
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;

        let strategies: Result<Vec<Strategy>> = rows.into_iter().map(|row| {
            Ok(Strategy {
                id: row.get("id"),
                name: row.get("name"),
                description: row.get("description"),
                strategy_type: parse_strategy_type(row.get("strategy_type"))?,
                status: StrategyStatus::Draft, // Default status
                parameters: serde_json::from_value(row.get::<serde_json::Value, _>("default_params"))?,
                symbols: row.get::<Option<Vec<String>>, _>("allowed_symbols").unwrap_or_default(),
                exchanges: row.get::<Option<Vec<String>>, _>("allowed_exchanges").unwrap_or_default(),
                created_by: row.get("created_by"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
                last_execution: None,
            })
        }).collect();

        strategies
    }

    pub async fn create_from_request(&self, request: CreateStrategyRequest, created_by: &str) -> Result<Strategy> {
        let strategy_id = Uuid::new_v4();
        let now = Utc::now();
        let strategy_type_str = request.strategy_type.to_string();
        let params_json = serde_json::to_value(&request.parameters)?;

        sqlx::query(
            r#"
            INSERT INTO strategies (
                id, name, description, strategy_type, default_params,
                allowed_symbols, allowed_exchanges, created_by, is_active,
                created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            "#,
        )
        .bind(strategy_id)
        .bind(&request.name)
        .bind(&request.description)
        .bind(&strategy_type_str)
        .bind(&params_json)
        .bind(&request.symbols)
        .bind(&request.exchanges)
        .bind(created_by)
        .bind(true)
        .bind(now)
        .bind(now)
        .execute(&self.pool)
        .await?;

        Ok(Strategy {
            id: strategy_id,
            name: request.name,
            description: request.description,
            strategy_type: request.strategy_type,
            status: StrategyStatus::Draft,
            parameters: request.parameters,
            symbols: request.symbols,
            exchanges: request.exchanges,
            created_by: created_by.to_string(),
            created_at: now,
            updated_at: now,
            last_execution: None,
        })
    }

    pub async fn update_from_request(&self, id: Uuid, request: UpdateStrategyRequest) -> Result<Option<Strategy>> {
        // Simple approach: build static queries for each combination
        let now = Utc::now();
        
        // Check if nothing to update
        if request.name.is_none() && request.description.is_none() && 
           request.parameters.is_none() && request.symbols.is_none() && 
           request.exchanges.is_none() {
            return self.find_by_id(id).await;
        }

        // For simplicity, do individual updates for each field that's present
        if let Some(name) = &request.name {
            sqlx::query("UPDATE strategies SET name = $1, updated_at = $2 WHERE id = $3 AND is_active = true")
                .bind(name)
                .bind(now)
                .bind(id)
                .execute(&self.pool)
                .await?;
        }
        
        if let Some(description) = &request.description {
            sqlx::query("UPDATE strategies SET description = $1, updated_at = $2 WHERE id = $3 AND is_active = true")
                .bind(description)
                .bind(now)
                .bind(id)
                .execute(&self.pool)
                .await?;
        }
        
        if let Some(parameters) = &request.parameters {
            let params_json = serde_json::to_value(parameters)?;
            sqlx::query("UPDATE strategies SET default_params = $1, updated_at = $2 WHERE id = $3 AND is_active = true")
                .bind(params_json)
                .bind(now)
                .bind(id)
                .execute(&self.pool)
                .await?;
        }
        
        if let Some(symbols) = &request.symbols {
            sqlx::query("UPDATE strategies SET allowed_symbols = $1, updated_at = $2 WHERE id = $3 AND is_active = true")
                .bind(symbols)
                .bind(now)
                .bind(id)
                .execute(&self.pool)
                .await?;
        }
        
        if let Some(exchanges) = &request.exchanges {
            sqlx::query("UPDATE strategies SET allowed_exchanges = $1, updated_at = $2 WHERE id = $3 AND is_active = true")
                .bind(exchanges)
                .bind(now)
                .bind(id)
                .execute(&self.pool)
                .await?;
        }

        // Return updated strategy
        self.find_by_id(id).await
    }
}

impl Repository<Strategy> for StrategyRepository {
    async fn create(&self, strategy: &Strategy) -> Result<Strategy> {
        let params_json = serde_json::to_value(&strategy.parameters)?;
        let strategy_type_str = strategy.strategy_type.to_string();

        sqlx::query(
            r#"
            INSERT INTO strategies (
                id, name, description, strategy_type, default_params,
                allowed_symbols, allowed_exchanges, created_by, is_active,
                created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            "#,
        )
        .bind(strategy.id)
        .bind(&strategy.name)
        .bind(&strategy.description)
        .bind(&strategy_type_str)
        .bind(&params_json)
        .bind(&strategy.symbols)
        .bind(&strategy.exchanges)
        .bind(&strategy.created_by)
        .bind(true)
        .bind(strategy.created_at)
        .bind(strategy.updated_at)
        .execute(&self.pool)
        .await?;

        Ok(strategy.clone())
    }

    async fn find_by_id(&self, id: Uuid) -> Result<Option<Strategy>> {
        let row = sqlx::query(
            r#"
            SELECT id, name, description, strategy_type, code_version, strategy_code,
                   config_schema, default_params, max_position_size, max_daily_loss,
                   max_drawdown_percent, allowed_symbols, allowed_exchanges,
                   expected_return, max_leverage, min_capital, created_by,
                   is_active, is_public, tags, created_at, updated_at
            FROM strategies 
            WHERE id = $1 AND is_active = true
            "#,
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;

        match row {
            Some(row) => Ok(Some(Strategy {
                id: row.get("id"),
                name: row.get("name"),
                description: row.get("description"),
                strategy_type: parse_strategy_type(row.get("strategy_type"))?,
                status: StrategyStatus::Draft,
                parameters: serde_json::from_value(row.get::<serde_json::Value, _>("default_params"))?,
                symbols: row.get::<Option<Vec<String>>, _>("allowed_symbols").unwrap_or_default(),
                exchanges: row.get::<Option<Vec<String>>, _>("allowed_exchanges").unwrap_or_default(),
                created_by: row.get("created_by"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
                last_execution: None,
            })),
            None => Ok(None),
        }
    }

    async fn update(&self, id: Uuid, strategy: &Strategy) -> Result<Option<Strategy>> {
        let params_json = serde_json::to_value(&strategy.parameters)?;
        let strategy_type_str = strategy.strategy_type.to_string();

        let result = sqlx::query(
            r#"
            UPDATE strategies SET 
                name = $2, description = $3, strategy_type = $4,
                default_params = $5, allowed_symbols = $6, allowed_exchanges = $7,
                updated_at = $8
            WHERE id = $1 AND is_active = true
            "#,
        )
        .bind(id)
        .bind(&strategy.name)
        .bind(&strategy.description)
        .bind(&strategy_type_str)
        .bind(&params_json)
        .bind(&strategy.symbols)
        .bind(&strategy.exchanges)
        .bind(Utc::now())
        .execute(&self.pool)
        .await?;

        if result.rows_affected() > 0 {
            self.find_by_id(id).await
        } else {
            Ok(None)
        }
    }

    async fn delete(&self, id: Uuid) -> Result<bool> {
        let result = sqlx::query(
            "UPDATE strategies SET is_active = false, updated_at = $2 WHERE id = $1 AND is_active = true"
        )
        .bind(id)
        .bind(Utc::now())
        .execute(&self.pool)
        .await?;

        Ok(result.rows_affected() > 0)
    }

    async fn list(&self, limit: Option<u32>, offset: Option<u32>) -> Result<Vec<Strategy>> {
        let limit = limit.unwrap_or(50).min(100) as i64;
        let offset = offset.unwrap_or(0) as i64;

        let rows = sqlx::query(
            r#"
            SELECT id, name, description, strategy_type, code_version, strategy_code,
                   config_schema, default_params, max_position_size, max_daily_loss,
                   max_drawdown_percent, allowed_symbols, allowed_exchanges,
                   expected_return, max_leverage, min_capital, created_by,
                   is_active, is_public, tags, created_at, updated_at
            FROM strategies 
            WHERE is_active = true
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
            "#,
        )
        .bind(limit)
        .bind(offset)
        .fetch_all(&self.pool)
        .await?;

        let strategies: Result<Vec<Strategy>> = rows.into_iter().map(|row| {
            Ok(Strategy {
                id: row.get("id"),
                name: row.get("name"),
                description: row.get("description"),
                strategy_type: parse_strategy_type(row.get("strategy_type"))?,
                status: StrategyStatus::Draft,
                parameters: serde_json::from_value(row.get::<serde_json::Value, _>("default_params"))?,
                symbols: row.get::<Option<Vec<String>>, _>("allowed_symbols").unwrap_or_default(),
                exchanges: row.get::<Option<Vec<String>>, _>("allowed_exchanges").unwrap_or_default(),
                created_by: row.get("created_by"),
                created_at: row.get("created_at"),
                updated_at: row.get("updated_at"),
                last_execution: None,
            })
        }).collect();

        strategies
    }
}

fn parse_strategy_type(type_str: String) -> Result<StrategyType> {
    match type_str.as_str() {
        "momentum" => Ok(StrategyType::Momentum),
        "mean_reversion" => Ok(StrategyType::MeanReversion),
        "arbitrage" => Ok(StrategyType::Arbitrage),
        "market_making" => Ok(StrategyType::GridTrading), // Map market_making to GridTrading
        "grid" => Ok(StrategyType::GridTrading),
        "dca" => Ok(StrategyType::Dca),
        "pairs_trading" => Ok(StrategyType::PairsTrading),
        "ml_prediction" => Ok(StrategyType::MLPrediction),
        "factor_based" => Ok(StrategyType::FactorBased),
        "custom" => Ok(StrategyType::Custom),
        _ => Ok(StrategyType::Custom),
    }
}

fn parse_strategy_status(status_str: String) -> Result<StrategyStatus> {
    match status_str.as_str() {
        "created" | "draft" => Ok(StrategyStatus::Draft),
        "validating" | "ready" => Ok(StrategyStatus::Draft),
        "running" | "active" => Ok(StrategyStatus::Active),
        "paused" => Ok(StrategyStatus::Paused),
        "stopping" | "stopped" => Ok(StrategyStatus::Stopped),
        "error" => Ok(StrategyStatus::Error),
        "backtesting" => Ok(StrategyStatus::Backtesting),
        _ => Ok(StrategyStatus::Draft),
    }
}