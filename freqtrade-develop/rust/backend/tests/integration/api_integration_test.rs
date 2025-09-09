// API集成测试
// Sprint 1 - 端到端测试

use anyhow::Result;
use reqwest::{Client, StatusCode};
use serde_json::json;
use tokio;
use uuid::Uuid;

const BASE_URL: &str = "http://localhost:8080/api/v1";

struct TestContext {
    client: Client,
    access_token: Option<String>,
    user_id: Option<String>,
}

impl TestContext {
    fn new() -> Self {
        Self {
            client: Client::new(),
            access_token: None,
            user_id: None,
        }
    }
    
    async fn login(&mut self, email: &str, password: &str) -> Result<()> {
        let response = self.client
            .post(&format!("{}/auth/login", BASE_URL))
            .json(&json!({
                "email": email,
                "password": password,
                "remember_me": false
            }))
            .send()
            .await?;
        
        assert_eq!(response.status(), StatusCode::OK);
        
        let body: serde_json::Value = response.json().await?;
        self.access_token = Some(body["data"]["access_token"].as_str().unwrap().to_string());
        self.user_id = Some(body["data"]["user"]["id"].as_str().unwrap().to_string());
        
        Ok(())
    }
    
    fn auth_header(&self) -> String {
        format!("Bearer {}", self.access_token.as_ref().unwrap())
    }
}

#[tokio::test]
async fn test_health_check() -> Result<()> {
    let client = Client::new();
    
    // 测试健康检查端点
    let response = client
        .get(&format!("{}/healthz", BASE_URL))
        .send()
        .await?;
    
    assert_eq!(response.status(), StatusCode::OK);
    
    let body: serde_json::Value = response.json().await?;
    assert_eq!(body["success"], true);
    assert_eq!(body["data"]["status"], "healthy");
    
    Ok(())
}

#[tokio::test]
async fn test_auth_flow() -> Result<()> {
    let mut ctx = TestContext::new();
    
    // 1. 测试登录
    ctx.login("test@example.com", "password123").await?;
    assert!(ctx.access_token.is_some());
    
    // 2. 测试获取用户信息
    let response = ctx.client
        .get(&format!("{}/auth/profile", BASE_URL))
        .header("Authorization", ctx.auth_header())
        .send()
        .await?;
    
    assert_eq!(response.status(), StatusCode::OK);
    
    let body: serde_json::Value = response.json().await?;
    assert_eq!(body["success"], true);
    assert_eq!(body["data"]["email"], "test@example.com");
    
    // 3. 测试Token刷新
    // TODO: 实现Token刷新测试
    
    // 4. 测试登出
    let response = ctx.client
        .post(&format!("{}/auth/logout", BASE_URL))
        .header("Authorization", ctx.auth_header())
        .json(&json!({}))
        .send()
        .await?;
    
    assert_eq!(response.status(), StatusCode::OK);
    
    Ok(())
}

#[tokio::test]
async fn test_order_lifecycle() -> Result<()> {
    let mut ctx = TestContext::new();
    ctx.login("test@example.com", "password123").await?;
    
    // 1. 创建订单
    let create_response = ctx.client
        .post(&format!("{}/orders", BASE_URL))
        .header("Authorization", ctx.auth_header())
        .json(&json!({
            "symbol": "BTCUSDT",
            "side": "BUY",
            "order_type": "LIMIT",
            "quantity": 0.01,
            "price": 45000,
            "time_in_force": "GTC"
        }))
        .send()
        .await?;
    
    assert_eq!(create_response.status(), StatusCode::CREATED);
    
    let order: serde_json::Value = create_response.json().await?;
    let order_id = order["data"]["id"].as_str().unwrap();
    
    // 2. 查询订单
    let get_response = ctx.client
        .get(&format!("{}/orders/{}", BASE_URL, order_id))
        .header("Authorization", ctx.auth_header())
        .send()
        .await?;
    
    assert_eq!(get_response.status(), StatusCode::OK);
    
    // 3. 修改订单
    let update_response = ctx.client
        .patch(&format!("{}/orders/{}", BASE_URL, order_id))
        .header("Authorization", ctx.auth_header())
        .json(&json!({
            "price": 44000
        }))
        .send()
        .await?;
    
    assert_eq!(update_response.status(), StatusCode::OK);
    
    // 4. 取消订单
    let cancel_response = ctx.client
        .delete(&format!("{}/orders/{}", BASE_URL, order_id))
        .header("Authorization", ctx.auth_header())
        .send()
        .await?;
    
    assert_eq!(cancel_response.status(), StatusCode::OK);
    
    let cancelled_order: serde_json::Value = cancel_response.json().await?;
    assert_eq!(cancelled_order["data"]["status"], "CANCELLED");
    
    Ok(())
}

#[tokio::test]
async fn test_market_data() -> Result<()> {
    let client = Client::new();
    
    // 1. 获取交易对列表
    let symbols_response = client
        .get(&format!("{}/symbols", BASE_URL))
        .send()
        .await?;
    
    assert_eq!(symbols_response.status(), StatusCode::OK);
    
    let symbols: serde_json::Value = symbols_response.json().await?;
    assert!(symbols["data"].as_array().unwrap().len() > 0);
    
    // 2. 获取K线数据
    let candles_response = client
        .get(&format!("{}/markets/BTCUSDT/candles?interval=1h&limit=10", BASE_URL))
        .send()
        .await?;
    
    assert_eq!(candles_response.status(), StatusCode::OK);
    
    let candles: serde_json::Value = candles_response.json().await?;
    assert!(candles["data"].as_array().unwrap().len() > 0);
    
    // 3. 获取订单簿
    let orderbook_response = client
        .get(&format!("{}/markets/BTCUSDT/orderbook?depth=20", BASE_URL))
        .send()
        .await?;
    
    assert_eq!(orderbook_response.status(), StatusCode::OK);
    
    let orderbook: serde_json::Value = orderbook_response.json().await?;
    assert!(orderbook["data"]["bids"].as_array().unwrap().len() > 0);
    assert!(orderbook["data"]["asks"].as_array().unwrap().len() > 0);
    
    // 4. 获取行情快照
    let ticker_response = client
        .get(&format!("{}/markets/BTCUSDT/ticker", BASE_URL))
        .send()
        .await?;
    
    assert_eq!(ticker_response.status(), StatusCode::OK);
    
    Ok(())
}

#[tokio::test]
async fn test_position_management() -> Result<()> {
    let mut ctx = TestContext::new();
    ctx.login("test@example.com", "password123").await?;
    
    // 1. 查询持仓
    let positions_response = ctx.client
        .get(&format!("{}/positions", BASE_URL))
        .header("Authorization", ctx.auth_header())
        .send()
        .await?;
    
    assert_eq!(positions_response.status(), StatusCode::OK);
    
    let positions: serde_json::Value = positions_response.json().await?;
    assert_eq!(positions["success"], true);
    
    // 2. 如果有持仓，测试平仓
    if let Some(positions_array) = positions["data"]["positions"].as_array() {
        if !positions_array.is_empty() {
            let symbol = positions_array[0]["symbol"].as_str().unwrap();
            
            let close_response = ctx.client
                .post(&format!("{}/positions/{}/close", BASE_URL, symbol))
                .header("Authorization", ctx.auth_header())
                .json(&json!({
                    "quantity": null,  // 全部平仓
                    "order_type": "MARKET"
                }))
                .send()
                .await?;
            
            assert_eq!(close_response.status(), StatusCode::OK);
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_strategy_management() -> Result<()> {
    let mut ctx = TestContext::new();
    ctx.login("test@example.com", "password123").await?;
    
    // 1. 创建策略
    let create_response = ctx.client
        .post(&format!("{}/strategies", BASE_URL))
        .header("Authorization", ctx.auth_header())
        .json(&json!({
            "name": "测试网格策略",
            "description": "集成测试用策略",
            "strategy_type": "GRID_TRADING",
            "parameters": {
                "grid_levels": 10,
                "grid_spacing": 100,
                "amount_per_grid": 1000
            }
        }))
        .send()
        .await?;
    
    assert_eq!(create_response.status(), StatusCode::CREATED);
    
    let strategy: serde_json::Value = create_response.json().await?;
    let strategy_id = strategy["data"]["id"].as_str().unwrap();
    
    // 2. 查询策略
    let get_response = ctx.client
        .get(&format!("{}/strategies/{}", BASE_URL, strategy_id))
        .header("Authorization", ctx.auth_header())
        .send()
        .await?;
    
    assert_eq!(get_response.status(), StatusCode::OK);
    
    // 3. 更新策略
    let update_response = ctx.client
        .put(&format!("{}/strategies/{}", BASE_URL, strategy_id))
        .header("Authorization", ctx.auth_header())
        .json(&json!({
            "description": "更新后的描述"
        }))
        .send()
        .await?;
    
    assert_eq!(update_response.status(), StatusCode::OK);
    
    // 4. 启动策略
    let start_response = ctx.client
        .post(&format!("{}/strategies/{}/start", BASE_URL, strategy_id))
        .header("Authorization", ctx.auth_header())
        .send()
        .await?;
    
    assert_eq!(start_response.status(), StatusCode::OK);
    
    // 5. 获取策略绩效
    let performance_response = ctx.client
        .get(&format!("{}/strategies/{}/performance", BASE_URL, strategy_id))
        .header("Authorization", ctx.auth_header())
        .send()
        .await?;
    
    assert_eq!(performance_response.status(), StatusCode::OK);
    
    // 6. 停止策略
    let stop_response = ctx.client
        .post(&format!("{}/strategies/{}/stop", BASE_URL, strategy_id))
        .header("Authorization", ctx.auth_header())
        .send()
        .await?;
    
    assert_eq!(stop_response.status(), StatusCode::OK);
    
    // 7. 删除策略
    let delete_response = ctx.client
        .delete(&format!("{}/strategies/{}", BASE_URL, strategy_id))
        .header("Authorization", ctx.auth_header())
        .send()
        .await?;
    
    assert_eq!(delete_response.status(), StatusCode::OK);
    
    Ok(())
}

#[tokio::test]
async fn test_rate_limiting() -> Result<()> {
    let client = Client::new();
    
    // 发送大量请求测试限流
    let mut responses = Vec::new();
    for _ in 0..150 {
        let response = client
            .get(&format!("{}/symbols", BASE_URL))
            .send()
            .await?;
        responses.push(response.status());
    }
    
    // 应该有一些请求返回429 (Too Many Requests)
    let rate_limited = responses.iter()
        .filter(|&&status| status == StatusCode::TOO_MANY_REQUESTS)
        .count();
    
    assert!(rate_limited > 0, "限流机制应该生效");
    
    Ok(())
}

#[tokio::test]
async fn test_error_handling() -> Result<()> {
    let mut ctx = TestContext::new();
    ctx.login("test@example.com", "password123").await?;
    
    // 1. 测试404错误
    let not_found_response = ctx.client
        .get(&format!("{}/orders/invalid-uuid", BASE_URL))
        .header("Authorization", ctx.auth_header())
        .send()
        .await?;
    
    assert_eq!(not_found_response.status(), StatusCode::NOT_FOUND);
    
    // 2. 测试400错误（无效输入）
    let bad_request_response = ctx.client
        .post(&format!("{}/orders", BASE_URL))
        .header("Authorization", ctx.auth_header())
        .json(&json!({
            "symbol": "BTCUSDT",
            "side": "INVALID_SIDE",  // 无效的买卖方向
            "order_type": "LIMIT",
            "quantity": -1,  // 无效的数量
            "price": 0
        }))
        .send()
        .await?;
    
    assert_eq!(bad_request_response.status(), StatusCode::BAD_REQUEST);
    
    // 3. 测试401错误（未授权）
    let unauthorized_response = ctx.client
        .get(&format!("{}/orders", BASE_URL))
        .send()
        .await?;
    
    assert_eq!(unauthorized_response.status(), StatusCode::UNAUTHORIZED);
    
    Ok(())
}

#[tokio::test]
async fn test_concurrent_requests() -> Result<()> {
    let mut ctx = TestContext::new();
    ctx.login("test@example.com", "password123").await?;
    
    // 并发创建多个订单
    let mut handles = Vec::new();
    let token = ctx.access_token.clone().unwrap();
    
    for i in 0..10 {
        let token_clone = token.clone();
        let handle = tokio::spawn(async move {
            let client = Client::new();
            let response = client
                .post(&format!("{}/orders", BASE_URL))
                .header("Authorization", format!("Bearer {}", token_clone))
                .json(&json!({
                    "symbol": "BTCUSDT",
                    "side": if i % 2 == 0 { "BUY" } else { "SELL" },
                    "order_type": "LIMIT",
                    "quantity": 0.01,
                    "price": 45000 + i * 100,
                    "time_in_force": "GTC",
                    "client_order_id": format!("test-order-{}", i)
                }))
                .send()
                .await;
            
            response
        });
        
        handles.push(handle);
    }
    
    // 等待所有请求完成
    let mut success_count = 0;
    for handle in handles {
        if let Ok(Ok(response)) = handle.await {
            if response.status() == StatusCode::CREATED {
                success_count += 1;
            }
        }
    }
    
    assert!(success_count > 0, "至少应该有一些请求成功");
    
    Ok(())
}

// WebSocket测试
#[tokio::test]
async fn test_websocket_connection() -> Result<()> {
    use tokio_tungstenite::{connect_async, tungstenite::Message};
    use futures_util::{SinkExt, StreamExt};
    
    let url = "ws://localhost:8080/ws";
    let (mut ws_stream, _) = connect_async(url).await?;
    
    // 订阅市场数据
    let subscribe_msg = json!({
        "method": "SUBSCRIBE",
        "params": ["btcusdt@kline_1m"],
        "id": 1
    });
    
    ws_stream.send(Message::Text(subscribe_msg.to_string())).await?;
    
    // 接收确认消息
    if let Some(msg) = ws_stream.next().await {
        let msg = msg?;
        if let Message::Text(text) = msg {
            let response: serde_json::Value = serde_json::from_str(&text)?;
            assert_eq!(response["result"], "subscribed");
        }
    }
    
    // 接收几条数据
    for _ in 0..3 {
        if let Some(msg) = ws_stream.next().await {
            let msg = msg?;
            if let Message::Text(text) = msg {
                let data: serde_json::Value = serde_json::from_str(&text)?;
                assert!(data["data"].is_object());
            }
        }
    }
    
    // 取消订阅
    let unsubscribe_msg = json!({
        "method": "UNSUBSCRIBE",
        "params": ["btcusdt@kline_1m"],
        "id": 2
    });
    
    ws_stream.send(Message::Text(unsubscribe_msg.to_string())).await?;
    
    // 关闭连接
    ws_stream.close(None).await?;
    
    Ok(())
}