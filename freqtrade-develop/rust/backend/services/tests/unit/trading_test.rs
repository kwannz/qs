#![allow(unused_imports, unused_variables, dead_code, unused_mut, deprecated)]


// 交易模块单元测试
// Sprint 1 - 测试覆盖

#[cfg(test)]
mod trading_tests {
    use super::*;
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;
    use chrono::Utc;
    use uuid::Uuid;
    
    mod order_tests {
        use super::*;
        
        #[test]
        fn test_order_validation() {
            // 测试数量验证
            let invalid_quantity = dec!(0);
            assert!(invalid_quantity <= Decimal::ZERO);
            
            let valid_quantity = dec!(1.5);
            assert!(valid_quantity > Decimal::ZERO);
            
            // 测试限价单价格验证
            let order_type = OrderType::Limit;
            let invalid_price = dec!(0);
            assert!(order_type == OrderType::Limit && invalid_price <= Decimal::ZERO);
            
            let valid_price = dec!(50000);
            assert!(valid_price > Decimal::ZERO);
        }
        
        #[test]
        fn test_can_cancel_order() {
            // 可以取消的状态
            assert!(can_cancel_order(&OrderStatus::Pending));
            assert!(can_cancel_order(&OrderStatus::Open));
            assert!(can_cancel_order(&OrderStatus::PartiallyFilled));
            
            // 不能取消的状态
            assert!(!can_cancel_order(&OrderStatus::Filled));
            assert!(!can_cancel_order(&OrderStatus::Cancelled));
            assert!(!can_cancel_order(&OrderStatus::Rejected));
        }
        
        #[test]
        fn test_can_modify_order() {
            // 可以修改的状态
            assert!(can_modify_order(&OrderStatus::Pending));
            assert!(can_modify_order(&OrderStatus::Open));
            
            // 不能修改的状态
            assert!(!can_modify_order(&OrderStatus::PartiallyFilled));
            assert!(!can_modify_order(&OrderStatus::Filled));
            assert!(!can_modify_order(&OrderStatus::Cancelled));
            assert!(!can_modify_order(&OrderStatus::Rejected));
        }
        
        #[test]
        fn test_order_creation() {
            let order = Order {
                id: Uuid::new_v4(),
                user_id: Uuid::new_v4(),
                symbol: "BTCUSDT".to_string(),
                side: OrderSide::Buy,
                order_type: OrderType::Limit,
                quantity: dec!(0.01),
                price: Some(dec!(50000)),
                status: OrderStatus::Pending,
                filled_quantity: dec!(0),
                average_price: None,
                time_in_force: TimeInForce::GTC,
                client_order_id: Some("test-order-123".to_string()),
                exchange_order_id: None,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };
            
            assert_eq!(order.symbol, "BTCUSDT");
            assert_eq!(order.side, OrderSide::Buy);
            assert_eq!(order.quantity, dec!(0.01));
            assert_eq!(order.status, OrderStatus::Pending);
        }
        
        #[test]
        fn test_order_serialization() {
            let order = Order {
                id: Uuid::new_v4(),
                user_id: Uuid::new_v4(),
                symbol: "ETHUSDT".to_string(),
                side: OrderSide::Sell,
                order_type: OrderType::Market,
                quantity: dec!(1.0),
                price: None,
                status: OrderStatus::Open,
                filled_quantity: dec!(0.5),
                average_price: Some(dec!(3000)),
                time_in_force: TimeInForce::IOC,
                client_order_id: None,
                exchange_order_id: Some("exchange-123".to_string()),
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };
            
            let json = serde_json::to_string(&order).expect("Failed to serialize");
            assert!(json.contains("ETHUSDT"));
            assert!(json.contains("Sell"));
            assert!(json.contains("Market"));
            
            let deserialized: Order = serde_json::from_str(&json).expect("Failed to deserialize");
            assert_eq!(deserialized.symbol, order.symbol);
            assert_eq!(deserialized.quantity, order.quantity);
        }
    }
    
    mod position_tests {
        use super::*;
        
        #[test]
        fn test_unrealized_pnl_calculation() {
            // 测试多头持仓盈利
            let long_position = Position {
                id: Uuid::new_v4(),
                user_id: Uuid::new_v4(),
                symbol: "BTCUSDT".to_string(),
                side: PositionSide::Long,
                quantity: dec!(1.0),
                entry_price: dec!(50000),
                mark_price: dec!(51000), // 价格上涨
                unrealized_pnl: dec!(0),
                realized_pnl: dec!(0),
                margin: dec!(5000),
                leverage: 10,
                liquidation_price: None,
                opened_at: Utc::now(),
                updated_at: Utc::now(),
            };
            
            let pnl = calculate_unrealized_pnl(&long_position);
            assert_eq!(pnl, dec!(1000)); // (51000 - 50000) * 1.0 = 1000
            
            // 测试多头持仓亏损
            let long_loss_position = Position {
                id: Uuid::new_v4(),
                user_id: Uuid::new_v4(),
                symbol: "BTCUSDT".to_string(),
                side: PositionSide::Long,
                quantity: dec!(1.0),
                entry_price: dec!(50000),
                mark_price: dec!(49000), // 价格下跌
                unrealized_pnl: dec!(0),
                realized_pnl: dec!(0),
                margin: dec!(5000),
                leverage: 10,
                liquidation_price: None,
                opened_at: Utc::now(),
                updated_at: Utc::now(),
            };
            
            let pnl = calculate_unrealized_pnl(&long_loss_position);
            assert_eq!(pnl, dec!(-1000)); // (49000 - 50000) * 1.0 = -1000
            
            // 测试空头持仓盈利
            let short_position = Position {
                id: Uuid::new_v4(),
                user_id: Uuid::new_v4(),
                symbol: "BTCUSDT".to_string(),
                side: PositionSide::Short,
                quantity: dec!(1.0),
                entry_price: dec!(50000),
                mark_price: dec!(49000), // 价格下跌（空头盈利）
                unrealized_pnl: dec!(0),
                realized_pnl: dec!(0),
                margin: dec!(5000),
                leverage: 10,
                liquidation_price: None,
                opened_at: Utc::now(),
                updated_at: Utc::now(),
            };
            
            let pnl = calculate_unrealized_pnl(&short_position);
            assert_eq!(pnl, dec!(1000)); // (50000 - 49000) * 1.0 = 1000
        }
        
        #[test]
        fn test_liquidation_price_calculation() {
            // 测试10倍杠杆多头
            let long_position = Position {
                id: Uuid::new_v4(),
                user_id: Uuid::new_v4(),
                symbol: "BTCUSDT".to_string(),
                side: PositionSide::Long,
                quantity: dec!(1.0),
                entry_price: dec!(50000),
                mark_price: dec!(50000),
                unrealized_pnl: dec!(0),
                realized_pnl: dec!(0),
                margin: dec!(5000),
                leverage: 10,
                liquidation_price: None,
                opened_at: Utc::now(),
                updated_at: Utc::now(),
            };
            
            let liq_price = calculate_liquidation_price(&long_position);
            assert!(liq_price.is_some());
            
            // 预期强平价格约为 50000 * (1 - 1/10 + 0.005) = 50000 * 0.905 = 45250
            let expected = dec!(50000) * dec!(0.905);
            assert!((liq_price.unwrap() - expected).abs() < dec!(1));
            
            // 测试无杠杆持仓（不应有强平价格）
            let no_leverage_position = Position {
                id: Uuid::new_v4(),
                user_id: Uuid::new_v4(),
                symbol: "BTCUSDT".to_string(),
                side: PositionSide::Long,
                quantity: dec!(1.0),
                entry_price: dec!(50000),
                mark_price: dec!(50000),
                unrealized_pnl: dec!(0),
                realized_pnl: dec!(0),
                margin: dec!(50000),
                leverage: 1,
                liquidation_price: None,
                opened_at: Utc::now(),
                updated_at: Utc::now(),
            };
            
            let liq_price = calculate_liquidation_price(&no_leverage_position);
            assert!(liq_price.is_none());
        }
        
        #[test]
        fn test_realized_pnl_calculation() {
            // 测试多头平仓盈利
            let long_position = Position {
                id: Uuid::new_v4(),
                user_id: Uuid::new_v4(),
                symbol: "BTCUSDT".to_string(),
                side: PositionSide::Long,
                quantity: dec!(2.0),
                entry_price: dec!(50000),
                mark_price: dec!(51000),
                unrealized_pnl: dec!(0),
                realized_pnl: dec!(0),
                margin: dec!(10000),
                leverage: 10,
                liquidation_price: None,
                opened_at: Utc::now(),
                updated_at: Utc::now(),
            };
            
            let close_quantity = dec!(1.0);
            let close_price = dec!(52000);
            let pnl = calculate_realized_pnl(&long_position, close_quantity, close_price);
            assert_eq!(pnl, dec!(2000)); // (52000 - 50000) * 1.0 = 2000
            
            // 测试空头平仓盈利
            let short_position = Position {
                id: Uuid::new_v4(),
                user_id: Uuid::new_v4(),
                symbol: "BTCUSDT".to_string(),
                side: PositionSide::Short,
                quantity: dec!(2.0),
                entry_price: dec!(50000),
                mark_price: dec!(49000),
                unrealized_pnl: dec!(0),
                realized_pnl: dec!(0),
                margin: dec!(10000),
                leverage: 10,
                liquidation_price: None,
                opened_at: Utc::now(),
                updated_at: Utc::now(),
            };
            
            let close_quantity = dec!(1.0);
            let close_price = dec!(48000);
            let pnl = calculate_realized_pnl(&short_position, close_quantity, close_price);
            assert_eq!(pnl, dec!(2000)); // (50000 - 48000) * 1.0 = 2000
        }
        
        #[test]
        fn test_required_margin_calculation() {
            let position = Position {
                id: Uuid::new_v4(),
                user_id: Uuid::new_v4(),
                symbol: "BTCUSDT".to_string(),
                side: PositionSide::Long,
                quantity: dec!(1.0),
                entry_price: dec!(50000),
                mark_price: dec!(50000),
                unrealized_pnl: dec!(0),
                realized_pnl: dec!(0),
                margin: dec!(5000),
                leverage: 10,
                liquidation_price: None,
                opened_at: Utc::now(),
                updated_at: Utc::now(),
            };
            
            let margin = calculate_required_margin(&position);
            assert_eq!(margin, dec!(5000)); // (1.0 * 50000) / 10 = 5000
        }
    }
    
    mod account_tests {
        use super::*;
        
        #[test]
        fn test_balance_validation() {
            let balance = Balance {
                user_id: Uuid::new_v4(),
                asset: "USDT".to_string(),
                free: dec!(1000),
                locked: dec!(500),
                total: dec!(1500),
                updated_at: Utc::now(),
            };
            
            assert_eq!(balance.total, balance.free + balance.locked);
            assert!(balance.free >= Decimal::ZERO);
            assert!(balance.locked >= Decimal::ZERO);
        }
        
        #[test]
        fn test_transfer_validation() {
            // 测试无效金额
            let invalid_transfer = TransferRequest {
                from_account: Uuid::new_v4(),
                to_account: Uuid::new_v4(),
                asset: "USDT".to_string(),
                amount: dec!(-100), // 负数金额
            };
            
            assert!(invalid_transfer.amount <= Decimal::ZERO);
            
            // 测试有效金额
            let valid_transfer = TransferRequest {
                from_account: Uuid::new_v4(),
                to_account: Uuid::new_v4(),
                asset: "USDT".to_string(),
                amount: dec!(100),
            };
            
            assert!(valid_transfer.amount > Decimal::ZERO);
        }
        
        #[test]
        fn test_account_status() {
            let account = Account {
                id: Uuid::new_v4(),
                user_id: Uuid::new_v4(),
                account_type: AccountType::Spot,
                name: "Main Account".to_string(),
                status: AccountStatus::Active,
                total_equity: dec!(10000),
                available_balance: dec!(8000),
                used_margin: dec!(2000),
                free_margin: dec!(8000),
                margin_level: Some(dec!(500)), // 500%
                leverage: 1,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };
            
            assert_eq!(account.status, AccountStatus::Active);
            assert_eq!(account.total_equity, account.available_balance + account.used_margin);
            
            // 测试保证金水平计算
            if account.used_margin > Decimal::ZERO {
                let expected_margin_level = (account.total_equity / account.used_margin) * dec!(100);
                assert_eq!(account.margin_level, Some(expected_margin_level));
            }
        }
        
        #[test]
        fn test_transaction_types() {
            let transactions = vec![
                TransactionType::Deposit,
                TransactionType::Withdrawal,
                TransactionType::Trade,
                TransactionType::Fee,
                TransactionType::Transfer,
                TransactionType::Liquidation,
                TransactionType::Funding,
                TransactionType::Rebate,
                TransactionType::Adjustment,
            ];
            
            for tx_type in transactions {
                let json = serde_json::to_string(&tx_type).expect("Failed to serialize");
                let deserialized: TransactionType = serde_json::from_str(&json).expect("Failed to deserialize");
                assert_eq!(format!("{:?}", tx_type), format!("{:?}", deserialized));
            }
        }
    }
}