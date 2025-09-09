#![allow(unused_imports, unused_variables, dead_code, unused_mut, deprecated)]


// 数据管道单元测试
// Sprint 1 - 测试覆盖

#[cfg(test)]
mod data_pipeline_tests {
    use super::*;
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;
    use chrono::{Utc, Duration};
    
    mod collector_tests {
        use super::*;
        
        #[tokio::test]
        async fn test_mock_collector() {
            let config = DataSourceConfig {
                source: DataSource::Mock,
                enabled: true,
                symbols: vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],
                data_types: vec![DataType::Trade, DataType::Candle],
                api_key: None,
                api_secret: None,
                endpoint: None,
            };
            
            let collector = MockCollector::new(config).unwrap();
            
            // 测试启动
            collector.start().await.unwrap();
            let status = collector.status();
            assert!(status.is_running);
            
            // 测试数据采集
            let data = collector.collect().await.unwrap();
            assert!(!data.is_empty());
            assert_eq!(data.len(), 4); // 2 symbols * 2 data types
            
            // 验证数据内容
            for item in &data {
                assert_eq!(item.source, DataSource::Mock);
                assert!(["BTCUSDT", "ETHUSDT"].contains(&item.symbol.as_str()));
            }
            
            // 测试停止
            collector.stop().await.unwrap();
        }
        
        #[test]
        fn test_trade_data_serialization() {
            let trade = TradeData {
                id: "trade-123".to_string(),
                symbol: "BTCUSDT".to_string(),
                price: dec!(50000),
                quantity: dec!(0.5),
                side: TradeSide::Buy,
                timestamp: Utc::now(),
            };
            
            let json = serde_json::to_string(&trade).unwrap();
            assert!(json.contains("BTCUSDT"));
            assert!(json.contains("50000"));
            
            let deserialized: TradeData = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized.symbol, trade.symbol);
            assert_eq!(deserialized.price, trade.price);
        }
        
        #[test]
        fn test_candle_data_validation() {
            let candle = CandleData {
                symbol: "BTCUSDT".to_string(),
                interval: "1h".to_string(),
                open_time: Utc::now() - Duration::hours(1),
                close_time: Utc::now(),
                open: dec!(50000),
                high: dec!(51000),
                low: dec!(49500),
                close: dec!(50500),
                volume: dec!(100),
                quote_volume: dec!(5050000),
            };
            
            // 验证OHLC关系
            assert!(candle.high >= candle.open);
            assert!(candle.high >= candle.close);
            assert!(candle.low <= candle.open);
            assert!(candle.low <= candle.close);
            assert!(candle.high >= candle.low);
            
            // 验证时间关系
            assert!(candle.close_time > candle.open_time);
        }
        
        #[test]
        fn test_orderbook_data() {
            let orderbook = OrderBookData {
                symbol: "BTCUSDT".to_string(),
                timestamp: Utc::now(),
                bids: vec![
                    (dec!(49990), dec!(1.0)),
                    (dec!(49980), dec!(2.0)),
                    (dec!(49970), dec!(3.0)),
                ],
                asks: vec![
                    (dec!(50010), dec!(1.0)),
                    (dec!(50020), dec!(2.0)),
                    (dec!(50030), dec!(3.0)),
                ],
                last_update_id: 123456789,
            };
            
            // 验证买单价格递减
            for i in 1..orderbook.bids.len() {
                assert!(orderbook.bids[i-1].0 > orderbook.bids[i].0);
            }
            
            // 验证卖单价格递增
            for i in 1..orderbook.asks.len() {
                assert!(orderbook.asks[i-1].0 < orderbook.asks[i].0);
            }
            
            // 验证买卖价差
            if !orderbook.bids.is_empty() && !orderbook.asks.is_empty() {
                assert!(orderbook.asks[0].0 > orderbook.bids[0].0);
            }
        }
    }
    
    mod cleaner_tests {
        use super::*;
        
        #[test]
        fn test_price_validator() {
            let validator = PriceValidator::new(
                dec!(1),
                dec!(100000),
                dec!(10), // 10%最大变化
            );
            
            // 测试价格范围
            assert!(validator.validate(dec!(50000), None));
            assert!(!validator.validate(dec!(0), None)); // 低于最小值
            assert!(!validator.validate(dec!(200000), None)); // 高于最大值
            
            // 测试价格变化
            let prev_price = Some(dec!(50000));
            assert!(validator.validate(dec!(54000), prev_price)); // 8%变化，有效
            assert!(!validator.validate(dec!(60000), prev_price)); // 20%变化，无效
            assert!(validator.validate(dec!(46000), prev_price)); // -8%变化，有效
            assert!(!validator.validate(dec!(40000), prev_price)); // -20%变化，无效
        }
        
        #[test]
        fn test_volume_validator() {
            let validator = VolumeValidator::new(
                dec!(0),
                dec!(1000000),
            );
            
            assert!(validator.validate(dec!(100)));
            assert!(validator.validate(dec!(0))); // 允许0成交量
            assert!(!validator.validate(dec!(-1))); // 不允许负数
            assert!(!validator.validate(dec!(2000000))); // 超过最大值
        }
        
        #[test]
        fn test_timestamp_validator() {
            let validator = TimestampValidator::new(
                Duration::seconds(60), // 最大60秒延迟
                Duration::milliseconds(100), // 最小100ms间隔
            );
            
            let now = Utc::now();
            
            // 测试正常时间戳
            assert!(validator.validate(now, None));
            assert!(validator.validate(now - Duration::seconds(30), None)); // 30秒前，有效
            assert!(!validator.validate(now - Duration::seconds(120), None)); // 120秒前，无效
            
            // 测试时间间隔
            let prev = Some(now - Duration::seconds(1));
            assert!(validator.validate(now, prev)); // 1秒间隔，有效
            
            let prev = Some(now - Duration::milliseconds(50));
            assert!(!validator.validate(now, prev)); // 50ms间隔，太短
        }
        
        #[test]
        fn test_candle_cleaning() {
            let cleaner = MarketDataCleaner::new(CleanerConfig {
                rules: vec![],
                batch_size: 100,
                buffer_size: 1000,
                quality_threshold: 0.95,
                enable_validation: true,
            });
            
            // 创建有问题的K线数据
            let mut candle = CandleData {
                symbol: "BTCUSDT".to_string(),
                interval: "1m".to_string(),
                open_time: Utc::now() - Duration::minutes(1),
                close_time: Utc::now(),
                open: dec!(50000),
                high: dec!(49000), // 错误：高点低于开盘价
                low: dec!(51000),  // 错误：低点高于高点
                close: dec!(50500),
                volume: dec!(100),
                quote_volume: dec!(5000000),
            };
            
            let issues = cleaner.clean_candle(&mut candle);
            
            // 验证问题被检测到
            assert!(!issues.is_empty());
            assert!(issues.contains(&DataQualityIssue::Inconsistent));
            
            // 验证数据被修正
            assert!(candle.high >= candle.low);
            assert!(candle.high >= candle.open);
            assert!(candle.high >= candle.close);
            assert!(candle.low <= candle.open);
            assert!(candle.low <= candle.close);
        }
        
        #[test]
        fn test_trade_cleaning() {
            let cleaner = MarketDataCleaner::new(CleanerConfig {
                rules: vec![],
                batch_size: 100,
                buffer_size: 1000,
                quality_threshold: 0.95,
                enable_validation: true,
            });
            
            // 创建正常的交易数据
            let mut trade = TradeData {
                id: "trade-123".to_string(),
                symbol: "BTCUSDT".to_string(),
                price: dec!(50000),
                quantity: dec!(1.0),
                side: TradeSide::Buy,
                timestamp: Utc::now(),
            };
            
            let issues = cleaner.clean_trade(&mut trade, Some(dec!(49000)));
            assert!(issues.is_empty()); // 应该没有问题
            
            // 创建有问题的交易数据（价格异常）
            let mut bad_trade = TradeData {
                id: "trade-456".to_string(),
                symbol: "BTCUSDT".to_string(),
                price: dec!(100000), // 价格翻倍，异常
                quantity: dec!(1.0),
                side: TradeSide::Buy,
                timestamp: Utc::now(),
            };
            
            let issues = cleaner.clean_trade(&mut bad_trade, Some(dec!(50000)));
            assert!(!issues.is_empty());
            assert!(issues.contains(&DataQualityIssue::Anomaly));
        }
        
        #[test]
        fn test_data_quality_issues() {
            let issues = vec![
                DataQualityIssue::MissingValue,
                DataQualityIssue::OutOfRange,
                DataQualityIssue::Duplicate,
                DataQualityIssue::Inconsistent,
                DataQualityIssue::Stale,
                DataQualityIssue::Anomaly,
            ];
            
            for issue in issues {
                let json = serde_json::to_string(&issue).unwrap();
                let deserialized: DataQualityIssue = serde_json::from_str(&json).unwrap();
                assert_eq!(format!("{:?}", issue), format!("{:?}", deserialized));
            }
        }
    }
}