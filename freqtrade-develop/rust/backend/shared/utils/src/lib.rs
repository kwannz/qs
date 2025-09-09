pub mod crypto;
pub mod validation;
pub mod error;
pub mod monitoring;
pub mod logging;

pub use monitoring::*;
pub use logging::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structured_logger_creation() {
        let logger = StructuredLogger::new(
            "test_service".to_string(),
            "1.0.0".to_string(),
            "test".to_string(),
        );
        
        let log_entry = logger.format_log("INFO", "Test message", None);
        assert!(log_entry["service"].as_str().unwrap() == "test_service");
        assert!(log_entry["level"].as_str().unwrap() == "INFO");
        assert!(log_entry["message"].as_str().unwrap() == "Test message");
    }

    #[test] 
    fn test_audit_logger_creation() {
        let audit_logger = AuditLogger::new(
            "test_service".to_string(),
            "1.0.0".to_string(),
            "test".to_string(),
        );
        
        // Test that audit logger is created without panic
        audit_logger.log_trade_signal("test_id", "BTCUSDT", "BUY", 0.8);
    }

    #[test]
    fn test_log_aggregator() {
        let mut aggregator = LogAggregator::new("test_service".to_string(), 10);
        let test_log = serde_json::json!({"test": "value"});
        
        aggregator.add_log(test_log.clone());
        let logs = aggregator.get_logs();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0], test_log);
    }
}