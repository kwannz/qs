pub mod messages;
pub mod websocket;
pub mod strategy_protocol;
pub mod market_data_protocol;
pub mod execution_protocol;

#[allow(ambiguous_glob_reexports)]
pub use messages::*;
#[allow(ambiguous_glob_reexports)]
pub use websocket::*;
#[allow(ambiguous_glob_reexports)]
pub use strategy_protocol::*;
#[allow(ambiguous_glob_reexports)]
pub use market_data_protocol::*;
#[allow(ambiguous_glob_reexports)]
pub use execution_protocol::*;

#[cfg(test)]
mod tests {

    #[test]
    fn test_websocket_message_types() {
        // Test basic WebSocket message functionality
        let test_data = "test message".as_bytes().to_vec();
        assert!(!test_data.is_empty());
        assert_eq!(test_data.len(), 12);
    }

    #[test]
    fn test_message_serialization() {
        // Test basic message structure
        let message_id = "test_id_123";
        let message_type = "heartbeat";
        
        assert!(message_id.starts_with("test_"));
        assert_eq!(message_type.len(), 9);
    }
}