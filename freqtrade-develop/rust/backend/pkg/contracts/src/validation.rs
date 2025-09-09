use crate::{ContractError, ContractResult, ContractValidator};
use anyhow::Result;

/// Validation utilities for contract data
pub struct Validator;

impl Validator {
    /// Validate symbol format
    pub fn validate_symbol(symbol: &str) -> ContractResult<()> {
        if symbol.is_empty() {
            return Err(ContractError::Validation("Symbol cannot be empty".to_string()));
        }
        
        if !symbol.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            return Err(ContractError::Validation(
                "Symbol contains invalid characters".to_string(),
            ));
        }
        
        Ok(())
    }
    
    /// Validate price value
    pub fn validate_price(price: f64) -> ContractResult<()> {
        if price < 0.0 {
            return Err(ContractError::Validation("Price cannot be negative".to_string()));
        }
        
        if !price.is_finite() {
            return Err(ContractError::Validation("Price must be finite".to_string()));
        }
        
        Ok(())
    }
    
    /// Validate quantity value
    pub fn validate_quantity(quantity: f64) -> ContractResult<()> {
        if quantity <= 0.0 {
            return Err(ContractError::Validation("Quantity must be positive".to_string()));
        }
        
        if !quantity.is_finite() {
            return Err(ContractError::Validation("Quantity must be finite".to_string()));
        }
        
        Ok(())
    }
    
    /// Validate timestamp
    pub fn validate_timestamp(timestamp: i64) -> ContractResult<()> {
        if timestamp <= 0 {
            return Err(ContractError::Validation("Timestamp must be positive".to_string()));
        }
        
        Ok(())
    }
}

// Implement validation for generated signal type
impl ContractValidator for crate::generated::Signal {
    fn validate(&self) -> Result<()> {
        Validator::validate_symbol(&self.symbol)?;
        
        // Validate timestamp if present
        if let Some(ref timestamp) = self.timestamp {
            let timestamp_secs = timestamp.seconds;
            Validator::validate_timestamp(timestamp_secs)?;
        }
        
        // Validate strength and confidence values
        if !self.strength.is_finite() {
            return Err(ContractError::Validation("Signal strength must be finite".to_string()).into());
        }
        
        if self.confidence < 0.0 || self.confidence > 1.0 {
            return Err(ContractError::Validation("Signal confidence must be between 0 and 1".to_string()).into());
        }
        
        Ok(())
    }
}

// Implement validation for Order type
impl ContractValidator for crate::generated::Order {
    fn validate(&self) -> Result<()> {
        Validator::validate_symbol(&self.symbol)?;
        Validator::validate_quantity(self.quantity)?;
        
        if let Some(price) = self.price {
            Validator::validate_price(price)?;
        }
        
        if let Some(stop_price) = self.stop_price {
            Validator::validate_price(stop_price)?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_symbol_validation() {
        assert!(Validator::validate_symbol("BTCUSDT").is_ok());
        assert!(Validator::validate_symbol("BTC_USD").is_ok());
        assert!(Validator::validate_symbol("BTC-USD").is_ok());
        assert!(Validator::validate_symbol("").is_err());
        assert!(Validator::validate_symbol("BTC@USD").is_err());
    }
    
    #[test]
    fn test_price_validation() {
        assert!(Validator::validate_price(100.0).is_ok());
        assert!(Validator::validate_price(0.0).is_ok());
        assert!(Validator::validate_price(-1.0).is_err());
        assert!(Validator::validate_price(f64::INFINITY).is_err());
        assert!(Validator::validate_price(f64::NAN).is_err());
    }
    
    #[test]
    fn test_quantity_validation() {
        assert!(Validator::validate_quantity(1.0).is_ok());
        assert!(Validator::validate_quantity(0.1).is_ok());
        assert!(Validator::validate_quantity(0.0).is_err());
        assert!(Validator::validate_quantity(-1.0).is_err());
        assert!(Validator::validate_quantity(f64::INFINITY).is_err());
    }
}