use std::fmt;

/// Data validation utilities
pub struct ValidationUtils;

impl ValidationUtils {
    /// Validate email format
    pub fn is_valid_email(email: &str) -> bool {
        email.contains('@') && email.contains('.') && email.len() > 5
    }

    /// Validate if string is not empty and within length limits
    pub fn validate_string_length(input: &str, min_len: usize, max_len: usize) -> Result<(), ValidationError> {
        if input.is_empty() {
            return Err(ValidationError::EmptyInput);
        }
        if input.len() < min_len {
            return Err(ValidationError::TooShort { 
                actual: input.len(), 
                required: min_len 
            });
        }
        if input.len() > max_len {
            return Err(ValidationError::TooLong { 
                actual: input.len(), 
                limit: max_len 
            });
        }
        Ok(())
    }

    /// Validate numeric range
    pub fn validate_numeric_range<T>(value: T, min: T, max: T) -> Result<(), ValidationError>
    where
        T: PartialOrd + std::fmt::Display + Copy,
    {
        if value < min || value > max {
            return Err(ValidationError::OutOfRange {
                value: value.to_string(),
                min: min.to_string(),
                max: max.to_string(),
            });
        }
        Ok(())
    }

    /// Validate trading symbol format (e.g., "BTC-USD")
    pub fn validate_trading_symbol(symbol: &str) -> Result<(), ValidationError> {
        if symbol.is_empty() {
            return Err(ValidationError::EmptyInput);
        }
        if !symbol.contains('-') {
            return Err(ValidationError::InvalidFormat {
                field: "symbol".to_string(),
                expected: "BASE-QUOTE format (e.g., BTC-USD)".to_string(),
            });
        }
        let parts: Vec<&str> = symbol.split('-').collect();
        if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
            return Err(ValidationError::InvalidFormat {
                field: "symbol".to_string(),
                expected: "BASE-QUOTE format (e.g., BTC-USD)".to_string(),
            });
        }
        Ok(())
    }
}

/// Validation error types
#[derive(Debug)]
pub enum ValidationError {
    EmptyInput,
    TooShort { actual: usize, required: usize },
    TooLong { actual: usize, limit: usize },
    OutOfRange { value: String, min: String, max: String },
    InvalidFormat { field: String, expected: String },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::EmptyInput => write!(f, "Input cannot be empty"),
            ValidationError::TooShort { actual, required } => {
                write!(f, "Input too short: {} characters, requires at least {}", actual, required)
            }
            ValidationError::TooLong { actual, limit } => {
                write!(f, "Input too long: {} characters, maximum allowed is {}", actual, limit)
            }
            ValidationError::OutOfRange { value, min, max } => {
                write!(f, "Value {} is out of range [{}, {}]", value, min, max)
            }
            ValidationError::InvalidFormat { field, expected } => {
                write!(f, "Invalid format for {}: expected {}", field, expected)
            }
        }
    }
}

impl std::error::Error for ValidationError {}