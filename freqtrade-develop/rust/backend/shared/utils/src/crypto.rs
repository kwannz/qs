use std::fmt;

/// Cryptographic utility functions
pub struct CryptoUtils;

impl CryptoUtils {
    /// Generate a secure random string of specified length
    pub fn generate_random_string(length: usize) -> String {
        use rand::Rng;
        const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ\
                                abcdefghijklmnopqrstuvwxyz\
                                0123456789";
        let mut rng = rand::thread_rng();
        
        (0..length)
            .map(|_| {
                let idx = rng.gen_range(0..CHARSET.len());
                CHARSET[idx] as char
            })
            .collect()
    }

    /// Hash a string using SHA-256
    pub fn hash_sha256(input: &str) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(input.as_bytes());
        format!("{:x}", hasher.finalize())
    }
}

/// Error type for crypto operations
#[derive(Debug)]
pub enum CryptoError {
    InvalidInput(String),
    HashingFailed(String),
}

impl fmt::Display for CryptoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CryptoError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            CryptoError::HashingFailed(msg) => write!(f, "Hashing failed: {}", msg),
        }
    }
}

impl std::error::Error for CryptoError {}