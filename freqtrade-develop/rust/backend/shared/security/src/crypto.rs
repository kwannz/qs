use anyhow::{Context, Result};
use ring::{
    aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM},
    pbkdf2,
    rand::{SecureRandom, SystemRandom},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::num::NonZeroU32;

/// Encrypted data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedData {
    pub ciphertext: Vec<u8>,
    pub nonce: Vec<u8>,
    pub salt: Vec<u8>,
    pub algorithm: String,
}

/// Field-level encryption configuration
#[derive(Debug, Clone)]
pub struct FieldEncryptionConfig {
    pub master_key: Vec<u8>,
    pub key_derivation_iterations: u32,
}

impl Default for FieldEncryptionConfig {
    fn default() -> Self {
        Self {
            master_key: vec![0; 32], // This should be loaded from secure configuration
            key_derivation_iterations: 100_000,
        }
    }
}

/// Crypto manager for handling encryption/decryption operations
pub struct CryptoManager {
    config: FieldEncryptionConfig,
    rng: SystemRandom,
    field_keys: HashMap<String, Vec<u8>>, // Cached derived keys for fields
}

impl CryptoManager {
    /// Create new CryptoManager instance
    pub fn new(config: FieldEncryptionConfig) -> Result<Self> {
        Ok(Self {
            config,
            rng: SystemRandom::new(),
            field_keys: HashMap::new(),
        })
    }

    /// Encrypt sensitive field data
    pub fn encrypt_field(&mut self, field_name: &str, plaintext: &str) -> Result<EncryptedData> {
        // Generate a unique salt for this encryption
        let mut salt = vec![0u8; 16];
        self.rng.fill(&mut salt)
            .context("Failed to generate salt")?;

        // Derive key for this specific field
        let key = self.derive_field_key(field_name, &salt)?;
        
        // Create AEAD cipher
        let unbound_key = UnboundKey::new(&AES_256_GCM, &key)
            .context("Failed to create encryption key")?;
        let less_safe_key = LessSafeKey::new(unbound_key);

        // Generate nonce
        let mut nonce_bytes = vec![0u8; 12]; // AES-GCM nonce size
        self.rng.fill(&mut nonce_bytes)
            .context("Failed to generate nonce")?;
        let nonce = Nonce::assume_unique_for_key(
            nonce_bytes.as_slice().try_into()
                .context("Invalid nonce size")?
        );

        // Encrypt data
        let mut ciphertext = plaintext.as_bytes().to_vec();
        less_safe_key.seal_in_place_append_tag(
            nonce,
            Aad::empty(),
            &mut ciphertext,
        ).context("Failed to encrypt data")?;

        tracing::debug!(
            field_name = field_name,
            data_len = plaintext.len(),
            "Field data encrypted successfully"
        );

        Ok(EncryptedData {
            ciphertext,
            nonce: nonce_bytes,
            salt,
            algorithm: "AES-256-GCM".to_string(),
        })
    }

    /// Decrypt sensitive field data
    pub fn decrypt_field(&mut self, field_name: &str, encrypted: &EncryptedData) -> Result<String> {
        if encrypted.algorithm != "AES-256-GCM" {
            return Err(anyhow::anyhow!("Unsupported encryption algorithm: {}", encrypted.algorithm));
        }

        // Derive the same key using the stored salt
        let key = self.derive_field_key(field_name, &encrypted.salt)?;
        
        // Create AEAD cipher
        let unbound_key = UnboundKey::new(&AES_256_GCM, &key)
            .context("Failed to create decryption key")?;
        let less_safe_key = LessSafeKey::new(unbound_key);

        // Create nonce from stored bytes
        let nonce = Nonce::assume_unique_for_key(
            encrypted.nonce.as_slice().try_into()
                .context("Invalid nonce size")?
        );

        // Decrypt data
        let mut ciphertext = encrypted.ciphertext.clone();
        let plaintext = less_safe_key.open_in_place(nonce, Aad::empty(), &mut ciphertext)
            .context("Failed to decrypt data - data may be corrupted or key is incorrect")?;

        let result = String::from_utf8(plaintext.to_vec())
            .context("Decrypted data is not valid UTF-8")?;

        tracing::debug!(
            field_name = field_name,
            data_len = result.len(),
            "Field data decrypted successfully"
        );

        Ok(result)
    }

    /// Encrypt API key for secure storage
    pub fn encrypt_api_key(&mut self, api_key: &str) -> Result<EncryptedData> {
        self.encrypt_field("api_key", api_key)
    }

    /// Decrypt API key from storage
    pub fn decrypt_api_key(&mut self, encrypted: &EncryptedData) -> Result<String> {
        self.decrypt_field("api_key", encrypted)
    }

    /// Encrypt database connection string
    pub fn encrypt_connection_string(&mut self, connection_string: &str) -> Result<EncryptedData> {
        self.encrypt_field("db_connection", connection_string)
    }

    /// Decrypt database connection string
    pub fn decrypt_connection_string(&mut self, encrypted: &EncryptedData) -> Result<String> {
        self.decrypt_field("db_connection", encrypted)
    }

    /// Encrypt user personal information
    pub fn encrypt_pii(&mut self, field_name: &str, data: &str) -> Result<EncryptedData> {
        self.encrypt_field(&format!("pii_{}", field_name), data)
    }

    /// Decrypt user personal information
    pub fn decrypt_pii(&mut self, field_name: &str, encrypted: &EncryptedData) -> Result<String> {
        self.decrypt_field(&format!("pii_{}", field_name), encrypted)
    }

    /// Hash password using Argon2
    pub fn hash_password(&self, password: &str) -> Result<String> {
        use argon2::{Argon2, PasswordHasher};
        use argon2::password_hash::{rand_core::OsRng, SaltString};

        let salt = SaltString::generate(&mut OsRng);
        let argon2 = Argon2::default();
        
        let hash = argon2.hash_password(password.as_bytes(), &salt)
            .map_err(|e| anyhow::anyhow!("Failed to hash password: {}", e))?
            .to_string();

        Ok(hash)
    }

    /// Verify password against hash
    pub fn verify_password(&self, password: &str, hash: &str) -> Result<bool> {
        use argon2::{Argon2, PasswordVerifier};
        use argon2::password_hash::PasswordHash;

        let parsed_hash = PasswordHash::new(hash)
            .map_err(|e| anyhow::anyhow!("Failed to parse password hash: {}", e))?;
        
        let argon2 = Argon2::default();
        Ok(argon2.verify_password(password.as_bytes(), &parsed_hash).is_ok())
    }

    /// Generate secure random token
    pub fn generate_secure_token(&self, length: usize) -> Result<String> {
        let mut token_bytes = vec![0u8; length];
        self.rng.fill(&mut token_bytes)
            .context("Failed to generate random token")?;
        
        Ok(hex::encode(token_bytes))
    }

    /// Generate TOTP secret for 2FA
    pub fn generate_totp_secret(&self) -> Result<String> {
        let mut secret_bytes = vec![0u8; 20]; // 160 bits for TOTP
        self.rng.fill(&mut secret_bytes)
            .context("Failed to generate TOTP secret")?;
        
        Ok(base32::encode(base32::Alphabet::RFC4648 { padding: false }, &secret_bytes))
    }

    /// Derive field-specific encryption key
    fn derive_field_key(&mut self, field_name: &str, salt: &[u8]) -> Result<Vec<u8>> {
        // Check if we have a cached key for this field + salt combo
        let cache_key = format!("{}:{}", field_name, hex::encode(salt));
        if let Some(cached_key) = self.field_keys.get(&cache_key) {
            return Ok(cached_key.clone());
        }

        // Derive key using PBKDF2
        let iterations = NonZeroU32::new(self.config.key_derivation_iterations)
            .ok_or_else(|| anyhow::anyhow!("Invalid iteration count"))?;
        
        let mut derived_key = vec![0u8; 32]; // 256-bit key
        pbkdf2::derive(
            pbkdf2::PBKDF2_HMAC_SHA256,
            iterations,
            salt,
            &self.config.master_key,
            &mut derived_key,
        );

        // Cache the derived key
        self.field_keys.insert(cache_key, derived_key.clone());

        tracing::debug!(
            field_name = field_name,
            "Derived encryption key for field"
        );

        Ok(derived_key)
    }

    /// Rotate master key (requires re-encrypting all data)
    pub fn rotate_master_key(&mut self, new_master_key: Vec<u8>) -> Result<()> {
        if new_master_key.len() != 32 {
            return Err(anyhow::anyhow!("Master key must be exactly 32 bytes"));
        }

        // Clear cached keys as they're now invalid
        self.field_keys.clear();
        
        // Update master key
        self.config.master_key = new_master_key;

        tracing::warn!("Master key rotated - all encrypted data will need to be re-encrypted");

        Ok(())
    }

    /// Get encryption statistics
    pub fn get_stats(&self) -> CryptoStats {
        CryptoStats {
            cached_keys: self.field_keys.len(),
            algorithm: "AES-256-GCM".to_string(),
            key_derivation_algorithm: "PBKDF2-HMAC-SHA256".to_string(),
            iterations: self.config.key_derivation_iterations,
        }
    }
}

/// Encryption statistics
#[derive(Debug, Serialize)]
pub struct CryptoStats {
    pub cached_keys: usize,
    pub algorithm: String,
    pub key_derivation_algorithm: String,
    pub iterations: u32,
}

/// Initialize crypto subsystem
pub fn init_crypto() -> Result<()> {
    // Perform any global crypto initialization here
    tracing::info!("Crypto subsystem initialized");
    Ok(())
}

/// Utility functions for working with encrypted configuration
pub mod config {
    use super::{Serialize, CryptoManager, Result, Context, Deserialize, EncryptedData};
    use std::fs;
    use std::path::Path;

    /// Save encrypted configuration to file
    pub fn save_encrypted_config<T>(
        crypto_manager: &mut CryptoManager,
        config: &T,
        file_path: &Path,
        field_name: &str,
    ) -> Result<()>
    where
        T: Serialize,
    {
        let config_json = serde_json::to_string_pretty(config)
            .context("Failed to serialize configuration")?;

        let encrypted = crypto_manager.encrypt_field(field_name, &config_json)?;
        let encrypted_json = serde_json::to_string_pretty(&encrypted)
            .context("Failed to serialize encrypted data")?;

        fs::write(file_path, encrypted_json)
            .context("Failed to write encrypted configuration file")?;

        tracing::info!(
            file_path = ?file_path,
            field_name = field_name,
            "Configuration saved with encryption"
        );

        Ok(())
    }

    /// Load encrypted configuration from file
    pub fn load_encrypted_config<T>(
        crypto_manager: &mut CryptoManager,
        file_path: &Path,
        field_name: &str,
    ) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        let encrypted_json = fs::read_to_string(file_path)
            .context("Failed to read encrypted configuration file")?;

        let encrypted: EncryptedData = serde_json::from_str(&encrypted_json)
            .context("Failed to parse encrypted configuration")?;

        let config_json = crypto_manager.decrypt_field(field_name, &encrypted)?;
        let config: T = serde_json::from_str(&config_json)
            .context("Failed to parse decrypted configuration")?;

        tracing::info!(
            file_path = ?file_path,
            field_name = field_name,
            "Configuration loaded with decryption"
        );

        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_encryption_decryption() {
        let config = FieldEncryptionConfig::default();
        let mut crypto = CryptoManager::new(config).unwrap();

        let original_data = "sensitive_information_123";
        let field_name = "test_field";

        // Encrypt
        let encrypted = crypto.encrypt_field(field_name, original_data).unwrap();
        assert_eq!(encrypted.algorithm, "AES-256-GCM");
        assert!(!encrypted.ciphertext.is_empty());
        assert_eq!(encrypted.nonce.len(), 12);
        assert_eq!(encrypted.salt.len(), 16);

        // Decrypt
        let decrypted = crypto.decrypt_field(field_name, &encrypted).unwrap();
        assert_eq!(decrypted, original_data);
    }

    #[test]
    fn test_password_hashing() {
        let config = FieldEncryptionConfig::default();
        let crypto = CryptoManager::new(config).unwrap();

        let password = "secure_password_123";
        let hash = crypto.hash_password(password).unwrap();
        
        // Verify correct password
        assert!(crypto.verify_password(password, &hash).unwrap());
        
        // Verify incorrect password
        assert!(!crypto.verify_password("wrong_password", &hash).unwrap());
    }

    #[test]
    fn test_secure_token_generation() {
        let config = FieldEncryptionConfig::default();
        let crypto = CryptoManager::new(config).unwrap();

        let token1 = crypto.generate_secure_token(32).unwrap();
        let token2 = crypto.generate_secure_token(32).unwrap();

        assert_eq!(token1.len(), 64); // 32 bytes = 64 hex chars
        assert_eq!(token2.len(), 64);
        assert_ne!(token1, token2); // Should be different
    }

    #[test]
    fn test_api_key_encryption() {
        let config = FieldEncryptionConfig::default();
        let mut crypto = CryptoManager::new(config).unwrap();

        let api_key = "pak_1234567890abcdef";
        let encrypted = crypto.encrypt_api_key(api_key).unwrap();
        let decrypted = crypto.decrypt_api_key(&encrypted).unwrap();

        assert_eq!(decrypted, api_key);
    }
}