#![allow(unused_imports, unused_variables, dead_code, unused_mut, deprecated)]


// 认证模块单元测试
// Sprint 1 - 测试覆盖

#[cfg(test)]
mod auth_tests {
    use super::*;
    use axum::http::{StatusCode, HeaderMap, HeaderValue};
    use chrono::{Utc, Duration};
    use jsonwebtoken::{encode, decode, Header, Validation, EncodingKey, DecodingKey};
    use uuid::Uuid;
    
    // 测试用的密钥
    const TEST_JWT_SECRET: &str = "test_secret_key_for_unit_tests_only";
    
    #[test]
    fn test_password_hashing_and_verification() {
        let password = "MySecurePassword123!";
        
        // 测试密码哈希
        let hash = hash_password(password).expect("Failed to hash password");
        assert!(!hash.is_empty());
        assert_ne!(hash, password); // 哈希值不应该等于原密码
        
        // 测试密码验证 - 正确密码
        assert!(verify_password(password, &hash).expect("Verification failed"));
        
        // 测试密码验证 - 错误密码
        assert!(!verify_password("WrongPassword", &hash).expect("Verification failed"));
        
        // 测试空密码
        let empty_hash = hash_password("").expect("Failed to hash empty password");
        assert!(verify_password("", &empty_hash).expect("Verification failed"));
    }
    
    #[test]
    fn test_jwt_token_generation_and_validation() {
        let user_id = Uuid::new_v4();
        let email = "test@example.com";
        let role = "user";
        
        // 创建Claims
        let now = Utc::now();
        let exp = now + Duration::hours(1);
        
        let claims = Claims {
            sub: user_id.to_string(),
            email: email.to_string(),
            role: role.to_string(),
            exp: exp.timestamp(),
            iat: now.timestamp(),
            jti: Uuid::new_v4().to_string(),
        };
        
        // 生成Token
        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(TEST_JWT_SECRET.as_bytes()),
        ).expect("Failed to encode token");
        
        assert!(!token.is_empty());
        
        // 解码并验证Token
        let decoded = decode::<Claims>(
            &token,
            &DecodingKey::from_secret(TEST_JWT_SECRET.as_bytes()),
            &Validation::default(),
        ).expect("Failed to decode token");
        
        assert_eq!(decoded.claims.sub, user_id.to_string());
        assert_eq!(decoded.claims.email, email);
        assert_eq!(decoded.claims.role, role);
    }
    
    #[test]
    fn test_token_expiration() {
        let user_id = Uuid::new_v4();
        
        // 创建已过期的Token
        let now = Utc::now();
        let expired_time = now - Duration::hours(1); // 1小时前过期
        
        let claims = Claims {
            sub: user_id.to_string(),
            email: "test@example.com".to_string(),
            role: "user".to_string(),
            exp: expired_time.timestamp(),
            iat: (expired_time - Duration::hours(2)).timestamp(),
            jti: Uuid::new_v4().to_string(),
        };
        
        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(TEST_JWT_SECRET.as_bytes()),
        ).expect("Failed to encode token");
        
        // 尝试解码过期的Token
        let result = decode::<Claims>(
            &token,
            &DecodingKey::from_secret(TEST_JWT_SECRET.as_bytes()),
            &Validation::default(),
        );
        
        assert!(result.is_err()); // 应该失败
    }
    
    #[test]
    fn test_token_rotation_logic() {
        let now = Utc::now();
        
        // 测试需要轮换的Token（已使用超过一半有效期）
        let old_claims = Claims {
            sub: "user123".to_string(),
            email: "test@example.com".to_string(),
            role: "user".to_string(),
            exp: (now + Duration::days(90)).timestamp(),
            iat: (now - Duration::days(50)).timestamp(), // 50天前签发
            jti: Uuid::new_v4().to_string(),
        };
        
        assert!(should_rotate_refresh_token(&old_claims));
        
        // 测试不需要轮换的Token（使用时间较短）
        let new_claims = Claims {
            sub: "user123".to_string(),
            email: "test@example.com".to_string(),
            role: "user".to_string(),
            exp: (now + Duration::days(90)).timestamp(),
            iat: (now - Duration::days(10)).timestamp(), // 10天前签发
            jti: Uuid::new_v4().to_string(),
        };
        
        assert!(!should_rotate_refresh_token(&new_claims));
    }
    
    #[test]
    fn test_extract_token_from_headers() {
        // 测试有效的Authorization header
        let mut headers = HeaderMap::new();
        headers.insert(
            "Authorization",
            HeaderValue::from_str("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9").unwrap(),
        );
        
        let token = extract_token_from_headers(&headers);
        assert!(token.is_ok());
        assert_eq!(token.unwrap(), "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9");
        
        // 测试缺少Bearer前缀
        let mut headers = HeaderMap::new();
        headers.insert(
            "Authorization",
            HeaderValue::from_str("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9").unwrap(),
        );
        
        let token = extract_token_from_headers(&headers);
        assert!(token.is_err());
        
        // 测试缺少Authorization header
        let headers = HeaderMap::new();
        let token = extract_token_from_headers(&headers);
        assert!(token.is_err());
    }
    
    #[tokio::test]
    async fn test_login_request_validation() {
        // 测试空邮箱
        let req = LoginRequest {
            email: "".to_string(),
            password: "password123".to_string(),
            remember_me: Some(false),
        };
        
        assert!(req.email.is_empty());
        
        // 测试空密码
        let req = LoginRequest {
            email: "test@example.com".to_string(),
            password: "".to_string(),
            remember_me: Some(false),
        };
        
        assert!(req.password.is_empty());
        
        // 测试有效请求
        let req = LoginRequest {
            email: "test@example.com".to_string(),
            password: "password123".to_string(),
            remember_me: Some(true),
        };
        
        assert!(!req.email.is_empty());
        assert!(!req.password.is_empty());
        assert_eq!(req.remember_me, Some(true));
    }
    
    #[test]
    fn test_different_hash_for_same_password() {
        let password = "TestPassword123";
        
        // 同一密码的两次哈希应该不同（因为使用了随机盐）
        let hash1 = hash_password(password).expect("Failed to hash password");
        let hash2 = hash_password(password).expect("Failed to hash password");
        
        assert_ne!(hash1, hash2);
        
        // 但两个哈希都应该能验证原密码
        assert!(verify_password(password, &hash1).expect("Verification failed"));
        assert!(verify_password(password, &hash2).expect("Verification failed"));
    }
    
    #[test]
    fn test_claims_serialization() {
        let claims = Claims {
            sub: "user123".to_string(),
            email: "test@example.com".to_string(),
            role: "admin".to_string(),
            exp: 1234567890,
            iat: 1234567800,
            jti: "unique-jwt-id".to_string(),
        };
        
        // 测试序列化
        let json = serde_json::to_string(&claims).expect("Failed to serialize");
        assert!(json.contains("user123"));
        assert!(json.contains("test@example.com"));
        assert!(json.contains("admin"));
        
        // 测试反序列化
        let deserialized: Claims = serde_json::from_str(&json).expect("Failed to deserialize");
        assert_eq!(deserialized.sub, claims.sub);
        assert_eq!(deserialized.email, claims.email);
        assert_eq!(deserialized.role, claims.role);
    }
}