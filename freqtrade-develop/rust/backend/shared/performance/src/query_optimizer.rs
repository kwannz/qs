use std::collections::HashMap;

/// Query optimization patterns and recommendations
pub struct QueryOptimizer {
    optimization_rules: HashMap<String, String>,
}

impl QueryOptimizer {
    pub fn new() -> Self {
        let mut rules = HashMap::new();
        
        // Common optimization rules
        rules.insert(
            "SELECT * FROM".to_string(),
            "Avoid SELECT *, specify only needed columns".to_string()
        );
        rules.insert(
            "ORDER BY without LIMIT".to_string(),
            "Add LIMIT clause to ORDER BY queries".to_string()
        );
        rules.insert(
            "Missing WHERE clause".to_string(),
            "Add WHERE clause to avoid full table scans".to_string()
        );
        rules.insert(
            "No index on JOIN".to_string(),
            "Ensure indexes exist on JOIN columns".to_string()
        );
        
        Self {
            optimization_rules: rules,
        }
    }
    
    /// Analyze query for optimization opportunities
    pub fn analyze_query(&self, query: &str) -> Vec<String> {
        let mut recommendations = Vec::new();
        let query_upper = query.to_uppercase();
        
        // Check for SELECT *
        if query_upper.contains("SELECT *") {
            recommendations.push(self.optimization_rules["SELECT * FROM"].clone());
        }
        
        // Check for ORDER BY without LIMIT
        if query_upper.contains("ORDER BY") && !query_upper.contains("LIMIT") {
            recommendations.push(self.optimization_rules["ORDER BY without LIMIT"].clone());
        }
        
        // Check for missing WHERE clause in SELECT statements
        if query_upper.starts_with("SELECT") && !query_upper.contains("WHERE") {
            recommendations.push(self.optimization_rules["Missing WHERE clause"].clone());
        }
        
        recommendations
    }
}

impl Default for QueryOptimizer {
    fn default() -> Self {
        Self::new()
    }
}