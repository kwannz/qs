use anyhow::Result;
use object_pool::{Pool, Reusable};
use platform_config::PlatformConfig;
use std::sync::{Arc, OnceLock};
use tracing::info;

/// Memory pool for frequently allocated objects
pub struct MemoryManager {
    string_pool: Pool<String>,
    vec_pool: Pool<Vec<u8>>,
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryManager {
    pub fn new() -> Self {
        Self {
            string_pool: Pool::new(100, || String::with_capacity(1024)),
            vec_pool: Pool::new(50, || Vec::with_capacity(4096)),
        }
    }
    
    /// Get a reusable string from the pool
    pub fn get_string(&self) -> Reusable<'_, String> {
        let mut string = self.string_pool.try_pull().unwrap_or_else(|| {
            Reusable::new(&self.string_pool, String::with_capacity(1024))
        });
        string.clear();
        string
    }
    
    /// Get a reusable byte vector from the pool
    pub fn get_vec(&self) -> Reusable<'_, Vec<u8>> {
        let mut vec = self.vec_pool.try_pull().unwrap_or_else(|| {
            Reusable::new(&self.vec_pool, Vec::with_capacity(4096))
        });
        vec.clear();
        vec
    }
    
    /// Get memory usage statistics
    pub fn get_stats(&self) -> MemoryStats {
        MemoryStats {
            string_pool_size: self.string_pool.len(),
            vec_pool_size: self.vec_pool.len(),
            estimated_memory_saved_mb: (self.string_pool.len() + self.vec_pool.len()) as f64 * 0.001, // Rough estimate
        }
    }
}

/// Memory usage statistics
pub struct MemoryStats {
    pub string_pool_size: usize,
    pub vec_pool_size: usize,
    pub estimated_memory_saved_mb: f64,
}

/// Global memory manager instance
static MEMORY_MANAGER: OnceLock<Arc<MemoryManager>> = OnceLock::new();

/// Initialize memory manager
pub async fn init_memory_manager(_config: &PlatformConfig) -> Result<()> {
    let manager = Arc::new(MemoryManager::new());
    
    MEMORY_MANAGER.set(manager)
        .map_err(|_| anyhow::anyhow!("Memory manager already initialized"))?;
    
    info!("Memory manager initialized with object pools");
    Ok(())
}

/// Get global memory manager instance
pub fn get_memory_manager() -> Option<Arc<MemoryManager>> {
    MEMORY_MANAGER.get().cloned()
}

/// Shutdown memory manager
pub async fn shutdown_memory_manager() -> Result<()> {
    info!("Memory manager shutdown completed");
    Ok(())
}