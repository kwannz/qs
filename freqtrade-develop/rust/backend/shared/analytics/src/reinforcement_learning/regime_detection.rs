use anyhow::{Result, Context};
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug};

/// Market regime states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    Bull,      // 上涨趋势
    Bear,      // 下跌趋势
    Sideways,  // 震荡盘整
    Volatile,  // 高波动
}

impl MarketRegime {
    pub fn to_index(&self) -> usize {
        match self {
            MarketRegime::Bull => 0,
            MarketRegime::Bear => 1,
            MarketRegime::Sideways => 2,
            MarketRegime::Volatile => 3,
        }
    }

    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(MarketRegime::Bull),
            1 => Some(MarketRegime::Bear),
            2 => Some(MarketRegime::Sideways),
            3 => Some(MarketRegime::Volatile),
            _ => None,
        }
    }
}

/// HMM configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HMMConfig {
    pub num_states: usize,
    pub num_observations: usize,
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub regularization: f64,
}

impl Default for HMMConfig {
    fn default() -> Self {
        Self {
            num_states: 4,           // Bull, Bear, Sideways, Volatile
            num_observations: 3,     // Return, Volatility, Volume
            max_iterations: 100,
            convergence_threshold: 1e-6,
            regularization: 1e-8,
        }
    }
}

/// Market observation data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketObservation {
    pub timestamp: i64,
    pub returns: f64,        // 收益率
    pub volatility: f64,     // 波动率
    pub volume: f64,         // 成交量标准化
    pub spread: Option<f64>, // 买卖价差
}

impl MarketObservation {
    pub fn to_vector(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.returns, self.volatility, self.volume])
    }

    pub fn discretize(&self) -> usize {
        // Simple discretization for HMM observations
        let vol_threshold = 0.02;
        let return_threshold = 0.01;
        
        match (self.returns.abs() > return_threshold, self.volatility > vol_threshold) {
            (true, true) => 0,   // High return + High vol
            (true, false) => 1,  // High return + Low vol
            (false, true) => 2,  // Low return + High vol
            (false, false) => 3, // Low return + Low vol
        }
    }
}

/// Hidden Markov Model for regime detection
#[derive(Debug, Clone)]
pub struct HiddenMarkovModel {
    config: HMMConfig,
    // State transition matrix (N x N)
    transition_matrix: DMatrix<f64>,
    // Emission probabilities (N x M)
    emission_matrix: DMatrix<f64>,
    // Initial state probabilities
    initial_probs: DVector<f64>,
    // Current state probabilities
    current_state_probs: DVector<f64>,
    // Observation sequence
    observations: Vec<usize>,
}

impl HiddenMarkovModel {
    pub fn new(config: HMMConfig) -> Self {
        let num_states = config.num_states;
        let num_observations = config.num_observations;
        
        // Initialize with uniform distributions
        let transition_matrix = DMatrix::from_fn(num_states, num_states, |_, _| 1.0 / num_states as f64);
        let emission_matrix = DMatrix::from_fn(num_states, num_observations, |_, _| 1.0 / num_observations as f64);
        let initial_probs = DVector::from_fn(num_states, |_, _| 1.0 / num_states as f64);
        let current_state_probs = initial_probs.clone();
        
        Self {
            config,
            transition_matrix,
            emission_matrix,
            initial_probs,
            current_state_probs,
            observations: Vec::new(),
        }
    }

    /// Train HMM using Baum-Welch algorithm
    pub fn train(&mut self, observations: &[usize]) -> Result<f64> {
        let n_states = self.config.num_states;
        let n_obs = observations.len();
        
        if n_obs == 0 {
            return Err(anyhow::anyhow!("Empty observation sequence"));
        }

        let mut log_likelihood = f64::NEG_INFINITY;
        
        for iteration in 0..self.config.max_iterations {
            let (forward, backward, gamma, xi) = self.forward_backward(observations)?;
            
            // Calculate current log likelihood
            let current_ll: f64 = (0..n_states)
                .map(|j| forward[(n_obs - 1, j)].ln())
                .sum();
            
            debug!("Iteration {}: Log likelihood = {}", iteration, current_ll);
            
            // Check convergence
            if (current_ll - log_likelihood).abs() < self.config.convergence_threshold {
                info!("HMM converged after {} iterations", iteration);
                break;
            }
            log_likelihood = current_ll;
            
            // M-step: Update parameters
            self.update_parameters(&gamma, &xi, observations)?;
        }
        
        self.observations = observations.to_vec();
        Ok(log_likelihood)
    }

    /// Forward-Backward algorithm
    fn forward_backward(&self, observations: &[usize]) -> Result<(DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, Vec<DMatrix<f64>>)> {
        let n_states = self.config.num_states;
        let n_obs = observations.len();
        
        // Forward algorithm
        let mut forward = DMatrix::zeros(n_obs, n_states);
        
        // Initialize
        for i in 0..n_states {
            forward[(0, i)] = self.initial_probs[i] * self.emission_matrix[(i, observations[0])];
        }
        
        // Forward pass
        for t in 1..n_obs {
            for j in 0..n_states {
                let mut sum = 0.0;
                for i in 0..n_states {
                    sum += forward[(t-1, i)] * self.transition_matrix[(i, j)];
                }
                forward[(t, j)] = sum * self.emission_matrix[(j, observations[t])];
            }
        }
        
        // Backward algorithm
        let mut backward = DMatrix::zeros(n_obs, n_states);
        
        // Initialize
        for i in 0..n_states {
            backward[(n_obs-1, i)] = 1.0;
        }
        
        // Backward pass
        for t in (0..n_obs-1).rev() {
            for i in 0..n_states {
                let mut sum = 0.0;
                for j in 0..n_states {
                    sum += self.transition_matrix[(i, j)] * 
                           self.emission_matrix[(j, observations[t+1])] * 
                           backward[(t+1, j)];
                }
                backward[(t, i)] = sum;
            }
        }
        
        // Calculate gamma (state probabilities)
        let mut gamma = DMatrix::zeros(n_obs, n_states);
        for t in 0..n_obs {
            let norm: f64 = (0..n_states).map(|i| forward[(t, i)] * backward[(t, i)]).sum();
            for i in 0..n_states {
                gamma[(t, i)] = forward[(t, i)] * backward[(t, i)] / norm;
            }
        }
        
        // Calculate xi (transition probabilities)
        let mut xi = Vec::with_capacity(n_obs - 1);
        for t in 0..n_obs-1 {
            let mut xi_t = DMatrix::zeros(n_states, n_states);
            let norm: f64 = (0..n_states).map(|i| forward[(n_obs-1, i)]).sum();
            
            for i in 0..n_states {
                for j in 0..n_states {
                    xi_t[(i, j)] = forward[(t, i)] * 
                                  self.transition_matrix[(i, j)] * 
                                  self.emission_matrix[(j, observations[t+1])] * 
                                  backward[(t+1, j)] / norm;
                }
            }
            xi.push(xi_t);
        }
        
        Ok((forward, backward, gamma, xi))
    }
    
    /// Update HMM parameters (M-step)
    fn update_parameters(&mut self, gamma: &DMatrix<f64>, xi: &[DMatrix<f64>], observations: &[usize]) -> Result<()> {
        let n_states = self.config.num_states;
        let n_obs = observations.len();
        let regularization = self.config.regularization;
        
        // Update initial state probabilities
        for i in 0..n_states {
            self.initial_probs[i] = gamma[(0, i)];
        }
        
        // Update transition matrix
        for i in 0..n_states {
            let gamma_sum: f64 = (0..n_obs-1).map(|t| gamma[(t, i)]).sum();
            for j in 0..n_states {
                let xi_sum: f64 = xi.iter().map(|xi_t| xi_t[(i, j)]).sum();
                self.transition_matrix[(i, j)] = (xi_sum + regularization) / (gamma_sum + regularization * n_states as f64);
            }
        }
        
        // Update emission matrix
        for i in 0..n_states {
            let gamma_sum: f64 = (0..n_obs).map(|t| gamma[(t, i)]).sum();
            for k in 0..self.config.num_observations {
                let mut emission_sum = 0.0;
                for t in 0..n_obs {
                    if observations[t] == k {
                        emission_sum += gamma[(t, i)];
                    }
                }
                self.emission_matrix[(i, k)] = (emission_sum + regularization) / (gamma_sum + regularization * self.config.num_observations as f64);
            }
        }
        
        Ok(())
    }
    
    /// Predict most likely state sequence using Viterbi algorithm
    pub fn viterbi(&self, observations: &[usize]) -> Result<Vec<usize>> {
        let n_states = self.config.num_states;
        let n_obs = observations.len();
        
        if n_obs == 0 {
            return Ok(Vec::new());
        }
        
        let mut delta = DMatrix::zeros(n_obs, n_states);
        let mut psi = DMatrix::zeros(n_obs, n_states);
        
        // Initialize
        for i in 0..n_states {
            delta[(0, i)] = self.initial_probs[i] * self.emission_matrix[(i, observations[0])];
        }
        
        // Forward pass
        for t in 1..n_obs {
            for j in 0..n_states {
                let mut max_val = f64::NEG_INFINITY;
                let mut max_idx = 0;
                
                for i in 0..n_states {
                    let val = delta[(t-1, i)] * self.transition_matrix[(i, j)];
                    if val > max_val {
                        max_val = val;
                        max_idx = i;
                    }
                }
                
                delta[(t, j)] = max_val * self.emission_matrix[(j, observations[t])];
                psi[(t, j)] = max_idx as f64;
            }
        }
        
        // Backward pass
        let mut path = vec![0; n_obs];
        
        // Find best final state
        let mut max_val = f64::NEG_INFINITY;
        for i in 0..n_states {
            if delta[(n_obs-1, i)] > max_val {
                max_val = delta[(n_obs-1, i)];
                path[n_obs-1] = i;
            }
        }
        
        // Backtrack
        for t in (0..n_obs-1).rev() {
            path[t] = psi[(t+1, path[t+1])] as usize;
        }
        
        Ok(path)
    }
    
    /// Update current state probabilities with new observation
    pub fn update_state_probabilities(&mut self, observation: usize) -> Result<&DVector<f64>> {
        let n_states = self.config.num_states;
        let mut new_probs = DVector::zeros(n_states);
        
        // Forward step
        for j in 0..n_states {
            let mut sum = 0.0;
            for i in 0..n_states {
                sum += self.current_state_probs[i] * self.transition_matrix[(i, j)];
            }
            new_probs[j] = sum * self.emission_matrix[(j, observation)];
        }
        
        // Normalize
        let norm: f64 = new_probs.iter().sum();
        if norm > 0.0 {
            new_probs /= norm;
        }
        
        self.current_state_probs = new_probs;
        Ok(&self.current_state_probs)
    }
    
    /// Get most likely current regime
    pub fn current_regime(&self) -> MarketRegime {
        let max_idx = self.current_state_probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
            
        MarketRegime::from_index(max_idx).unwrap_or(MarketRegime::Sideways)
    }
}

/// K-means clustering for regime detection
#[derive(Debug, Clone)]
pub struct KMeansRegimeDetector {
    pub k: usize,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub centroids: Vec<DVector<f64>>,
    pub labels: Vec<usize>,
}

impl KMeansRegimeDetector {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iterations: 100,
            tolerance: 1e-6,
            centroids: Vec::new(),
            labels: Vec::new(),
        }
    }
    
    /// Train K-means clustering
    pub fn fit(&mut self, observations: &[MarketObservation]) -> Result<()> {
        if observations.is_empty() {
            return Err(anyhow::anyhow!("Empty observation set"));
        }
        
        let n = observations.len();
        let dim = observations[0].to_vector().len();
        
        // Initialize centroids randomly
        self.centroids.clear();
        for _ in 0..self.k {
            let mut centroid = DVector::zeros(dim);
            for j in 0..dim {
                centroid[j] = observations[fastrand::usize(..n)].to_vector()[j];
            }
            self.centroids.push(centroid);
        }
        
        self.labels = vec![0; n];
        
        for iteration in 0..self.max_iterations {
            let mut changed = false;
            
            // Assignment step
            for (i, obs) in observations.iter().enumerate() {
                let obs_vec = obs.to_vector();
                let mut min_dist = f64::INFINITY;
                let mut closest_cluster = 0;
                
                for (j, centroid) in self.centroids.iter().enumerate() {
                    let dist = (&obs_vec - centroid).norm_squared();
                    if dist < min_dist {
                        min_dist = dist;
                        closest_cluster = j;
                    }
                }
                
                if self.labels[i] != closest_cluster {
                    self.labels[i] = closest_cluster;
                    changed = true;
                }
            }
            
            // Update step
            let mut new_centroids = vec![DVector::zeros(dim); self.k];
            let mut counts = vec![0; self.k];
            
            for (i, obs) in observations.iter().enumerate() {
                let cluster = self.labels[i];
                new_centroids[cluster] += obs.to_vector();
                counts[cluster] += 1;
            }
            
            for (j, centroid) in new_centroids.iter_mut().enumerate() {
                if counts[j] > 0 {
                    *centroid /= counts[j] as f64;
                }
            }
            
            // Check convergence
            let mut max_change = 0.0_f64;
            for (old, new) in self.centroids.iter().zip(new_centroids.iter()) {
                let change = (old - new).norm();
                max_change = max_change.max(change);
            }
            
            self.centroids = new_centroids;
            
            if max_change < self.tolerance {
                info!("K-means converged after {} iterations", iteration);
                break;
            }
            
            if !changed {
                break;
            }
        }
        
        Ok(())
    }
    
    /// Predict cluster for new observation
    pub fn predict(&self, observation: &MarketObservation) -> Result<usize> {
        if self.centroids.is_empty() {
            return Err(anyhow::anyhow!("Model not trained"));
        }
        
        let obs_vec = observation.to_vector();
        let mut min_dist = f64::INFINITY;
        let mut closest_cluster = 0;
        
        for (j, centroid) in self.centroids.iter().enumerate() {
            let dist = (&obs_vec - centroid).norm_squared();
            if dist < min_dist {
                min_dist = dist;
                closest_cluster = j;
            }
        }
        
        Ok(closest_cluster)
    }
    
    /// Map cluster to market regime
    pub fn cluster_to_regime(&self, cluster: usize) -> Result<MarketRegime> {
        if cluster >= self.centroids.len() {
            return Err(anyhow::anyhow!("Invalid cluster index"));
        }
        
        let centroid = &self.centroids[cluster];
        let returns = centroid[0];
        let volatility = centroid[1];
        
        // Simple heuristic mapping
        match (returns > 0.0, volatility > 0.02) {
            (true, false) => Ok(MarketRegime::Bull),
            (false, false) => Ok(MarketRegime::Bear), 
            (_, true) => Ok(MarketRegime::Volatile),
            _ => Ok(MarketRegime::Sideways),
        }
    }
}

/// Regime detection service combining HMM and K-means
#[derive(Debug)]
pub struct RegimeDetectionService {
    hmm: Arc<RwLock<HiddenMarkovModel>>,
    kmeans: Arc<RwLock<KMeansRegimeDetector>>,
    config: HMMConfig,
    historical_observations: Arc<RwLock<Vec<MarketObservation>>>,
}

impl RegimeDetectionService {
    pub fn new(config: HMMConfig) -> Self {
        let hmm = Arc::new(RwLock::new(HiddenMarkovModel::new(config.clone())));
        let kmeans = Arc::new(RwLock::new(KMeansRegimeDetector::new(config.num_states)));
        
        Self {
            hmm,
            kmeans,
            config,
            historical_observations: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Train both HMM and K-means models
    pub async fn train(&self, observations: &[MarketObservation]) -> Result<()> {
        info!("Training regime detection models with {} observations", observations.len());
        
        // Store observations
        {
            let mut hist_obs = self.historical_observations.write().await;
            hist_obs.clear();
            hist_obs.extend_from_slice(observations);
        }
        
        // Prepare discretized observations for HMM
        let discrete_obs: Vec<usize> = observations.iter().map(|obs| obs.discretize()).collect();
        
        // Train HMM
        {
            let mut hmm = self.hmm.write().await;
            let log_likelihood = hmm.train(&discrete_obs)
                .context("Failed to train HMM")?;
            info!("HMM training completed with log likelihood: {}", log_likelihood);
        }
        
        // Train K-means
        {
            let mut kmeans = self.kmeans.write().await;
            kmeans.fit(observations)
                .context("Failed to train K-means")?;
            info!("K-means training completed");
        }
        
        Ok(())
    }
    
    /// Detect current market regime
    pub async fn detect_regime(&self, observation: &MarketObservation) -> Result<(MarketRegime, f64)> {
        // HMM prediction
        let hmm_regime = {
            let mut hmm = self.hmm.write().await;
            let discrete_obs = observation.discretize();
            hmm.update_state_probabilities(discrete_obs)?;
            hmm.current_regime()
        };
        
        // K-means prediction
        let kmeans_regime = {
            let kmeans = self.kmeans.read().await;
            let cluster = kmeans.predict(observation)?;
            kmeans.cluster_to_regime(cluster)?
        };
        
        // Combine predictions (simple voting, could be made more sophisticated)
        let final_regime = if hmm_regime == kmeans_regime {
            hmm_regime
        } else {
            // Default to HMM in case of disagreement
            hmm_regime
        };
        
        // Calculate confidence (simplified)
        let confidence = if hmm_regime == kmeans_regime { 0.8 } else { 0.6 };
        
        debug!("Regime detection: HMM={:?}, K-means={:?}, Final={:?}, Confidence={}", 
               hmm_regime, kmeans_regime, final_regime, confidence);
        
        Ok((final_regime, confidence))
    }
    
    /// Get regime transition probabilities
    pub async fn get_transition_probabilities(&self) -> Result<HashMap<MarketRegime, HashMap<MarketRegime, f64>>> {
        let hmm = self.hmm.read().await;
        let mut result = HashMap::new();
        
        for i in 0..self.config.num_states {
            if let Some(from_regime) = MarketRegime::from_index(i) {
                let mut transitions = HashMap::new();
                for j in 0..self.config.num_states {
                    if let Some(to_regime) = MarketRegime::from_index(j) {
                        transitions.insert(to_regime, hmm.transition_matrix[(i, j)]);
                    }
                }
                result.insert(from_regime, transitions);
            }
        }
        
        Ok(result)
    }
    
    /// Add new observation for continuous learning
    pub async fn add_observation(&self, observation: MarketObservation) -> Result<()> {
        let mut hist_obs = self.historical_observations.write().await;
        hist_obs.push(observation);
        
        // Retrain periodically (every 100 observations)
        if hist_obs.len() % 100 == 0 {
            info!("Retraining regime detection models");
            let obs_clone = hist_obs.clone();
            drop(hist_obs);
            self.train(&obs_clone).await?;
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_market_observation_discretization() {
        let obs1 = MarketObservation {
            timestamp: 1234567890,
            returns: 0.05,      // High return
            volatility: 0.03,   // High vol
            volume: 1.2,
            spread: None,
        };
        assert_eq!(obs1.discretize(), 0);

        let obs2 = MarketObservation {
            timestamp: 1234567891,
            returns: 0.005,     // Low return
            volatility: 0.01,   // Low vol
            volume: 0.8,
            spread: None,
        };
        assert_eq!(obs2.discretize(), 3);
    }

    #[test]
    fn test_regime_mapping() {
        assert_eq!(MarketRegime::Bull.to_index(), 0);
        assert_eq!(MarketRegime::from_index(0), Some(MarketRegime::Bull));
        assert_eq!(MarketRegime::from_index(99), None);
    }

    #[tokio::test]
    async fn test_kmeans_basic() {
        let observations = vec![
            MarketObservation {
                timestamp: 1,
                returns: 0.05,
                volatility: 0.02,
                volume: 1.0,
                spread: None,
            },
            MarketObservation {
                timestamp: 2,
                returns: -0.03,
                volatility: 0.01,
                volume: 0.8,
                spread: None,
            },
        ];

        let mut kmeans = KMeansRegimeDetector::new(2);
        let result = kmeans.fit(&observations);
        assert!(result.is_ok());
    }
}