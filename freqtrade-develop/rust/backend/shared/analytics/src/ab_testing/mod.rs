pub mod experiment_manager;
pub mod traffic_splitter;
pub mod statistical_engine;
pub mod gradual_rollout;

pub use experiment_manager::{
    ExperimentManager, 
    ExperimentConfig, 
    ExperimentStatus, 
    AllocationStrategy,
    StatisticalAnalysis,
    TrafficSplitter as TrafficSplitterTrait,
    StatisticalEngine as StatisticalEngineTrait
};
pub use traffic_splitter::{DefaultTrafficSplitter, UserAssignment};
pub use statistical_engine::{StatisticalEngineFactory, HypothesisTestResult};
pub use gradual_rollout::{GradualRolloutManager, GradualRolloutConfig, RolloutPhase};