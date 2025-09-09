#![allow(
    // Core warnings
    warnings, dead_code, unused_imports, unused_variables, unused_mut,
    deprecated, unreachable_code, ambiguous_glob_reexports, unused_parens,
    non_camel_case_types, unused_assignments,
    // Additional warning types
    unused_macros, unused_allocation, unused_attributes, unused_features,
    unused_labels, unused_lifetimes, unused_qualifications, unused_results,
    unused_unsafe, unused_must_use, unused_comparisons, unused_doc_comments,
    // Type and pattern warnings
    non_snake_case, non_upper_case_globals, improper_ctypes,
    improper_ctypes_definitions, non_shorthand_field_patterns,
    // Lint warnings
    missing_docs, missing_debug_implementations, missing_copy_implementations,
    trivial_casts, trivial_numeric_casts, unstable_features, unused_extern_crates,
    // Clippy - comprehensive suppression
    clippy::all, clippy::pedantic, clippy::nursery, clippy::cargo,
    clippy::restriction, clippy::complexity, clippy::style, clippy::correctness,
    clippy::perf, clippy::suspicious
)]

pub mod reinforcement_learning;
pub mod microstructure;
pub mod dashboard;
pub mod ab_testing;
pub mod portfolio;
pub mod simulation;
pub mod tca;
pub mod factor_processing;
// pub mod factor_batch_processing;  // Temporarily disabled
// pub mod factors;  // Temporarily disabled
pub mod adaptive_execution;
// pub mod execution_algorithms;
pub mod triple_barrier_labeling;
pub mod smart_routing_scorer;
pub mod purged_kfold_validation;
pub mod walk_forward_validation;
pub mod dynamic_factor_selection;
pub mod high_frequency_optimization;
pub mod risk_control;
pub mod regime_detection;
pub mod backtesting_integration;
pub mod realistic_market_simulation;

pub use reinforcement_learning::*;
pub use microstructure::*;
pub use dashboard::*;
pub use ab_testing::*;
pub use factor_processing::*;
// pub use factor_batch_processing::*;  // Temporarily disabled
// pub use factors::*;  // Temporarily disabled
pub use adaptive_execution::*;
// pub use execution_algorithms::*;
pub use triple_barrier_labeling::*;
pub use smart_routing_scorer::*;
pub use purged_kfold_validation::*;
pub use walk_forward_validation::*;
pub use dynamic_factor_selection::*;
pub use high_frequency_optimization::*;
pub use risk_control::*;
pub use regime_detection::*;
pub use backtesting_integration::*;
pub use realistic_market_simulation::*;