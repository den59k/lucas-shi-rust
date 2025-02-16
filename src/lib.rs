//! High-performance computer vision algorithms for real-time applications
//!
//! Provides implementations of:
//! - Lucas-Kanade optical flow
//! - Shi-Tomasi feature detection
//! - Optimized image processing pipelines
//!
//! Designed to be compatible with WebAssembly (Wasm).

mod features;
mod lk;
mod pyramid;
mod utils;

// Re-export main functionality
pub use features::good_features_to_track;
pub use lk::calc_optical_flow;
pub use pyramid::build_pyramid;
