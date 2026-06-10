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
#[allow(deprecated)]
pub use lk::calc_optical_flow;
pub use lk::{DEFAULT_MIN_EIGEN_THRESHOLD, TrackResult, TrackStatus, calc_optical_flow_ex};
pub use pyramid::build_pyramid;
