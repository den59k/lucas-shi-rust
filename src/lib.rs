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
pub use features::{good_features_to_track, good_features_to_track_grid};
#[allow(deprecated)]
pub use lk::calc_optical_flow;
pub use lk::{
    DEFAULT_FB_THRESHOLD, DEFAULT_MIN_EIGEN_THRESHOLD, TrackResult, TrackStatus, TrackerContext,
    calc_optical_flow_ex, calc_optical_flow_fb,
};
pub use pyramid::{build_pyramid, build_pyramid_into};
