//! High-performance computer vision algorithms for real-time applications
//!
//! Provides implementations of:
//! - Lucas-Kanade optical flow
//! - Shi-Tomasi feature detection
//! - Optimized image processing pipelines
//!
//! Designed to be compatible with WebAssembly (Wasm).

mod features;
mod float_features;
mod float_image;
mod lk;
mod pyramid;
mod utils;

// Re-export main functionality
pub use features::{
    Corner, DetectorCapacity, DetectorContext, Grid, good_features_to_track,
    good_features_to_track_grid,
};
pub use float_features::{FloatDetectorContext, ScoreThreshold};
pub use float_image::{FloatImageError, FloatImageView};
pub use lk::{
    CONVERGENCE_EPSILON, DEFAULT_FB_THRESHOLD, DEFAULT_MIN_EIGEN_THRESHOLD, FloatPyramidLevel,
    FloatTrackerContext, TrackOptions, TrackParams, TrackResult, TrackStatus, TrackerCapacity,
    TrackerContext, calc_optical_flow_ex, calc_optical_flow_fb,
};
pub use pyramid::{build_pyramid, build_pyramid_into};
