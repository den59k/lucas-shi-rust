//! Benchmark wrapper: holds a frame pair and a point set inside wasm memory so
//! timed calls contain only detection or pyramid-build-plus-tracking work.

use image::GrayImage;
use optical_flow_lk::{
    DEFAULT_MIN_EIGEN_THRESHOLD, TrackParams, TrackStatus, TrackerContext, good_features_to_track,
};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Bench {
    prev: GrayImage,
    next: GrayImage,
    points: Vec<(f32, f32)>,
    ctx: TrackerContext,
}

fn status_code(status: TrackStatus) -> f32 {
    match status {
        TrackStatus::Tracked => 0.0,
        TrackStatus::OutOfBounds => 1.0,
        TrackStatus::Diverged => 2.0,
        TrackStatus::LowTexture => 3.0,
        TrackStatus::FbInconsistent => 4.0,
    }
}

#[wasm_bindgen]
impl Bench {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32, prev: &[u8], next: &[u8]) -> Result<Bench, JsError> {
        let prev = GrayImage::from_raw(width, height, prev.to_vec())
            .ok_or_else(|| JsError::new("prev frame size mismatch"))?;
        let next = GrayImage::from_raw(width, height, next.to_vec())
            .ok_or_else(|| JsError::new("next frame size mismatch"))?;
        Ok(Bench { prev, next, points: Vec::new(), ctx: TrackerContext::new() })
    }

    /// True when this build selected the simd128 kernels.
    pub fn simd128(&self) -> bool {
        cfg!(target_feature = "simd128")
    }

    /// Flat `[x0, y0, x1, y1, ...]` points to track.
    pub fn set_points(&mut self, xy: &[f32]) {
        self.points = xy.chunks_exact(2).map(|c| (c[0], c[1])).collect();
    }

    /// Detection only; returns the corner count so no result copy is timed.
    pub fn detect(&self, quality: f32, min_distance: u32) -> usize {
        good_features_to_track(&self.prev, quality, min_distance).len()
    }

    /// Flat `[x, y, score, ...]` corners for parity checks.
    pub fn detect_points(&self, quality: f32, min_distance: u32) -> Vec<f32> {
        good_features_to_track(&self.prev, quality, min_distance)
            .into_iter()
            .flat_map(|c| [c.x as f32, c.y as f32, c.score])
            .collect()
    }

    /// Builds both pyramids and tracks the stored points; returns the number
    /// of points with `Tracked` status.
    pub fn track(&mut self, levels: usize, window: usize, iterations: usize) -> usize {
        self.ctx.prepare(&self.prev, &self.next, levels);
        let params = TrackParams {
            window_size: window,
            max_iterations: iterations,
            min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
        };
        self.ctx
            .track(&self.points, None, params)
            .iter()
            .filter(|r| r.status == TrackStatus::Tracked)
            .count()
    }

    /// Flat `[x, y, status, ...]` results for parity checks.
    pub fn track_results(&mut self, levels: usize, window: usize, iterations: usize) -> Vec<f32> {
        self.ctx.prepare(&self.prev, &self.next, levels);
        let params = TrackParams {
            window_size: window,
            max_iterations: iterations,
            min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
        };
        self.ctx
            .track(&self.points, None, params)
            .iter()
            .flat_map(|r| [r.pos.0, r.pos.1, status_code(r.status)])
            .collect()
    }
}
