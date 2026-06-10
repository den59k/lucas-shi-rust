//! Thin `wasm-bindgen` wrapper around the Lucas-Kanade tracker for the browser
//! demo. JS pushes one grayscale frame per tick; the tracker keeps the previous
//! frame and advances every point along the optical flow.

use image::{GrayImage, ImageBuffer};
use optical_flow_lk::{TrackStatus, TrackerContext, DEFAULT_MIN_EIGEN_THRESHOLD};
use wasm_bindgen::prelude::*;

/// Status codes mirrored into the flattened JS output (`[x, y, status, ...]`).
const STATUS_TRACKED: f32 = 0.0;

#[wasm_bindgen]
pub struct Tracker {
    width: u32,
    height: u32,
    levels: usize,
    window: usize,
    iters: usize,
    min_eigen: f32,
    ctx: TrackerContext,
    prev: Option<GrayImage>,
    points: Vec<(f32, f32)>,
}

#[wasm_bindgen]
impl Tracker {
    /// Creates a tracker for frames of the given size. `window`/`levels`/`iters`
    /// are tuned for real-time use on a phone; tweak from JS if desired.
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> Tracker {
        Tracker {
            width,
            height,
            levels: 3,
            window: 21,
            iters: 15,
            min_eigen: DEFAULT_MIN_EIGEN_THRESHOLD,
            ctx: TrackerContext::new(),
            prev: None,
            points: Vec::new(),
        }
    }

    /// Adds a point to track, in frame pixel coordinates.
    pub fn add_point(&mut self, x: f32, y: f32) {
        self.points.push((x, y));
    }

    /// Removes all tracked points.
    pub fn clear(&mut self) {
        self.points.clear();
    }

    /// Number of points currently being tracked.
    pub fn count(&self) -> usize {
        self.points.len()
    }

    /// Pushes a new grayscale frame (`width * height` bytes), advances every
    /// point along the flow from the previous frame, drops points that leave the
    /// frame or fail to track, and returns the survivors as a flat
    /// `[x, y, status, ...]` array.
    pub fn push_frame(&mut self, gray: &[u8]) -> Vec<f32> {
        let next: GrayImage = ImageBuffer::from_raw(self.width, self.height, gray.to_vec())
            .expect("grayscale buffer length must equal width * height");

        if !self.points.is_empty() && self.prev.is_some() {
            // Build both pyramids; this copies the frames internally so the
            // borrow of `prev` ends before we touch other fields.
            {
                let prev = self.prev.as_ref().unwrap();
                self.ctx.prepare(prev, &next, self.levels);
            }

            let results =
                self.ctx
                    .track(&self.points, None, self.window, self.iters, self.min_eigen);

            let mut kept = Vec::with_capacity(results.len());
            let mut out = Vec::with_capacity(results.len() * 3);
            for r in results {
                if r.status == TrackStatus::Tracked {
                    kept.push(r.pos);
                    out.push(r.pos.0);
                    out.push(r.pos.1);
                    out.push(STATUS_TRACKED);
                }
            }
            self.points = kept;
            self.prev = Some(next);
            return out;
        }

        // First frame, or nothing to track yet: just store the frame and echo
        // any points the user has already placed.
        self.prev = Some(next);
        let mut out = Vec::with_capacity(self.points.len() * 3);
        for &(x, y) in &self.points {
            out.push(x);
            out.push(y);
            out.push(STATUS_TRACKED);
        }
        out
    }
}
