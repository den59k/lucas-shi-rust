//! Thin `wasm-bindgen` wrapper around the Lucas-Kanade tracker for the browser
//! demo. JS pushes one grayscale frame per tick; the tracker keeps the previous
//! frame and advances every point along the optical flow.

use image::{imageops, GrayImage, ImageBuffer};
use optical_flow_lk::{
    good_features_to_track, good_features_to_track_grid, TrackStatus, TrackerContext,
    DEFAULT_FB_THRESHOLD, DEFAULT_MIN_EIGEN_THRESHOLD,
};
use wasm_bindgen::prelude::*;

/// Status code mirrored into the flattened JS output (`[x, y, status, ...]`).
const STATUS_TRACKED: f32 = 0.0;

#[wasm_bindgen]
pub struct Tracker {
    width: u32,
    height: u32,
    levels: usize,
    window: usize,
    iters: usize,
    min_eigen: f32,
    fb_threshold: f32,
    /// Drop a tracked point once its mean photometric residual (0..255) exceeds
    /// this, to cull drifters and partially-occluded points.
    max_error: f32,
    ctx: TrackerContext,
    prev: Option<GrayImage>,
    points: Vec<(f32, f32)>,
    /// Per-point velocity (level-0 px/frame), kept in lockstep with `points`.
    /// Used to seed the next frame's search with a constant-velocity prediction.
    vel: Vec<(f32, f32)>,
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
            levels: 4,
            window: 21,
            iters: 25,
            min_eigen: DEFAULT_MIN_EIGEN_THRESHOLD,
            fb_threshold: DEFAULT_FB_THRESHOLD,
            max_error: 30.0,
            ctx: TrackerContext::new(),
            prev: None,
            points: Vec::new(),
            vel: Vec::new(),
        }
    }

    /// Adds a point to track, in frame pixel coordinates.
    pub fn add_point(&mut self, x: f32, y: f32) {
        self.points.push((x, y));
        self.vel.push((0.0, 0.0));
    }

    /// Adds a point, but first snaps it to the strongest Shi-Tomasi corner near
    /// the tap so it latches onto a trackable feature instead of a flat region.
    /// Falls back to the raw position if no corner is found nearby (or before the
    /// first frame).
    pub fn add_point_snapped(&mut self, x: f32, y: f32) {
        let (sx, sy) = self.snap_to_corner(x, y).unwrap_or((x, y));
        self.add_point(sx, sy);
    }

    /// Removes all tracked points.
    pub fn clear(&mut self) {
        self.points.clear();
        self.vel.clear();
    }

    /// Detects strong Shi-Tomasi corners on the most recent frame and adds them
    /// to the tracked set, spread uniformly over a grid and kept clear of points
    /// already being tracked. Call it repeatedly to top up after points are
    /// lost. Returns the new total point count. No-op until a frame has arrived.
    pub fn auto_detect(&mut self) -> usize {
        // Heuristics scaled to the processing resolution: ~64px grid cells, a
        // couple of corners per cell, spacing of roughly width/24 pixels.
        let cols = (self.width / 64).max(1);
        let rows = (self.height / 64).max(1);
        let min_distance = (self.width.min(self.height) / 24).max(6);
        let quality = 0.05;
        let max_per_cell = 2;

        let existing = self.points.clone();
        let feats = match self.prev.as_ref() {
            Some(prev) => good_features_to_track_grid(
                prev,
                cols,
                rows,
                max_per_cell,
                quality,
                min_distance,
                &existing,
            ),
            None => return self.points.len(),
        };

        for (x, y, _q) in feats {
            self.points.push((x as f32, y as f32));
            self.vel.push((0.0, 0.0));
        }
        self.points.len()
    }

    /// Number of points currently being tracked.
    pub fn count(&self) -> usize {
        self.points.len()
    }

    /// Pushes a new grayscale frame (`width * height` bytes) and advances every
    /// point along the optical flow from the previous frame. Points are kept only
    /// if they pass the forward-backward round-trip check and stay below the
    /// photometric-error threshold; the rest are dropped. The search is seeded
    /// with a constant-velocity prediction per point so fast motion is followed
    /// more reliably. Returns the survivors as a flat `[x, y, status, ...]` array.
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

            // Constant-velocity seed: where we expect each point to land.
            let predicted: Vec<(f32, f32)> = self
                .points
                .iter()
                .zip(&self.vel)
                .map(|(&(px, py), &(vx, vy))| (px + vx, py + vy))
                .collect();

            let results = self.ctx.track_fb(
                &self.points,
                Some(&predicted),
                self.window,
                self.iters,
                self.min_eigen,
                self.fb_threshold,
            );

            let mut kept = Vec::with_capacity(results.len());
            let mut kept_vel = Vec::with_capacity(results.len());
            let mut out = Vec::with_capacity(results.len() * 3);
            for (i, r) in results.iter().enumerate() {
                if r.status == TrackStatus::Tracked && r.error <= self.max_error {
                    let (ox, oy) = self.points[i];
                    kept.push(r.pos);
                    kept_vel.push((r.pos.0 - ox, r.pos.1 - oy));
                    out.push(r.pos.0);
                    out.push(r.pos.1);
                    out.push(STATUS_TRACKED);
                }
            }
            self.points = kept;
            self.vel = kept_vel;
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

    /// Snaps `(x, y)` to the nearest strong Shi-Tomasi corner within a small
    /// window on the most recent frame. Returns `None` if no frame has arrived
    /// yet, the window is too small, or the region is too flat to yield a corner.
    fn snap_to_corner(&self, x: f32, y: f32) -> Option<(f32, f32)> {
        let prev = self.prev.as_ref()?;

        let radius: i32 = 24;
        let (w, h) = (self.width as i32, self.height as i32);
        let cx = x.round() as i32;
        let cy = y.round() as i32;
        let x0 = (cx - radius).clamp(0, w - 1);
        let y0 = (cy - radius).clamp(0, h - 1);
        let x1 = (cx + radius).clamp(0, w - 1);
        let y1 = (cy + radius).clamp(0, h - 1);
        let cw = (x1 - x0 + 1) as u32;
        let ch = (y1 - y0 + 1) as u32;
        // Need room for the 3x3 gradient + non-max-suppression border.
        if cw < 7 || ch < 7 {
            return None;
        }

        let crop = imageops::crop_imm(prev, x0 as u32, y0 as u32, cw, ch).to_image();
        let feats = good_features_to_track(&crop, 0.1, 3);

        // Pick the corner closest to the tap (coords are relative to the crop).
        let (tx, ty) = ((cx - x0) as f32, (cy - y0) as f32);
        let best = feats.iter().min_by(|a, b| {
            let da = (a.0 as f32 - tx).powi(2) + (a.1 as f32 - ty).powi(2);
            let db = (b.0 as f32 - tx).powi(2) + (b.1 as f32 - ty).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })?;

        Some((x0 as f32 + best.0 as f32, y0 as f32 + best.1 as f32))
    }
}
