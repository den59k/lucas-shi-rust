//! Thin `wasm-bindgen` wrapper around the Lucas-Kanade tracker for the browser
//! demo. JS pushes one grayscale frame per tick; the tracker keeps the previous
//! frame and advances every point along the optical flow.

use image::{imageops, GrayImage, ImageBuffer, Luma};
use optical_flow_lk::{
    good_features_to_track, DetectorContext, Grid, TrackOptions, TrackParams, TrackResult,
    TrackStatus, TrackerContext, DEFAULT_FB_THRESHOLD, DEFAULT_MIN_EIGEN_THRESHOLD,
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
    detector: DetectorContext,
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
            detector: DetectorContext::new(),
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
            Some(prev) => self.detector.detect_grid(
                prev,
                Grid {
                    cols,
                    rows,
                    max_per_cell,
                },
                quality,
                min_distance,
                &existing,
            ),
            None => return self.points.len(),
        };

        for corner in feats {
            self.points.push(corner.position());
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

            let params = TrackParams {
                window_size: self.window,
                max_iterations: self.iters,
                min_eigen_threshold: self.min_eigen,
            };
            let results =
                self.ctx
                    .track_fb(&self.points, Some(&predicted), params, self.fb_threshold);

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
            let da = (a.x as f32 - tx).powi(2) + (a.y as f32 - ty).powi(2);
            let db = (b.x as f32 - tx).powi(2) + (b.y as f32 - ty).powi(2);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })?;

        Some((x0 as f32 + best.x as f32, y0 as f32 + best.y as f32))
    }
}

/// Hash noise band-limited by two 3x3 means, as in the crate's
/// `benches/tracking.rs`.
fn bench_texture(w: u32, h: u32, seed: u32) -> Vec<f32> {
    let mut buf: Vec<f32> = (0..w * h)
        .map(|i| {
            let mut s = i
                .wrapping_mul(2654435761)
                .wrapping_add(seed.wrapping_mul(40503))
                ^ 0x9e3779b9;
            s ^= s >> 15;
            s = s.wrapping_mul(0x85ebca6b);
            s ^= s >> 13;
            (s & 0xff) as f32
        })
        .collect();
    for _ in 0..2 {
        let src = buf.clone();
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let (mut sum, mut n) = (0.0, 0.0);
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let (nx, ny) = (x + dx, y + dy);
                        if nx >= 0 && ny >= 0 && nx < w as i32 && ny < h as i32 {
                            sum += src[(ny as u32 * w + nx as u32) as usize];
                            n += 1.0;
                        }
                    }
                }
                buf[(y as u32 * w + x as u32) as usize] = sum / n;
            }
        }
    }
    buf
}

/// `texture` resampled so that content moves by `(sx, sy)`; outside reads 0.
fn bench_frame(texture: &[f32], w: u32, h: u32, sx: f32, sy: f32) -> GrayImage {
    let pixel = |x: i32, y: i32| {
        if x >= 0 && y >= 0 && x < w as i32 && y < h as i32 {
            texture[(y as u32 * w + x as u32) as usize]
        } else {
            0.0
        }
    };
    GrayImage::from_fn(w, h, |x, y| {
        let (fx, fy) = (x as f32 - sx, y as f32 - sy);
        let (x0, y0) = (fx.floor(), fy.floor());
        let (dx, dy) = (fx - x0, fy - y0);
        let (x0, y0) = (x0 as i32, y0 as i32);
        let value = pixel(x0, y0) * (1.0 - dx) * (1.0 - dy)
            + pixel(x0 + 1, y0) * dx * (1.0 - dy)
            + pixel(x0, y0 + 1) * (1.0 - dx) * dy
            + pixel(x0 + 1, y0 + 1) * dx * dy;
        Luma([value as u8])
    })
}

/// Timed per-frame tracking for `index.html?bench=1`: two synthetic textured
/// frames 3 px apart are passed to `TrackerContext::advance` in turn, and
/// every step tracks the strongest corners of the first frame with `track`
/// or `track_fb`.
#[wasm_bindgen]
pub struct Bench {
    frames: [GrayImage; 2],
    corners: Vec<(f32, f32)>,
    points: Vec<(f32, f32)>,
    params: TrackParams,
    levels: usize,
    index: usize,
    ctx: TrackerContext,
}

#[wasm_bindgen]
impl Bench {
    /// Synthesizes the frame pair and detects its corners (quality 0.01, at
    /// least 10 px apart).
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> Bench {
        let texture = bench_texture(width, height, 1);
        let frames = [
            bench_frame(&texture, width, height, 0.0, 0.0),
            bench_frame(&texture, width, height, 3.0, 2.0),
        ];
        let corners = good_features_to_track(&frames[0], 0.01, 10)
            .iter()
            .map(|corner| corner.position())
            .collect();
        Bench {
            frames,
            corners,
            points: Vec::new(),
            params: TrackParams {
                window_size: 21,
                max_iterations: 30,
                min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
            },
            levels: 4,
            index: 0,
            ctx: TrackerContext::new(),
        }
    }

    /// Whether `fb_levels` other than "all" are supported by the linked crate.
    pub fn supports_fb_levels() -> bool {
        true
    }

    /// Selects the case: the `points` strongest corners, the window size,
    /// iteration cap and pyramid levels. Returns the point count used.
    pub fn configure(
        &mut self,
        points: usize,
        window: usize,
        iterations: usize,
        levels: usize,
    ) -> usize {
        self.points = self.corners.iter().take(points).copied().collect();
        self.params.window_size = window;
        self.params.max_iterations = iterations;
        self.levels = levels;
        self.ctx = TrackerContext::new();
        self.index = 0;
        self.ctx.advance(&self.frames[0], levels);
        self.points.len()
    }

    /// One frame step: `advance` with the other frame, then `track`
    /// (`fb_levels < 0`) or `track_fb` over all levels (`fb_levels == 0`) or
    /// the finest `fb_levels`. Returns the number of `Tracked` points.
    pub fn step(&mut self, fb_levels: i32) -> usize {
        self.index ^= 1;
        self.ctx.advance(&self.frames[self.index], self.levels);
        let results = self.track(fb_levels);
        results
            .iter()
            .filter(|r| r.status == TrackStatus::Tracked)
            .count()
    }

    /// FNV-1a hash of the exact result bits of tracking the first frame into
    /// the second, to compare the wasm build with a native run.
    pub fn checksum(&mut self, fb_levels: i32) -> String {
        self.ctx
            .prepare(&self.frames[0], &self.frames[1], self.levels);
        self.index = 1;
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for r in self.track(fb_levels) {
            let status = r.status as u32;
            for word in [
                r.pos.0.to_bits(),
                r.pos.1.to_bits(),
                r.error.to_bits(),
                status,
            ] {
                for byte in word.to_le_bytes() {
                    hash = (hash ^ byte as u64).wrapping_mul(0x0100_0000_01b3);
                }
            }
        }
        format!("{hash:016x}")
    }

    fn track(&mut self, fb_levels: i32) -> &[TrackResult] {
        let options = match fb_levels {
            0 => TrackOptions::new(),
            levels => TrackOptions::new().with_fb_levels(levels.max(1) as usize),
        };
        if fb_levels < 0 {
            self.ctx.track(&self.points, None, self.params)
        } else {
            self.ctx.track_fb_with(
                &self.points,
                None,
                self.params,
                DEFAULT_FB_THRESHOLD,
                options,
            )
        }
    }
}
