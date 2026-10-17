use image::GrayImage;
use std::cmp::Ordering;

use crate::utils::{box_filter_3x3::box_filter_3x3_raw, fast_gradients::compute_gradients_into};

/// Finds good features points using the Shi-Tomasi algorithm
///
/// This is a convenience wrapper that allocates a fresh [`DetectorContext`];
/// per-frame callers should keep one context and call
/// [`DetectorContext::detect`] instead.
///
/// # Arguments
/// * `image` - Target image (grayscale)
/// * `quality_level` - Minimum fraction of the strongest local corner response
/// * `min_distance` - Minimum Euclidean spacing in pixels; zero disables spacing
///
///
/// # Returns
/// `(x, y, min_eigenvalue)` tuples sorted by descending quality; equal scores
/// retain row-major order. Only finite, positive scores are returned. Flat
/// images and images narrower or shorter than three pixels yield no features.
/// Scores use Scharr gradients divided by 32 with integer truncation, followed
/// by a separable 3x3 average with integer truncation after each pass.
pub fn good_features_to_track(
    image: &GrayImage,
    quality_level: f32,
    min_distance: u32,
) -> Vec<(u32, u32, f32)> {
    let mut ctx = DetectorContext::new();
    ctx.detect(image, quality_level, min_distance);
    std::mem::take(&mut ctx.results)
}

/// Finds good feature points with uniform frame coverage by detecting per grid
/// cell, while respecting features that are already being tracked.
///
/// The image is split into a `grid_cols` x `grid_rows` grid. For every cell up
/// to `max_per_cell` best Shi-Tomasi corners are returned, where `min_distance`
/// is respected both among the freshly detected corners and against
/// `existing_points`. Cells whose budget is already met by `existing_points` are
/// skipped, so this can be called every few frames to top up lost tracks
/// without clustering new detections on top of surviving ones.
///
/// This is a convenience wrapper that allocates a fresh [`DetectorContext`];
/// per-frame callers should keep one context and call
/// [`DetectorContext::detect_grid`] instead.
///
/// # Arguments
/// * `image` - Target image (grayscale)
/// * `grid_cols` / `grid_rows` - grid dimensions (both must be non-zero)
/// * `max_per_cell` - feature budget per cell, counting `existing_points`
/// * `quality_level` - Shi-Tomasi quality level, see [`good_features_to_track`]
/// * `min_distance` - minimum spacing in pixels between any two kept points
///   (new or existing)
/// * `existing_points` - already-tracked points (level-0 pixel coordinates) that
///   occupy budget and must be kept clear by `min_distance`
///
/// # Returns
/// Newly detected corners as `(x, y, min_eigenvalue)`, sorted by descending
/// quality. `existing_points` are never included in the output.
#[allow(clippy::too_many_arguments)]
pub fn good_features_to_track_grid(
    image: &GrayImage,
    grid_cols: u32,
    grid_rows: u32,
    max_per_cell: u32,
    quality_level: f32,
    min_distance: u32,
    existing_points: &[(f32, f32)],
) -> Vec<(u32, u32, f32)> {
    let mut ctx = DetectorContext::new();
    ctx.detect_grid(
        image,
        grid_cols,
        grid_rows,
        max_per_cell,
        quality_level,
        min_distance,
        existing_points,
    );
    std::mem::take(&mut ctx.results)
}

/// Reusable owner of every buffer the Shi-Tomasi detector touches: gradient and
/// structure-tensor planes, the score plane, the candidate list, the spacing
/// grid and the result vector.
///
/// Keep one per detection thread and call [`detect`](Self::detect) or
/// [`detect_grid`](Self::detect_grid) each time. After the first call with a
/// given image size, further calls with the same image size, grid size and
/// spacing perform **no heap allocation**. The free functions
/// [`good_features_to_track`] and [`good_features_to_track_grid`] create a
/// context per call.
#[derive(Default)]
pub struct DetectorContext {
    planes: Planes,
    results: Vec<(u32, u32, f32)>,
    occupancy: OccupancyGrid,
    cell_count: Vec<u32>,
}

impl DetectorContext {
    /// Creates an empty context. Buffers grow to fit on the first detection
    /// and are reused thereafter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Detects corners as [`good_features_to_track`] does, returning them from
    /// the context's own result buffer. The slice is valid until the next call.
    pub fn detect(
        &mut self,
        image: &GrayImage,
        quality_level: f32,
        min_distance: u32,
    ) -> &[(u32, u32, f32)] {
        self.detect_candidates(image, quality_level);
        let (width, height) = image.dimensions();
        enforce_spacing(
            &self.planes.candidates,
            min_distance,
            width,
            height,
            &mut self.occupancy,
            &mut self.results,
        );
        &self.results
    }

    /// Detects corners per grid cell as [`good_features_to_track_grid`] does,
    /// returning them from the context's own result buffer. The slice is valid
    /// until the next call.
    #[allow(clippy::too_many_arguments)]
    pub fn detect_grid(
        &mut self,
        image: &GrayImage,
        grid_cols: u32,
        grid_rows: u32,
        max_per_cell: u32,
        quality_level: f32,
        min_distance: u32,
        existing_points: &[(f32, f32)],
    ) -> &[(u32, u32, f32)] {
        assert!(grid_cols > 0 && grid_rows > 0, "grid must be non-empty");

        self.results.clear();
        if max_per_cell == 0 {
            return &self.results;
        }
        self.detect_candidates(image, quality_level);
        if self.planes.candidates.is_empty() {
            return &self.results;
        }
        let (width, height) = image.dimensions();

        // Detection cell of a point, clamped to the grid.
        let cell_of = |x: f32, y: f32| -> usize {
            let col = ((x / width as f32) * grid_cols as f32) as u32;
            let row = ((y / height as f32) * grid_rows as f32) as u32;
            let col = col.min(grid_cols - 1);
            let row = row.min(grid_rows - 1);
            (row * grid_cols + col) as usize
        };

        // Per-cell occupancy (existing + accepted) against the feature budget.
        self.cell_count.clear();
        self.cell_count.resize((grid_cols * grid_rows) as usize, 0);
        // Spatial hash for min_distance, independent of the detection grid.
        self.occupancy.reset(width, height, min_distance);

        for &(ex, ey) in existing_points {
            self.cell_count[cell_of(ex, ey)] += 1;
            self.occupancy.insert(ex, ey);
        }

        let min_distance = min_distance as f32;
        for &(x, y, q) in &self.planes.candidates {
            let (xf, yf) = (x as f32, y as f32);
            let cell = cell_of(xf, yf);

            if self.cell_count[cell] >= max_per_cell {
                continue;
            }
            if self.occupancy.has_neighbor_within(xf, yf, min_distance) {
                continue;
            }

            self.occupancy.insert(xf, yf);
            self.cell_count[cell] += 1;
            self.results.push((x, y, q));
        }

        &self.results
    }

    /// Runs the Shi-Tomasi pipeline, leaving candidate corners sorted by
    /// descending quality in `self.planes.candidates`, before any spacing
    /// constraint.
    fn detect_candidates(&mut self, image: &GrayImage, quality_level: f32) {
        let candidates = &mut self.planes.candidates;
        candidates.clear();
        if image.width() < 3 || image.height() < 3 {
            return;
        }

        let max_quality = self.planes.run(image);
        let threshold = quality_level * max_quality;
        let candidates = &mut self.planes.candidates;
        candidates.retain(|&(_, _, q)| q >= threshold);

        // Descending quality; ties keep row-major order, which is the order the
        // candidates were collected in. The unstable sort does not allocate.
        candidates.sort_unstable_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(Ordering::Equal)
                .then(a.1.cmp(&b.1))
                .then(a.0.cmp(&b.0))
        });
    }
}

/// The per-pixel stages of the detector and their buffers.
#[derive(Default)]
struct Planes {
    grad_x: Vec<i16>,
    grad_y: Vec<i16>,
    ix_sq: Vec<i16>,
    iy_sq: Vec<i16>,
    ix_iy: Vec<i16>,
    scores: Vec<f32>,
    row_scratch: Vec<i16>,
    candidates: Vec<(u32, u32, f32)>,
}

impl Planes {
    /// Gradients, structure tensor, smoothing, scores and non-maximum
    /// suppression. Leaves the unthresholded local maxima in `candidates` in
    /// row-major order and returns the strongest score. The image must be at
    /// least 3x3.
    ///
    /// The stages are plain loops the compiler vectorizes for the target's
    /// baseline (SSE2, NEON, `simd128`); on x86 an AVX2 copy is selected at
    /// runtime.
    fn run(&mut self, image: &GrayImage) -> f32 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by the runtime feature check.
            return unsafe { run_avx2(self, image) };
        }
        run_impl(self, image)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn run_avx2(planes: &mut Planes, image: &GrayImage) -> f32 {
    run_impl(planes, image)
}

#[inline(always)]
fn run_impl(p: &mut Planes, image: &GrayImage) -> f32 {
    let (w, h) = (image.width() as usize, image.height() as usize);
    let n = w * h;
    for plane in [
        &mut p.grad_x,
        &mut p.grad_y,
        &mut p.ix_sq,
        &mut p.iy_sq,
        &mut p.ix_iy,
    ] {
        plane.resize(n, 0);
    }
    p.scores.resize(n, 0.0);

    compute_gradients_into(image, &mut p.grad_x, &mut p.grad_y);
    compute_gradient_products(
        &p.grad_x,
        &p.grad_y,
        &mut p.ix_sq,
        &mut p.iy_sq,
        &mut p.ix_iy,
    );
    for plane in [&mut p.ix_sq, &mut p.iy_sq, &mut p.ix_iy] {
        box_filter_3x3_raw(plane, w, h, &mut p.row_scratch);
    }
    compute_min_eigenvalues(&p.ix_sq, &p.iy_sq, &p.ix_iy, &mut p.scores);
    collect_local_maxima(&p.scores, w, h, &mut p.candidates)
}

/// Greedy minimum-distance filter over quality-sorted candidates, writing the
/// kept points into `out`.
fn enforce_spacing(
    candidates: &[(u32, u32, f32)],
    min_distance: u32,
    width: u32,
    height: u32,
    occupancy: &mut OccupancyGrid,
    out: &mut Vec<(u32, u32, f32)>,
) {
    out.clear();
    if candidates.is_empty() {
        return;
    }
    if min_distance == 0 {
        out.extend_from_slice(candidates);
        return;
    }
    occupancy.reset(width, height, min_distance);
    for &(x, y, q) in candidates {
        let (xf, yf) = (x as f32, y as f32);
        if !occupancy.has_neighbor_within(xf, yf, min_distance as f32) {
            // Multiple accepted points can occupy one cell (diagonally). Keep
            // every one so later candidates cannot hide an earlier neighbor.
            occupancy.insert(xf, yf);
            out.push((x, y, q));
        }
    }
}

/// Uniform spatial hash used to enforce `min_distance` between kept points.
///
/// Cells are `min_distance` wide, so any point closer than `min_distance` to a
/// query is guaranteed to live in the query's cell or one of its 8 neighbors.
/// Points in a cell form an intrusive linked list (`heads` / `next`), so the
/// grid reuses three flat vectors and never allocates once warm.
#[derive(Default)]
struct OccupancyGrid {
    cell_size: u32,
    cols: u32,
    rows: u32,
    heads: Vec<u32>,
    next: Vec<u32>,
    points: Vec<(f32, f32)>,
}

const NO_POINT: u32 = u32::MAX;

impl OccupancyGrid {
    fn reset(&mut self, width: u32, height: u32, min_distance: u32) {
        self.cell_size = min_distance.max(1);
        self.cols = width.div_ceil(self.cell_size).max(1);
        self.rows = height.div_ceil(self.cell_size).max(1);
        self.heads.clear();
        self.heads
            .resize((self.cols * self.rows) as usize, NO_POINT);
        self.next.clear();
        self.points.clear();
    }

    fn cell(&self, x: f32, y: f32) -> (u32, u32) {
        let col = (x.max(0.0) as u32 / self.cell_size).min(self.cols - 1);
        let row = (y.max(0.0) as u32 / self.cell_size).min(self.rows - 1);
        (col, row)
    }

    fn insert(&mut self, x: f32, y: f32) {
        let (col, row) = self.cell(x, y);
        let cell = (row * self.cols + col) as usize;
        let index = self.points.len() as u32;
        self.points.push((x, y));
        self.next.push(self.heads[cell]);
        self.heads[cell] = index;
    }

    fn has_neighbor_within(&self, x: f32, y: f32, min_distance: f32) -> bool {
        if min_distance <= 0.0 {
            return false;
        }

        let min_dist_sq = min_distance * min_distance;
        let (col, row) = self.cell(x, y);

        for drow in -1..=1 {
            for dcol in -1..=1 {
                let cx = col as i32 + dcol;
                let cy = row as i32 + drow;
                if cx < 0 || cy < 0 || cx >= self.cols as i32 || cy >= self.rows as i32 {
                    continue;
                }

                let mut index = self.heads[(cy as u32 * self.cols + cx as u32) as usize];
                while index != NO_POINT {
                    let (px, py) = self.points[index as usize];
                    let dist_sq = (x - px).powi(2) + (y - py).powi(2);
                    if dist_sq < min_dist_sq {
                        return true;
                    }
                    index = self.next[index as usize];
                }
            }
        }

        false
    }
}

/// Structure-tensor planes from Scharr gradients: each gradient is divided by
/// 32 with integer truncation first so the products fit in `i16`.
#[inline(always)]
fn compute_gradient_products(
    gx: &[i16],
    gy: &[i16],
    ix_sq: &mut [i16],
    iy_sq: &mut [i16],
    ix_iy: &mut [i16],
) {
    for ((((gx, gy), a), b), c) in gx
        .iter()
        .zip(gy)
        .zip(ix_sq.iter_mut())
        .zip(iy_sq.iter_mut())
        .zip(ix_iy.iter_mut())
    {
        let ix = gx / 32;
        let iy = gy / 32;
        *a = ix * ix;
        *b = iy * iy;
        *c = ix * iy;
    }
}

/// Minimum eigenvalue of the smoothed structure tensor at every pixel.
#[inline(always)]
fn compute_min_eigenvalues(a: &[i16], b: &[i16], c: &[i16], scores: &mut [f32]) {
    for (((a, b), c), score) in a.iter().zip(b).zip(c).zip(scores.iter_mut()) {
        // The tensor components have magnitude <= 16_129, so the determinant
        // and the discriminant are exact in i32. Evaluating the minimum
        // eigenvalue as 2*det / (trace + sqrt(disc)) instead of
        // (trace - sqrt(disc)) / 2 avoids the cancellation that would otherwise
        // turn an edge's exact zero response into f32 noise: the sign comes
        // from the exact determinant, and the rest is a single rounding.
        let (a, b, c) = (*a as i32, *b as i32, *c as i32);
        let det = a * b - c * c;
        let disc = (a - b) * (a - b) + 4 * c * c;
        let trace = (a + b) as f32;
        let value = 2.0 * det as f32 / (trace + (disc as f32).sqrt());
        // Integer gradient/filter rounding can make the tensor slightly
        // indefinite (det < 0); that is a zero corner response, as is a flat
        // tensor (det == 0, where the formula would divide 0 by 0).
        *score = if det > 0 { value } else { 0.0 };
    }
}

/// 3x3 non-maximum suppression over the interior of the score plane. A pixel
/// is kept when its score is positive and no neighbour scores strictly higher.
/// Candidates are appended in row-major order; returns the strongest score.
#[inline(always)]
fn collect_local_maxima(
    scores: &[f32],
    width: usize,
    height: usize,
    out: &mut Vec<(u32, u32, f32)>,
) -> f32 {
    let mut max_quality = 0.0f32;
    for y in 1..height - 1 {
        let above = &scores[(y - 1) * width..y * width];
        let current = &scores[y * width..(y + 1) * width];
        let below = &scores[(y + 1) * width..(y + 2) * width];
        for x in 1..width - 1 {
            let s = current[x];
            // A zero response is not a corner, including on a flat image.
            if s > 0.0
                && s >= current[x - 1]
                && s >= current[x + 1]
                && s >= above[x - 1]
                && s >= above[x]
                && s >= above[x + 1]
                && s >= below[x - 1]
                && s >= below[x]
                && s >= below[x + 1]
            {
                out.push((x as u32, y as u32, s));
                max_quality = max_quality.max(s);
            }
        }
    }
    max_quality
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spaced(candidates: &[(u32, u32, f32)], min_distance: u32) -> Vec<(u32, u32, f32)> {
        let mut occupancy = OccupancyGrid::default();
        let mut out = Vec::new();
        enforce_spacing(candidates, min_distance, 30, 30, &mut occupancy, &mut out);
        out
    }

    #[test]
    fn minimum_eigenvalues_match_analytic_tensors() {
        let tensors = [
            (4, 9, 0, 4.0),
            (5, 5, 4, 1.0),
            (5, 5, -4, 1.0),
            (4, 9, 6, 0.0),
            (0, 0, 0, 0.0),
            (16_129, 16_129, 0, 16_129.0),
            (16_129, 16_129, 16_129, 0.0),
            // Independent integer rounding can make a tensor slightly indefinite.
            (0, 0, 1, 0.0),
        ];
        let a: Vec<i16> = tensors.iter().map(|t| t.0).collect();
        let b: Vec<i16> = tensors.iter().map(|t| t.1).collect();
        let c: Vec<i16> = tensors.iter().map(|t| t.2).collect();
        let mut scores = vec![0f32; tensors.len()];
        compute_min_eigenvalues(&a, &b, &c, &mut scores);
        for (score, tensor) in scores.iter().zip(tensors) {
            assert!(score.is_finite());
            assert!(
                (score - tensor.3).abs() <= 1e-6 * f32::max(tensor.3, 1.0),
                "wrong score for {tensor:?}"
            );
        }
    }

    #[test]
    fn local_maxima_exclude_borders_zeros_and_dominated_pixels() {
        // 6x5 plane. Two equal adjacent scores are both kept; a pixel next to a
        // strictly higher one is suppressed; border pixels are never candidates.
        let width = 6;
        let mut scores = vec![0f32; width * 5];
        scores[width + 1] = 5.0;
        scores[width + 2] = 5.0;
        scores[3 * width + 4] = 7.0;
        scores[3 * width + 3] = 6.0; // neighbour of the 7, suppressed
        scores[4 * width] = 100.0; // border, never a candidate
        let mut out = Vec::new();
        let max = collect_local_maxima(&scores, width, 5, &mut out);
        assert_eq!(out, vec![(1, 1, 5.0), (2, 1, 5.0), (4, 3, 7.0)]);
        assert_eq!(max, 7.0);
    }

    #[test]
    fn spacing_keeps_all_accepted_points_in_each_cell() {
        let candidates = [(10, 10, 3.0), (19, 19, 2.0), (10, 11, 1.0)];
        assert_eq!(spaced(&candidates, 10), candidates[..2]);
    }

    #[test]
    fn zero_spacing_keeps_all_candidates_and_large_spacing_does_not_overflow() {
        let candidates = [(1, 1, 3.0), (1, 2, 2.0), (20, 20, 1.0)];
        assert_eq!(spaced(&candidates, 0), candidates);
        assert_eq!(spaced(&candidates, u32::MAX), candidates[..1]);
    }

    #[test]
    fn context_reuse_gives_identical_results_to_free_functions() {
        let image = GrayImage::from_fn(64, 48, |x, y| {
            image::Luma([((x * 31 + y * 17 + (x ^ y) * 13) & 255) as u8])
        });
        let mut ctx = DetectorContext::new();
        for _ in 0..2 {
            assert_eq!(
                ctx.detect(&image, 0.01, 5),
                good_features_to_track(&image, 0.01, 5)
            );
            let existing = [(10.0, 10.0), (40.0, 30.0)];
            assert_eq!(
                ctx.detect_grid(&image, 4, 3, 3, 0.01, 6, &existing),
                good_features_to_track_grid(&image, 4, 3, 3, 0.01, 6, &existing)
            );
        }
    }
}
