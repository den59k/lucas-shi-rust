use image::{GrayImage, ImageBuffer};
use std::cmp::Ordering;

use crate::utils::{
    box_filter_3x3::{sum_row_3, sum_row_3_f64, sum_rows_3, sum_rows_3_f64},
    fast_gradients::compute_gradients_into,
};

/// A detected corner: its pixel position and its Shi-Tomasi score, the
/// minimum eigenvalue of the mean 3x3 structure tensor.
///
/// Scores are in (grey levels / pixel)^2 for 8-bit images and in
/// (input unit / pixel)^2 for f32 images.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Corner {
    /// Column in pixels.
    pub x: u32,
    /// Row in pixels.
    pub y: u32,
    /// Corner score.
    pub score: f32,
}

impl Corner {
    /// The position as tracker input, `(x, y)` in pixels.
    pub fn position(&self) -> (f32, f32) {
        (self.x as f32, self.y as f32)
    }
}

impl From<(u32, u32, f32)> for Corner {
    fn from((x, y, score): (u32, u32, f32)) -> Self {
        Self { x, y, score }
    }
}

impl From<Corner> for (u32, u32, f32) {
    fn from(corner: Corner) -> Self {
        (corner.x, corner.y, corner.score)
    }
}

/// Detection grid of [`good_features_to_track_grid`] and `detect_grid`: the
/// image is split into `cols` x `rows` equal cells, and each cell keeps up to
/// `max_per_cell` corners, counting existing points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    /// Number of columns; must be non-zero.
    pub cols: u32,
    /// Number of rows; must be non-zero.
    pub rows: u32,
    /// Corner budget per cell, counting existing points in the cell.
    pub max_per_cell: u32,
}

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
/// [`Corner`]s sorted by descending score; equal scores retain row-major
/// order. Only finite, positive scores are returned. Flat images and images
/// narrower or shorter than three pixels yield no features. Scores are
/// eigenvalues of the mean 3x3 gradient tensor, using Scharr gradients
/// normalized by 32, in (grey levels / pixel)^2. Raw products and window
/// sums are accumulated exactly; normalization occurs once, at the final score.
pub fn good_features_to_track(
    image: &GrayImage,
    quality_level: f32,
    min_distance: u32,
) -> Vec<Corner> {
    let mut ctx = DetectorContext::new();
    ctx.detect(image, quality_level, min_distance);
    ctx.detector.take_results()
}

/// Finds good feature points with uniform frame coverage by detecting per grid
/// cell, while respecting features that are already being tracked.
///
/// For every cell of `grid` up to `max_per_cell` best Shi-Tomasi corners are
/// returned, where `min_distance` is respected both among the freshly detected
/// corners and against `existing_points`. Cells whose budget is already met by
/// `existing_points` are skipped, so this can be called every few frames to
/// top up lost tracks without clustering new detections on top of surviving
/// ones.
///
/// This is a convenience wrapper that allocates a fresh [`DetectorContext`];
/// per-frame callers should keep one context and call
/// [`DetectorContext::detect_grid`] instead.
///
/// # Arguments
/// * `image` - Target image (grayscale)
/// * `grid` - grid dimensions and per-cell budget, counting `existing_points`
/// * `quality_level` - Shi-Tomasi quality level, see [`good_features_to_track`]
/// * `min_distance` - minimum spacing in pixels between any two kept points
///   (new or existing)
/// * `existing_points` - already-tracked points (level-0 pixel coordinates) that
///   occupy budget and must be kept clear by `min_distance`
///
/// # Returns
/// Newly detected [`Corner`]s, sorted by descending score. `existing_points`
/// are never included in the output.
///
/// # Panics
/// Panics if `grid.cols` or `grid.rows` is zero.
pub fn good_features_to_track_grid(
    image: &GrayImage,
    grid: Grid,
    quality_level: f32,
    min_distance: u32,
    existing_points: &[(f32, f32)],
) -> Vec<Corner> {
    let mut ctx = DetectorContext::new();
    ctx.detect_grid(image, grid, quality_level, min_distance, existing_points);
    ctx.detector.take_results()
}

/// Upper bounds for the work a detector context will see, used to allocate
/// every buffer up front with [`DetectorContext::with_capacity`] or
/// [`DetectorContext::reserve`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DetectorCapacity {
    /// Largest image width in pixels.
    pub width: u32,
    /// Largest image height in pixels.
    pub height: u32,
    /// Smallest `min_distance` that will be passed; zero reserves for no
    /// spacing at all.
    pub min_distance: u32,
    /// Largest `grid_cols * grid_rows` that will be passed to `detect_grid`.
    pub grid_cells: u32,
    /// Most `existing_points` that will be passed to `detect_grid`.
    pub existing_points: usize,
}

impl DetectorCapacity {
    fn max_pixels(&self) -> usize {
        self.width as usize * self.height as usize
    }

    /// Interior pixels: every one can be a local maximum on a score plateau.
    fn max_candidates(&self) -> usize {
        self.width.saturating_sub(2) as usize * self.height.saturating_sub(2) as usize
    }

    /// Points that can be pairwise at least `min_distance` apart. Two points
    /// in one half-open square of side `min_distance / sqrt(2)` are closer, so
    /// each square holds at most one.
    fn max_spaced(&self) -> usize {
        let candidates = self.max_candidates();
        if self.min_distance < 2 {
            return candidates;
        }
        let side = self.min_distance as f64 / std::f64::consts::SQRT_2;
        let cells = |length: u32| (length as f64 / side).ceil() as usize + 1;
        candidates.min(cells(self.width) * cells(self.height))
    }
}

/// Grows `buffer`'s capacity to at least `total` elements.
pub(crate) fn reserve_total<T>(buffer: &mut Vec<T>, total: usize) {
    buffer.reserve(total.saturating_sub(buffer.len()));
}

/// Reusable owner of every buffer the Shi-Tomasi detector touches: gradient
/// planes, structure-tensor row buffers, the score plane, the candidate list,
/// the spacing grid and the result vector.
///
/// Keep one per detection thread and call [`detect`](Self::detect) or
/// [`detect_grid`](Self::detect_grid) each time. Buffers grow on demand and are
/// reused. To keep allocation out of the per-frame path, create the context
/// with [`with_capacity`](Self::with_capacity): calls within its
/// [`DetectorCapacity`] then perform **no heap allocation**, whatever the image
/// content. Without a reservation, the first call allocates, and a later image
/// with more candidates than any before can still grow a buffer. The free
/// functions [`good_features_to_track`] and [`good_features_to_track_grid`]
/// create a context per call.
#[derive(Default)]
pub struct DetectorContext {
    detector: Detector<i32>,
}

impl DetectorContext {
    /// Creates an empty context. Buffers grow to fit on the first detection
    /// and are reused thereafter.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a context with every buffer allocated for `capacity`.
    pub fn with_capacity(capacity: DetectorCapacity) -> Self {
        let mut context = Self::default();
        context.reserve(capacity);
        context
    }

    /// Allocates every buffer for `capacity` now, so that later calls within
    /// it do not allocate. Existing capacity is kept.
    pub fn reserve(&mut self, capacity: DetectorCapacity) {
        self.detector.reserve(&capacity);
    }

    /// Detects corners as [`good_features_to_track`] does, returning them from
    /// the context's own result buffer. The slice is valid until the next call.
    pub fn detect(
        &mut self,
        image: &GrayImage,
        quality_level: f32,
        min_distance: u32,
    ) -> &[Corner] {
        self.detector
            .detect(image, Threshold::relative(quality_level), min_distance)
    }

    /// Detects corners per grid cell as [`good_features_to_track_grid`] does,
    /// returning them from the context's own result buffer. The slice is valid
    /// until the next call.
    ///
    /// # Panics
    /// Panics if `grid.cols` or `grid.rows` is zero.
    pub fn detect_grid(
        &mut self,
        image: &GrayImage,
        grid: Grid,
        quality_level: f32,
        min_distance: u32,
        existing_points: &[(f32, f32)],
    ) -> &[Corner] {
        self.detector.detect_grid(
            image,
            grid,
            Threshold::relative(quality_level),
            min_distance,
            existing_points,
        )
    }
}

/// An image the detector runs on: its size and its Scharr gradients, whose
/// products the structure tensor sums in `Sum`.
pub(crate) trait DetectorImage: Copy {
    type Sum: TensorSum;

    fn dimensions(self) -> (u32, u32);

    /// Scharr gradients, 32 times the slope per pixel, zero on the border.
    fn gradients_into(
        self,
        grad_x: &mut [<Self::Sum as TensorSum>::Gradient],
        grad_y: &mut [<Self::Sum as TensorSum>::Gradient],
    );
}

impl DetectorImage for &GrayImage {
    type Sum = i32;

    fn dimensions(self) -> (u32, u32) {
        ImageBuffer::dimensions(self)
    }

    fn gradients_into(self, grad_x: &mut [i16], grad_y: &mut [i16]) {
        compute_gradients_into(self, grad_x, grad_y);
    }
}

/// Which local maxima a detection keeps: those scoring at least `absolute`
/// and at least `relative` times the strongest local maximum of the image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Threshold {
    pub(crate) absolute: f32,
    pub(crate) relative: f32,
}

impl Threshold {
    /// The 8-bit detector's `quality_level`, unvalidated as it always was.
    fn relative(quality_level: f32) -> Self {
        Self {
            absolute: 0.0,
            relative: quality_level,
        }
    }

    /// A score below which no pixel can pass, so NMS may skip it. The
    /// relative part costs a pass over the scores, saved when it is disabled.
    // Inlined into the pipeline, this pass slows the surrounding loops under
    // WASM by 5-10% (Node, simd128); kept out of line there, detection is
    // 2-9% faster than with the pass inlined. Native targets inline it, which
    // also gives it the AVX2 code of the pipeline on x86.
    #[cfg_attr(target_arch = "wasm32", inline(never))]
    #[cfg_attr(not(target_arch = "wasm32"), inline(always))]
    fn floor(&self, scores: &[f32], width: usize, height: usize) -> f32 {
        let relative = if self.relative > 0.0 {
            quality_floor(scores, width, height, self.relative)
        } else {
            0.0
        };
        self.absolute.max(relative)
    }

    /// Whether a local maximum scoring `score` passes, given the strongest.
    fn keeps(&self, score: f32, strongest: f32) -> bool {
        // Two comparisons rather than one against the larger bound, so that a
        // NaN quality level keeps nothing, as it always did.
        score >= self.absolute && score >= self.relative * strongest
    }
}

/// The detector pipeline with every buffer it touches, for tensor sums of
/// type `T`: the core of [`DetectorContext`] and of the f32 detector.
#[derive(Default)]
pub(crate) struct Detector<T: TensorSum> {
    planes: Planes<T>,
    selection: Selection,
}

impl<T: TensorSum> Detector<T> {
    pub(crate) fn reserve(&mut self, capacity: &DetectorCapacity) {
        self.planes.reserve(capacity);
        self.selection.reserve(capacity);
    }

    /// Corners passing `threshold`, greedily at least `min_distance` apart,
    /// from the result buffer.
    pub(crate) fn detect<I: DetectorImage<Sum = T>>(
        &mut self,
        image: I,
        threshold: Threshold,
        min_distance: u32,
    ) -> &[Corner] {
        self.detect_candidates(image, threshold);
        let (width, height) = image.dimensions();
        self.selection
            .spaced(&self.planes.candidates, min_distance, width, height)
    }

    /// Corners per grid cell, see [`DetectorContext::detect_grid`], from the
    /// result buffer.
    ///
    /// # Panics
    /// Panics if `grid.cols` or `grid.rows` is zero.
    pub(crate) fn detect_grid<I: DetectorImage<Sum = T>>(
        &mut self,
        image: I,
        grid: Grid,
        threshold: Threshold,
        min_distance: u32,
        existing_points: &[(f32, f32)],
    ) -> &[Corner] {
        assert!(grid.cols > 0 && grid.rows > 0, "grid must be non-empty");
        if grid.max_per_cell == 0 {
            return self.selection.clear();
        }
        self.detect_candidates(image, threshold);
        let (width, height) = image.dimensions();
        self.selection.grid(
            &self.planes.candidates,
            width,
            height,
            grid,
            min_distance,
            existing_points,
        )
    }

    /// Moves the result buffer out, for the allocating free functions.
    pub(crate) fn take_results(&mut self) -> Vec<Corner> {
        std::mem::take(&mut self.selection.results)
    }

    /// The score plane of the last detection.
    #[cfg(test)]
    pub(crate) fn scores(&self) -> &[f32] {
        &self.planes.scores
    }

    /// Runs the Shi-Tomasi pipeline, leaving the candidates that pass
    /// `threshold` sorted by descending score in `self.planes.candidates`,
    /// before any spacing constraint.
    fn detect_candidates<I: DetectorImage<Sum = T>>(&mut self, image: I, threshold: Threshold) {
        self.planes.candidates.clear();
        let (width, height) = image.dimensions();
        if width < 3 || height < 3 {
            return;
        }

        self.planes.run(image, threshold);
        let candidates = &mut self.planes.candidates;
        filter_by_threshold(candidates, threshold);
        sort_by_score(candidates);
    }
}

/// Descending score; ties keep row-major order, which is the order the
/// candidates were collected in. The unstable sort does not allocate.
fn sort_by_score(candidates: &mut [(u32, u32, f32)]) {
    candidates.sort_unstable_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap_or(Ordering::Equal)
            .then(a.1.cmp(&b.1))
            .then(a.0.cmp(&b.0))
    });
}

/// The per-pixel stages of a detector and their buffers, for tensor sums of
/// type `T`.
#[derive(Default)]
struct Planes<T: TensorSum> {
    grad_x: Vec<T::Gradient>,
    grad_y: Vec<T::Gradient>,
    tensor: TensorRows<T>,
    scores: Vec<f32>,
    /// Horizontal 3-pixel score maxima of rows y - 1, y and y + 1 for NMS.
    maxima_rows: [Vec<f32>; 3],
    candidates: Vec<(u32, u32, f32)>,
}

impl<T: TensorSum> Planes<T> {
    fn reserve(&mut self, capacity: &DetectorCapacity) {
        let (width, pixels) = (capacity.width as usize, capacity.max_pixels());
        reserve_total(&mut self.grad_x, pixels);
        reserve_total(&mut self.grad_y, pixels);
        reserve_total(&mut self.scores, pixels);
        self.tensor.reserve(width);
        for row in &mut self.maxima_rows {
            reserve_total(row, width.saturating_sub(2));
        }
        reserve_total(&mut self.candidates, capacity.max_candidates());
    }

    /// Gradients, structure tensor, scores and non-maximum suppression. Leaves
    /// the local maxima that can pass `threshold` in `candidates`, in
    /// row-major order. The image must be at least 3x3.
    ///
    /// The stages are plain loops compiled for the target's baseline (SSE2,
    /// NEON, `simd128`); on x86 an AVX2 copy is selected at runtime.
    fn run<I: DetectorImage<Sum = T>>(&mut self, image: I, threshold: Threshold) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by the runtime feature check.
            return unsafe { run_avx2(self, image, threshold) };
        }
        run_impl(self, image, threshold)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn run_avx2<I: DetectorImage>(planes: &mut Planes<I::Sum>, image: I, threshold: Threshold) {
    run_impl(planes, image, threshold)
}

#[inline(always)]
fn run_impl<I: DetectorImage>(p: &mut Planes<I::Sum>, image: I, threshold: Threshold) {
    let (w, h) = image.dimensions();
    let (w, h) = (w as usize, h as usize);
    let n = w * h;
    p.grad_x.resize(n, Default::default());
    p.grad_y.resize(n, Default::default());
    p.scores.resize(n, 0.0);

    image.gradients_into(&mut p.grad_x, &mut p.grad_y);
    min_eigenvalue_scores(&p.grad_x, &p.grad_y, w, &mut p.tensor, &mut p.scores);
    let floor = threshold.floor(&p.scores, w, h);
    local_maxima(
        &p.scores,
        w,
        h,
        floor,
        &mut p.maxima_rows,
        &mut p.candidates,
    );
}

/// Keeps the candidates that pass `threshold`, relative to the strongest one.
fn filter_by_threshold(candidates: &mut Vec<(u32, u32, f32)>, threshold: Threshold) {
    let strongest = candidates.iter().fold(0.0f32, |max, p| max.max(p.2));
    candidates.retain(|&(_, _, score)| threshold.keeps(score, strongest));
}

/// Greedy minimum-distance filter over quality-sorted candidates, writing the
/// kept points into `out`, as [`Corner`]s or as plain `(x, y, score)` triples.
fn enforce_spacing<O: From<(u32, u32, f32)>>(
    candidates: &[(u32, u32, f32)],
    min_distance: u32,
    width: u32,
    height: u32,
    occupancy: &mut OccupancyGrid,
    out: &mut Vec<O>,
) {
    out.clear();
    if candidates.is_empty() {
        return;
    }
    if min_distance == 0 {
        out.extend(candidates.iter().map(|&candidate| O::from(candidate)));
        return;
    }
    occupancy.reset(width, height, min_distance);
    for &(x, y, q) in candidates {
        let (xf, yf) = (x as f32, y as f32);
        if !occupancy.has_neighbor_within(xf, yf, min_distance as f32) {
            // Multiple accepted points can occupy one cell (diagonally). Keep
            // every one so later candidates cannot hide an earlier neighbor.
            occupancy.insert(xf, yf);
            out.push(O::from((x, y, q)));
        }
    }
}

/// Spacing and grid selection over score-sorted candidates, with the result
/// buffer.
#[derive(Default)]
struct Selection {
    results: Vec<Corner>,
    occupancy: OccupancyGrid,
    cell_count: Vec<u32>,
}

impl Selection {
    fn reserve(&mut self, capacity: &DetectorCapacity) {
        let spaced = capacity.max_spaced();
        reserve_total(&mut self.results, spaced);
        self.occupancy
            .reserve(capacity, capacity.existing_points + spaced);
        reserve_total(&mut self.cell_count, capacity.grid_cells as usize);
    }

    fn clear(&mut self) -> &[Corner] {
        self.results.clear();
        &self.results
    }

    /// Greedy minimum-distance selection, see [`enforce_spacing`].
    fn spaced(
        &mut self,
        candidates: &[(u32, u32, f32)],
        min_distance: u32,
        width: u32,
        height: u32,
    ) -> &[Corner] {
        enforce_spacing(
            candidates,
            min_distance,
            width,
            height,
            &mut self.occupancy,
            &mut self.results,
        );
        &self.results
    }

    /// Per-cell selection for [`DetectorContext::detect_grid`].
    fn grid(
        &mut self,
        candidates: &[(u32, u32, f32)],
        width: u32,
        height: u32,
        grid: Grid,
        min_distance: u32,
        existing_points: &[(f32, f32)],
    ) -> &[Corner] {
        self.results.clear();
        if candidates.is_empty() {
            return &self.results;
        }
        let Grid {
            cols: grid_cols,
            rows: grid_rows,
            max_per_cell,
        } = grid;

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
        for &(x, y, q) in candidates {
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
            self.results.push(Corner { x, y, score: q });
        }

        &self.results
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
    /// Allocates the cells for the smallest spacing in `capacity` and room
    /// for `points` points.
    fn reserve(&mut self, capacity: &DetectorCapacity, points: usize) {
        let cell_size = capacity.min_distance.max(1);
        let cells = |length: u32| length.div_ceil(cell_size).max(1) as usize;
        reserve_total(
            &mut self.heads,
            cells(capacity.width) * cells(capacity.height),
        );
        reserve_total(&mut self.next, points);
        reserve_total(&mut self.points, points);
    }

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

/// Accumulator type of the structure tensor and the gradient type it sums:
/// exact i32 sums of i16 Scharr gradients for 8-bit images, f64 sums of f32
/// gradients for f32 images. Products of two f32 values are exact in f64.
pub(crate) trait TensorSum: Copy + Default {
    type Gradient: Copy + Default;

    /// `(gx * gx, gy * gy, gx * gy)`.
    fn products(gx: Self::Gradient, gy: Self::Gradient) -> (Self, Self, Self);

    /// Horizontal pass of the 3x3 window sum, see [`sum_row_3`].
    fn sum_row_3(src: &[Self], dst: &mut [Self]);

    /// Vertical pass of the 3x3 window sum, see [`sum_rows_3`].
    fn sum_rows_3(
        above: Option<&[Self]>,
        center: &[Self],
        below: Option<&[Self]>,
        dst: &mut [Self],
    );

    /// Small eigenvalue of [[a, c], [c, b]] for summed gradient products,
    /// divided by `normalization`.
    fn scaled_min_eigenvalue(a: Self, b: Self, c: Self, normalization: f32) -> f32;
}

impl TensorSum for i32 {
    type Gradient = i16;

    #[inline(always)]
    fn products(gx: i16, gy: i16) -> (i32, i32, i32) {
        let (ix, iy) = (gx as i32, gy as i32);
        (ix * ix, iy * iy, ix * iy)
    }

    #[inline(always)]
    fn sum_row_3(src: &[i32], dst: &mut [i32]) {
        sum_row_3(src, dst);
    }

    #[inline(always)]
    fn sum_rows_3(above: Option<&[i32]>, center: &[i32], below: Option<&[i32]>, dst: &mut [i32]) {
        sum_rows_3(above, center, below, dst);
    }

    #[inline(always)]
    fn scaled_min_eigenvalue(a: i32, b: i32, c: i32, normalization: f32) -> f32 {
        scaled_min_eigenvalue(a, b, c, normalization)
    }
}

impl TensorSum for f64 {
    type Gradient = f32;

    #[inline(always)]
    fn products(gx: f32, gy: f32) -> (f64, f64, f64) {
        let (ix, iy) = (gx as f64, gy as f64);
        (ix * ix, iy * iy, ix * iy)
    }

    #[inline(always)]
    fn sum_row_3(src: &[f64], dst: &mut [f64]) {
        sum_row_3_f64(src, dst);
    }

    #[inline(always)]
    fn sum_rows_3(above: Option<&[f64]>, center: &[f64], below: Option<&[f64]>, dst: &mut [f64]) {
        sum_rows_3_f64(above, center, below, dst);
    }

    #[inline(always)]
    fn scaled_min_eigenvalue(a: f64, b: f64, c: f64, normalization: f32) -> f32 {
        // As for i32 sums, with the determinant in f64. Products are exact,
        // so its only rounding comes from the window sums, the two products
        // and their difference, far below the precision of f32 inputs. Pixel
        // magnitudes within FloatImageView::MAX_ABS_VALUE keep the determinant
        // and every f32 intermediate finite.
        let determinant = a * b - c * c;
        let (trace, difference, c) = ((a + b) as f32, (a - b) as f32, c as f32);
        let largest = (trace + (difference * difference + 4.0 * c * c).sqrt()) * 0.5;
        let score = determinant as f32 / (largest * normalization);
        if determinant > 0.0 { score } else { 0.0 }
    }
}

/// One image row of the structure tensor components `[[xx, xy], [xy, yy]]`:
/// raw gradient products, or their horizontal or full 3x3 window sums.
#[derive(Default)]
struct TensorRow<T> {
    xx: Vec<T>,
    yy: Vec<T>,
    xy: Vec<T>,
}

impl<T: TensorSum> TensorRow<T> {
    #[cfg(test)]
    fn new(width: usize) -> Self {
        let mut row = Self::default();
        row.resize(width);
        row
    }

    fn resize(&mut self, width: usize) {
        for plane in [&mut self.xx, &mut self.yy, &mut self.xy] {
            plane.resize(width, T::default());
        }
    }

    fn reserve(&mut self, width: usize) {
        for plane in [&mut self.xx, &mut self.yy, &mut self.xy] {
            reserve_total(plane, width);
        }
    }

    fn planes(&self) -> [&[T]; 3] {
        [&self.xx, &self.yy, &self.xy]
    }

    fn planes_mut(&mut self) -> [&mut [T]; 3] {
        [&mut self.xx, &mut self.yy, &mut self.xy]
    }

    /// Vertical 3-row sum of horizontally summed rows; `None` is outside the image.
    #[inline(always)]
    fn sum_rows(&mut self, above: Option<&Self>, center: &Self, below: Option<&Self>) {
        let (above, below) = (above.map(Self::planes), below.map(Self::planes));
        for (k, dst) in self.planes_mut().into_iter().enumerate() {
            let (a, b) = (above.map(|p| p[k]), below.map(|p| p[k]));
            T::sum_rows_3(a, center.planes()[k], b, dst);
        }
    }
}

/// Row buffers of the fused tensor pass: raw products of one row, horizontal
/// sums of rows y - 1, y and y + 1 (stored at row % 3), and the 3x3 window
/// sums of row y.
#[derive(Default)]
struct TensorRows<T> {
    products: TensorRow<T>,
    ring: [TensorRow<T>; 3],
    window: TensorRow<T>,
}

impl<T: TensorSum> TensorRows<T> {
    fn reserve(&mut self, width: usize) {
        let Self {
            products,
            ring,
            window,
        } = self;
        for row in [products, window].into_iter().chain(ring) {
            row.reserve(width);
        }
    }
}

/// Raw Scharr products of one row. For u8 images, |Scharr| <= 4080: products
/// <= 16_646_400 and 3x3 sums <= 149_817_600 fit in i32. Deferring the /32
/// normalization avoids discarding sub-grey-level gradients before squaring.
#[inline(always)]
fn gradient_products_row<T: TensorSum>(
    gx: &[T::Gradient],
    gy: &[T::Gradient],
    products: &mut TensorRow<T>,
) {
    let rows = products
        .xx
        .iter_mut()
        .zip(&mut products.yy)
        .zip(&mut products.xy)
        .zip(gx.iter().zip(gy));
    for (((xx, yy), xy), (&ix, &iy)) in rows {
        (*xx, *yy, *xy) = T::products(ix, iy);
    }
}

/// Products of one gradient row, summed horizontally over three pixels.
#[inline(always)]
fn horizontal_tensor_row<T: TensorSum>(
    gx: &[T::Gradient],
    gy: &[T::Gradient],
    products: &mut TensorRow<T>,
    sums: &mut TensorRow<T>,
) {
    gradient_products_row(gx, gy, products);
    T::sum_row_3(&products.xx, &mut sums.xx);
    T::sum_row_3(&products.yy, &mut sums.yy);
    T::sum_row_3(&products.xy, &mut sums.xy);
}

/// Minimum eigenvalue of the mean 3x3 structure tensor at every pixel of a
/// row-major `width`-wide plane. Products and window sums are computed one row
/// at a time, keeping only three horizontally summed rows instead of tensor
/// planes.
#[inline(always)]
fn min_eigenvalue_scores<T: TensorSum>(
    gx: &[T::Gradient],
    gy: &[T::Gradient],
    width: usize,
    rows: &mut TensorRows<T>,
    scores: &mut [f32],
) {
    if width == 0 || scores.is_empty() {
        return;
    }
    let height = scores.len() / width;
    let TensorRows {
        products,
        ring,
        window,
    } = rows;
    for row in [&mut *products, &mut *window]
        .into_iter()
        .chain(ring.iter_mut())
    {
        row.resize(width);
    }

    let row = |y: usize| y * width..(y + 1) * width;
    horizontal_tensor_row(&gx[row(0)], &gy[row(0)], products, &mut ring[0]);
    for y in 0..height {
        if y + 1 < height {
            let next = &mut ring[(y + 1) % 3];
            horizontal_tensor_row(&gx[row(y + 1)], &gy[row(y + 1)], products, next);
        }
        let above = (y > 0).then(|| &ring[(y + 2) % 3]);
        let below = (y + 1 < height).then(|| &ring[(y + 1) % 3]);
        window.sum_rows(above, &ring[y % 3], below);
        score_row(window, y as u32, height as u32, &mut scores[row(y)]);
    }
}

/// Normalized minimum eigenvalues of one row of 3x3 window sums.
#[inline(always)]
fn score_row<T: TensorSum>(window: &TensorRow<T>, y: u32, height: u32, scores: &mut [f32]) {
    let width = scores.len();
    let rows = 1 + u32::from(y > 0) + u32::from(y + 1 < height);
    // Eigenvalues scale linearly with a common tensor scale. Divide once by
    // Scharr gain squared and the actual sample count, including edges.
    let normalization = |cols: u32| (32 * 32 * cols * rows) as f32;
    let interior = normalization(3);
    let sums = window.xx.iter().zip(&window.yy).zip(&window.xy);
    for (score, ((&a, &b), &c)) in scores.iter_mut().zip(sums) {
        *score = T::scaled_min_eigenvalue(a, b, c, interior);
    }
    // Boundary columns average fewer samples; a single column has none beside it.
    let edge = normalization(if width == 1 { 1 } else { 2 });
    for x in [0, width - 1] {
        scores[x] = T::scaled_min_eigenvalue(window.xx[x], window.yy[x], window.xy[x], edge);
    }
}

/// Small eigenvalue of [[a, c], [c, b]] for summed gradient products, divided
/// by `normalization`.
#[inline(always)]
fn scaled_min_eigenvalue(a: i32, b: i32, c: i32, normalization: f32) -> f32 {
    // Compute the determinant exactly in i64: i32 tensor sums can have a
    // discriminant larger than i32, and ab - c^2 cancels for nearly rank-one
    // tensors. Then det / lambda_max leaves no cancellation, so f32 suffices
    // for the rest: lambda_max only adds non-negative terms. Valid u8 tensors
    // have |entry| <= 149_817_600, so i64 is ample and a + b fits in i32.
    let determinant = a as i64 * b as i64 - c as i64 * c as i64;
    let (trace, difference, c) = ((a + b) as f32, (a - b) as f32, c as f32);
    let largest = (trace + (difference * difference + 4.0 * c * c).sqrt()) * 0.5;
    // Select instead of an early return so the row loop vectorizes; the
    // discarded lane of a zero tensor computes 0 / 0.
    let score = determinant as f32 / (largest * normalization);
    if determinant > 0 { score } else { 0.0 }
}

/// A score below which no pixel can pass the quality filter, so NMS may skip it.
///
/// The quality threshold is relative to the strongest local maximum. A global
/// maximum inside the image is one, which fixes the threshold before NMS. NMS
/// never selects border pixels, so a maximum found only on the border does not;
/// nothing beyond non-positive scores is skipped then.
#[inline(always)]
fn quality_floor(scores: &[f32], width: usize, height: usize, quality_level: f32) -> f32 {
    let (mut interior, mut border) = (0.0f32, 0.0f32);
    for (y, row) in scores.chunks_exact(width).enumerate() {
        if y == 0 || y + 1 == height {
            border = max_of(row, border);
        } else {
            interior = max_of(&row[1..width - 1], interior);
            border = max_score(max_score(border, row[0]), row[width - 1]);
        }
    }
    if interior >= border {
        quality_level * interior
    } else {
        0.0
    }
}

/// Appends the positive interior local maxima at or above `floor` to `maxima`,
/// in row-major order. A pixel is kept when no 8-neighbor has a greater score,
/// so plateaus survive; it equals the maximum of its 3x3 neighborhood, which is
/// computed separably in `rows`.
///
/// Almost all pixels fail the test, so each row is tested in groups of
/// [`MAXIMA_GROUP`] pixels first, without branches, which vectorizes. Only
/// groups containing a kept pixel are scanned one by one.
#[inline(always)]
fn local_maxima(
    scores: &[f32],
    width: usize,
    height: usize,
    floor: f32,
    rows: &mut [Vec<f32>; 3],
    maxima: &mut Vec<(u32, u32, f32)>,
) {
    if width < 3 || height < 3 {
        return;
    }
    for row in rows.iter_mut() {
        row.resize(width - 2, 0.0);
    }
    let row = |y: usize| &scores[y * width..(y + 1) * width];
    // Horizontal 3-pixel maxima for the interior columns of rows y - 1, y and
    // y + 1, stored at row % 3.
    max_row_3(row(0), &mut rows[0]);
    max_row_3(row(1), &mut rows[1]);
    for y in 1..height - 1 {
        max_row_3(row(y + 1), &mut rows[(y + 1) % 3]);
        let (above, center, below) = (&rows[(y + 2) % 3], &rows[y % 3], &rows[(y + 1) % 3]);
        let interior = &row(y)[1..width - 1];
        let groups = interior
            .as_chunks::<MAXIMA_GROUP>()
            .0
            .iter()
            .zip(above.as_chunks().0)
            .zip(center.as_chunks().0)
            .zip(below.as_chunks().0);
        let mut start = 0;
        for (((s, a), c), b) in groups {
            let mut mask = kept_mask(s, a, c, b, floor);
            while mask != 0 {
                let i = start + mask.trailing_zeros() as usize;
                maxima.push((i as u32 + 1, y as u32, interior[i]));
                mask &= mask - 1;
            }
            start += MAXIMA_GROUP;
        }
        for i in start..interior.len() {
            let neighborhood = max_score(max_score(above[i], center[i]), below[i]);
            if is_kept(interior[i], neighborhood, floor) {
                maxima.push((i as u32 + 1, y as u32, interior[i]));
            }
        }
    }
}

/// Pixels per branch-free group in [`local_maxima`].
const MAXIMA_GROUP: usize = 8;

/// Whether [`local_maxima`] keeps a pixel with `score` whose 3x3 neighborhood
/// maximum is `neighborhood`. Evaluates all three comparisons, without
/// branches.
#[inline(always)]
fn is_kept(score: f32, neighborhood: f32, floor: f32) -> bool {
    (score > 0.0) & (score >= floor) & (score == neighborhood)
}

/// Bit `k` is set when [`local_maxima`] keeps pixel `k` of a group, given the
/// horizontal 3-pixel maxima of the rows above, at and below it.
#[inline(always)]
fn kept_mask(
    scores: &[f32; MAXIMA_GROUP],
    above: &[f32; MAXIMA_GROUP],
    center: &[f32; MAXIMA_GROUP],
    below: &[f32; MAXIMA_GROUP],
    floor: f32,
) -> u32 {
    let pixels = scores.iter().zip(above).zip(center).zip(below);
    let mask = |(k, (((&score, &above), &center), &below))| {
        let neighborhood = max_score(max_score(above, center), below);
        u32::from(is_kept(score, neighborhood, floor)) << k
    };
    pixels.enumerate().map(mask).fold(0, |all, bit| all | bit)
}

/// `dst[i] = max(src[i], src[i + 1], src[i + 2])`, one value per interior column.
#[inline(always)]
fn max_row_3(src: &[f32], dst: &mut [f32]) {
    let windows = src.iter().zip(&src[1..]).zip(&src[2..]);
    for (value, ((&left, &center), &right)) in dst.iter_mut().zip(windows) {
        *value = max_score(max_score(left, center), right);
    }
}

/// Maximum of two scores. Scores are never NaN, so both forms below are exact.
/// `f32::max` is one instruction on aarch64 (`fmaxnm`); elsewhere, such as on
/// WASM, its NaN handling costs extra instructions, so compare instead.
#[inline(always)]
fn max_score(a: f32, b: f32) -> f32 {
    if cfg!(target_arch = "aarch64") {
        a.max(b)
    } else if a > b {
        a
    } else {
        b
    }
}

/// Maximum of `init` and all `scores`. The maximum is exact in any order, so
/// independent lanes may reduce in parallel, which lets the loop vectorize.
#[inline(always)]
fn max_of(scores: &[f32], init: f32) -> f32 {
    let mut lanes = [init; 8];
    let chunks = scores.chunks_exact(lanes.len());
    let rest = chunks.remainder();
    for chunk in chunks {
        for (lane, &score) in lanes.iter_mut().zip(chunk) {
            *lane = max_score(*lane, score);
        }
    }
    let max = |max, &score: &f32| max_score(max, score);
    rest.iter().chain(&lanes).fold(init, max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::box_filter_3x3::box_sum_3x3_in_place;
    use image::{ImageBuffer, Luma};

    fn spaced(candidates: &[(u32, u32, f32)], min_distance: u32) -> Vec<(u32, u32, f32)> {
        let mut occupancy = OccupancyGrid::default();
        let mut out = Vec::new();
        enforce_spacing(candidates, min_distance, 30, 30, &mut occupancy, &mut out);
        out
    }

    fn maxima(scores: &[f32], width: usize, height: usize, floor: f32) -> Vec<(u32, u32, f32)> {
        let mut rows = Default::default();
        let mut out = Vec::new();
        local_maxima(scores, width, height, floor, &mut rows, &mut out);
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
            // Clamp indefinite tensors to zero as well.
            (0, 0, 1, 0.0),
        ];
        for tensor in tensors {
            let score = scaled_min_eigenvalue(tensor.0, tensor.1, tensor.2, 1.0);
            assert!(score.is_finite());
            assert!(
                (score - tensor.3).abs() <= 1e-6 * tensor.3.max(1.0),
                "wrong score for {tensor:?}"
            );
        }
    }

    type Plane = ImageBuffer<Luma<i32>, Vec<i32>>;

    /// Score plane of full-image window-sum planes, via the production row scorer.
    fn scores_of_window_planes(a: &Plane, b: &Plane, c: &Plane) -> Vec<f32> {
        let (width, height) = a.dimensions();
        let mut scores = vec![0.0; (width * height) as usize];
        let mut window = TensorRow::<i32>::new(width as usize);
        for y in 0..height {
            let row = (y * width) as usize..((y + 1) * width) as usize;
            window.xx.copy_from_slice(&a.as_raw()[row.clone()]);
            window.yy.copy_from_slice(&b.as_raw()[row.clone()]);
            window.xy.copy_from_slice(&c.as_raw()[row.clone()]);
            score_row(&window, y, height, &mut scores[row]);
        }
        scores
    }

    #[test]
    fn raw_products_preserve_fractional_normalized_gradients_and_extremes() {
        let gx = [1i16, -31, 32, 4080, -4080];
        let gy = [-1i16, 1, 31, -4080, -4080];
        let mut products = TensorRow::<i32>::new(5);
        gradient_products_row(&gx, &gy, &mut products);
        assert_eq!(products.xx, [1, 961, 1024, 16_646_400, 16_646_400]);
        assert_eq!(products.yy, [1, 1, 961, 16_646_400, 16_646_400]);
        assert_eq!(products.xy, [-1, -31, 992, -16_646_400, 16_646_400]);
    }

    #[test]
    fn row_pipeline_matches_full_plane_box_sums_for_small_and_odd_sizes() {
        // One set of row buffers across sizes also checks that resizing them
        // between calls leaves no stale values behind.
        let mut rows = TensorRows::<i32>::default();
        for width in 1..8 {
            for height in 1..8 {
                let gradient = |seed: u32| {
                    ImageBuffer::from_fn(width, height, |x, y| {
                        let hash = (x * 7 + y * 13 + seed).wrapping_mul(2_654_435_761) >> 19;
                        Luma([(hash % 8161) as i16 - 4080])
                    })
                };
                let (gx, gy) = (gradient(1), gradient(2));
                let plane = |product: fn(i32, i32) -> i32| {
                    let mut plane = ImageBuffer::from_fn(width, height, |x, y| {
                        let (ix, iy) = (gx.get_pixel(x, y)[0], gy.get_pixel(x, y)[0]);
                        Luma([product(ix as i32, iy as i32)])
                    });
                    box_sum_3x3_in_place(&mut plane);
                    plane
                };
                let expected = scores_of_window_planes(
                    &plane(|ix, _| ix * ix),
                    &plane(|_, iy| iy * iy),
                    &plane(|ix, iy| ix * iy),
                );
                let mut scores = vec![0.0; (width * height) as usize];
                min_eigenvalue_scores(
                    gx.as_raw(),
                    gy.as_raw(),
                    width as usize,
                    &mut rows,
                    &mut scores,
                );
                assert_eq!(scores, expected, "{width}x{height}");
            }
        }
    }

    #[test]
    fn normalization_preserves_tensor_mean_at_borders_and_small_dimensions() {
        for (width, height) in [(1, 1), (1, 5), (5, 1), (2, 2), (3, 3), (7, 9)] {
            let mut a = ImageBuffer::from_pixel(width, height, Luma([1024]));
            let mut b = ImageBuffer::from_pixel(width, height, Luma([2048]));
            let c = ImageBuffer::new(width, height);
            box_sum_3x3_in_place(&mut a);
            box_sum_3x3_in_place(&mut b);
            let scores = scores_of_window_planes(&a, &b, &c);
            assert_eq!(scores.len(), (width * height) as usize);
            assert!(scores.iter().all(|&score| score == 1.0));
        }
    }

    #[test]
    fn fractional_window_mean_is_not_truncated() {
        let a = ImageBuffer::from_pixel(3, 3, Luma([1]));
        let b = a.clone();
        let c = ImageBuffer::new(3, 3);
        let scores = scores_of_window_planes(&a, &b, &c);
        let expected = 1.0 / (32.0 * 32.0 * 9.0);
        assert_eq!(scores[4], expected);
    }

    /// Positive interior pixels without a strictly greater 8-neighbor.
    fn brute_force_maxima(scores: &[f32], width: usize, height: usize) -> Vec<(u32, u32, f32)> {
        let mut maxima = Vec::new();
        for y in 1..height.saturating_sub(1) {
            for x in 1..width.saturating_sub(1) {
                let score = scores[y * width + x];
                let neighbor = |nx: usize, ny: usize| scores[ny * width + nx];
                let is_max =
                    (y - 1..=y + 1).all(|ny| (x - 1..=x + 1).all(|nx| neighbor(nx, ny) <= score));
                if score > 0.0 && is_max {
                    maxima.push((x as u32, y as u32, score));
                }
            }
        }
        maxima
    }

    #[test]
    fn local_maxima_match_brute_force_with_ties_and_floors() {
        let mut state = 0x2545_f491u32;
        // Interior widths cover no, exactly one, and several full groups of
        // MAXIMA_GROUP pixels, with and without a remainder.
        for (width, height) in [(3, 3), (4, 3), (3, 7), (8, 5), (10, 4), (17, 11), (27, 6)] {
            for levels in [2, 4, 1000] {
                // Few distinct levels produce plateaus; zero scores are never maxima.
                let scores: Vec<f32> = (0..width * height)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        (state % levels) as f32
                    })
                    .collect();
                let expected = brute_force_maxima(&scores, width, height);
                assert_eq!(maxima(&scores, width, height, 0.0), expected);
                for floor in [0.5, 1.0, 3.0, 999.0] {
                    let mut above = expected.clone();
                    above.retain(|p| p.2 >= floor);
                    assert_eq!(maxima(&scores, width, height, floor), above);
                }
            }
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
        assert_eq!(
            maxima(&scores, width, 5, 0.0),
            vec![(1, 1, 5.0), (2, 1, 5.0), (4, 3, 7.0)]
        );
    }

    #[test]
    fn quality_floor_is_skipped_when_the_maximum_is_only_on_the_border() {
        // The border pixel (0, 0) suppresses (1, 1); the strongest local maximum
        // is (3, 2), so a floor from the global maximum would drop it.
        #[rustfmt::skip]
        let scores = [
            9.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 5.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 2.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(quality_floor(&scores, 5, 4, 0.5), 0.0);
        assert_eq!(maxima(&scores, 5, 4, 0.0), [(3, 2, 2.0)]);

        let mut interior = scores;
        interior[0] = 1.0;
        assert_eq!(quality_floor(&interior, 5, 4, 0.5), 2.5);
        assert_eq!(maxima(&interior, 5, 4, 2.5), [(1, 1, 5.0)]);
    }

    #[test]
    fn nearly_rank_one_tensor_keeps_its_small_positive_eigenvalue() {
        // Outer products of (4080, 4079) and (4079, 4078) have determinant 1.
        let (a, b, c) = (33_284_641, 33_268_325, 33_276_482);
        assert_eq!(a as i64 * b as i64 - c as i64 * c as i64, 1);
        let score = scaled_min_eigenvalue(a, b, c, 1.0) as f64;
        let expected = 1.0 / (a as f64 + b as f64);
        assert!((score - expected).abs() <= expected * 1e-6);
    }

    #[test]
    fn maximum_product_sum_is_finite_and_rank_one_stays_zero() {
        let max = 9 * 4080 * 4080;
        let score = scaled_min_eigenvalue(max, max, 0, 1.0);
        assert!((score - max as f32).abs() <= max as f32 * 1e-6);
        assert_eq!(scaled_min_eigenvalue(max, max, max, 1.0), 0.0);
        assert_eq!(scaled_min_eigenvalue(max, max, -max, 1.0), 0.0);
    }

    #[test]
    fn f32_score_matches_f64_eigenvalue_of_window_tensors() {
        // Sums of nine gradient outer products, including nearly collinear ones.
        let mut state = 0x9e37_79b9u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state % 8161) as i32 - 4080
        };
        for trial in 0..20_000 {
            let (dx, dy) = (next(), next());
            let (mut a, mut b, mut c) = (0, 0, 0);
            for _ in 0..9 {
                let jitter = if trial % 2 == 0 { 4080 } else { 3 };
                let scale = next().signum();
                let ix = (scale * dx + next() % jitter).clamp(-4080, 4080);
                let iy = (scale * dy + next() % jitter).clamp(-4080, 4080);
                (a, b, c) = (a + ix * ix, b + iy * iy, c + ix * iy);
            }
            let determinant = a as i64 * b as i64 - c as i64 * c as i64;
            let (af, bf, cf) = (a as f64, b as f64, c as f64);
            let largest = (af + bf + ((af - bf).powi(2) + 4.0 * cf * cf).sqrt()) / 2.0;
            let expected = (determinant.max(0) as f64 / largest) as f32;
            let score = scaled_min_eigenvalue(a, b, c, 1.0);
            assert!(
                (score - expected).abs() <= expected * 1e-6,
                "{a} {b} {c}: {score} vs {expected}"
            );
        }
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
    fn reservation_covers_plateaus_and_densest_spacing() {
        // On a plateau every interior pixel is a local maximum.
        let capacity = DetectorCapacity {
            width: 9,
            height: 7,
            ..Default::default()
        };
        let mut planes = Planes::<i32>::default();
        planes.reserve(&capacity);
        let buffer = planes.candidates.as_ptr();
        let scores = vec![1.0; 9 * 7];
        local_maxima(
            &scores,
            9,
            7,
            0.0,
            &mut planes.maxima_rows,
            &mut planes.candidates,
        );
        assert_eq!(planes.candidates.len(), capacity.max_candidates());
        assert_eq!(planes.candidates.as_ptr(), buffer);

        // Greedy spacing over every interior pixel packs points most densely.
        for min_distance in [0, 1, 2, 3, 5, 8, 13, 100] {
            let capacity = DetectorCapacity {
                width: 61,
                height: 47,
                min_distance,
                ..Default::default()
            };
            let candidates: Vec<_> = (1..46)
                .flat_map(|y| (1..60).map(move |x| (x, y, 1.0)))
                .collect();
            let mut selection = Selection::default();
            selection.reserve(&capacity);
            let buffer = selection.results.as_ptr();
            let kept = selection.spaced(&candidates, min_distance, 61, 47).len();
            assert!(kept <= capacity.max_spaced(), "{min_distance}: {kept}");
            assert_eq!(selection.results.as_ptr(), buffer);
        }
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
            let grid = Grid {
                cols: 4,
                rows: 3,
                max_per_cell: 3,
            };
            assert_eq!(
                ctx.detect_grid(&image, grid, 0.01, 6, &existing),
                good_features_to_track_grid(&image, grid, 0.01, 6, &existing)
            );
        }
    }

    #[test]
    fn threshold_requires_both_parts() {
        let both = Threshold {
            absolute: 2.0,
            relative: 0.1,
        };
        assert!(both.keeps(10.0, 100.0) && !both.keeps(9.9, 100.0));
        assert!(both.keeps(2.0, 5.0) && !both.keeps(1.9, 5.0));
        // 8-bit quality levels are not validated: NaN keeps nothing, a
        // negative level every positive score.
        assert!(!Threshold::relative(f32::NAN).keeps(1.0, 1.0));
        assert!(Threshold::relative(-1.0).keeps(1e-30, 1.0));
    }

    /// Minimum eigenvalue of a window tensor in f64 throughout.
    fn reference(a: f64, b: f64, c: f64) -> f64 {
        let determinant = a * b - c * c;
        let largest = (a + b + ((a - b).powi(2) + 4.0 * c * c).sqrt()) / 2.0;
        if determinant > 0.0 {
            determinant / largest
        } else {
            0.0
        }
    }

    #[test]
    fn f64_tensor_scores_match_reference_for_real_valued_gradients() {
        let mut state = 0x2545_f491u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as f32 / u32::MAX as f32 * 2.0 - 1.0
        };
        // Gradient magnitudes from tiny physical slopes to the largest
        // permitted ones, with nearly collinear windows. Below about 1e-6,
        // nearly collinear windows lose precision once the determinant is a
        // subnormal f32; their scores are then negligible anyway.
        for scale in [1e-6f32, 1e-3, 0.37, 41.0, 5e8] {
            for collinear in [false, true] {
                let (dx, dy) = (next(), next());
                let (mut a, mut b, mut c) = (0.0f64, 0.0f64, 0.0f64);
                for _ in 0..9 {
                    let jitter = if collinear { 1e-3 } else { 1.0 };
                    let t = next();
                    let gx = (t * dx + next() * jitter) * scale;
                    let gy = (t * dy + next() * jitter) * scale;
                    let (xx, yy, xy) = <f64 as TensorSum>::products(gx, gy);
                    (a, b, c) = (a + xx, b + yy, c + xy);
                }
                let score = <f64 as TensorSum>::scaled_min_eigenvalue(a, b, c, 1.0);
                let expected = reference(a, b, c);
                assert!(
                    (score as f64 - expected).abs() <= expected * 1e-5,
                    "scale {scale}, collinear {collinear}: {score} vs {expected}"
                );
            }
        }
    }
}
