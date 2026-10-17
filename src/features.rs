use image::{GrayImage, ImageBuffer, Luma};
use std::cmp::Ordering;

use crate::utils::{
    box_filter_3x3::{sum_row_3, sum_rows_3},
    fast_gradients::compute_gradients,
};

/// Finds good features points using the Shi-Tomasi algorithm
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
/// Scores are eigenvalues of the mean 3x3 gradient tensor, using Scharr gradients
/// normalized by 32, in (grey levels / pixel)^2. Raw products and window
/// sums are accumulated exactly; normalization occurs once, at the final score.
pub fn good_features_to_track(
    image: &GrayImage,
    quality_level: f32,
    min_distance: u32,
) -> Vec<(u32, u32, f32)> {
    let features = detect_candidates(image, quality_level);

    // Filter by distance
    filter_by_distance(&features, min_distance, image.width(), image.height())
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
pub fn good_features_to_track_grid(
    image: &GrayImage,
    grid_cols: u32,
    grid_rows: u32,
    max_per_cell: u32,
    quality_level: f32,
    min_distance: u32,
    existing_points: &[(f32, f32)],
) -> Vec<(u32, u32, f32)> {
    assert!(grid_cols > 0 && grid_rows > 0, "grid must be non-empty");

    if max_per_cell == 0 {
        return Vec::new();
    }
    let candidates = detect_candidates(image, quality_level);
    if candidates.is_empty() {
        return candidates;
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
    let mut cell_count = vec![0u32; (grid_cols * grid_rows) as usize];
    // Spatial hash for min_distance, independent of the detection grid.
    let mut occupancy = OccupancyGrid::new(width, height, min_distance);

    for &(ex, ey) in existing_points {
        cell_count[cell_of(ex, ey)] += 1;
        occupancy.insert(ex, ey);
    }

    let min_distance = min_distance as f32;
    let mut result = Vec::new();

    for &(x, y, q) in &candidates {
        let (xf, yf) = (x as f32, y as f32);
        let cell = cell_of(xf, yf);

        if cell_count[cell] >= max_per_cell {
            continue;
        }
        if occupancy.has_neighbor_within(xf, yf, min_distance) {
            continue;
        }

        occupancy.insert(xf, yf);
        cell_count[cell] += 1;
        result.push((x, y, q));
    }

    result
}

/// Runs the Shi-Tomasi pipeline and returns candidate corners sorted by
/// descending quality, before any spacing constraint is applied.
fn detect_candidates(image: &GrayImage, quality_level: f32) -> Vec<(u32, u32, f32)> {
    if image.width() < 3 || image.height() < 3 {
        return Vec::new();
    }
    // Compute gradients
    let (gx, gy) = compute_gradients(image);

    // Gradient products, 3x3 window sums and minimum eigenvalues, row by row
    let scores = min_eigenvalue_scores(&gx, &gy);
    let (width, height) = (image.width() as usize, image.height() as usize);

    // Non-maximum suppression, skipping scores that cannot pass the quality filter
    let floor = quality_floor(&scores, width, height, quality_level);
    let mut features = local_maxima(&scores, width, height, floor);

    // Filter by quality
    filter_by_quality(&mut features, quality_level);

    // Sort by descending quality
    features.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));

    features
}

/// Uniform spatial hash used to enforce `min_distance` between kept points.
///
/// Cells are `min_distance` wide, so any point closer than `min_distance` to a
/// query is guaranteed to live in the query's cell or one of its 8 neighbors.
struct OccupancyGrid {
    cell_size: u32,
    cols: u32,
    rows: u32,
    cells: Vec<Vec<(f32, f32)>>,
}

impl OccupancyGrid {
    fn new(width: u32, height: u32, min_distance: u32) -> Self {
        let cell_size = min_distance.max(1);
        let cols = width.div_ceil(cell_size).max(1);
        let rows = height.div_ceil(cell_size).max(1);
        OccupancyGrid {
            cell_size,
            cols,
            rows,
            cells: vec![Vec::new(); (cols * rows) as usize],
        }
    }

    fn cell(&self, x: f32, y: f32) -> (u32, u32) {
        let col = (x.max(0.0) as u32 / self.cell_size).min(self.cols - 1);
        let row = (y.max(0.0) as u32 / self.cell_size).min(self.rows - 1);
        (col, row)
    }

    fn insert(&mut self, x: f32, y: f32) {
        let (col, row) = self.cell(x, y);
        self.cells[(row * self.cols + col) as usize].push((x, y));
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

                for &(px, py) in &self.cells[(cy as u32 * self.cols + cx as u32) as usize] {
                    let dist_sq = (x - px).powi(2) + (y - py).powi(2);
                    if dist_sq < min_dist_sq {
                        return true;
                    }
                }
            }
        }

        false
    }
}

/// One image row of the structure tensor components `[[xx, xy], [xy, yy]]`:
/// raw gradient products, or their horizontal or full 3x3 window sums.
struct TensorRow {
    xx: Vec<i32>,
    yy: Vec<i32>,
    xy: Vec<i32>,
}

impl TensorRow {
    fn new(width: usize) -> Self {
        TensorRow {
            xx: vec![0; width],
            yy: vec![0; width],
            xy: vec![0; width],
        }
    }

    fn planes(&self) -> [&[i32]; 3] {
        [&self.xx, &self.yy, &self.xy]
    }

    fn planes_mut(&mut self) -> [&mut [i32]; 3] {
        [&mut self.xx, &mut self.yy, &mut self.xy]
    }

    /// Vertical 3-row sum of horizontally summed rows; `None` is outside the image.
    fn sum_rows(&mut self, above: Option<&Self>, center: &Self, below: Option<&Self>) {
        let (above, below) = (above.map(Self::planes), below.map(Self::planes));
        for (k, dst) in self.planes_mut().into_iter().enumerate() {
            let (a, b) = (above.map(|p| p[k]), below.map(|p| p[k]));
            sum_rows_3(a, center.planes()[k], b, dst);
        }
    }
}

/// Raw Scharr products of one row. For u8 images, |Scharr| <= 4080: products
/// <= 16_646_400 and 3x3 sums <= 149_817_600 fit in i32. Deferring the /32
/// normalization avoids discarding sub-grey-level gradients before squaring.
fn gradient_products_row(gx: &[i16], gy: &[i16], products: &mut TensorRow) {
    let rows = products
        .xx
        .iter_mut()
        .zip(&mut products.yy)
        .zip(&mut products.xy)
        .zip(gx.iter().zip(gy));
    for (((xx, yy), xy), (&ix, &iy)) in rows {
        let (ix, iy) = (ix as i32, iy as i32);
        *xx = ix * ix;
        *yy = iy * iy;
        *xy = ix * iy;
    }
}

/// Products of one gradient row, summed horizontally over three pixels.
fn horizontal_tensor_row(gx: &[i16], gy: &[i16], products: &mut TensorRow, sums: &mut TensorRow) {
    gradient_products_row(gx, gy, products);
    sum_row_3(&products.xx, &mut sums.xx);
    sum_row_3(&products.yy, &mut sums.yy);
    sum_row_3(&products.xy, &mut sums.xy);
}

/// Minimum eigenvalue of the mean 3x3 structure tensor at every pixel, as a
/// row-major score plane. Products and window sums are computed one row at a
/// time, keeping only three horizontally summed rows instead of tensor planes.
fn min_eigenvalue_scores(
    gx: &ImageBuffer<Luma<i16>, Vec<i16>>,
    gy: &ImageBuffer<Luma<i16>, Vec<i16>>,
) -> Vec<f32> {
    let (width, height) = gx.dimensions();
    let (w, h) = (width as usize, height as usize);
    let mut scores = vec![0.0f32; w * h];
    if w == 0 || h == 0 {
        return scores;
    }

    let (gx, gy) = (gx.as_raw(), gy.as_raw());
    let row = |y: usize| y * w..(y + 1) * w;
    let mut products = TensorRow::new(w);
    // Horizontal sums of the rows above, at and below y, stored at row % 3.
    let mut ring = [TensorRow::new(w), TensorRow::new(w), TensorRow::new(w)];
    let mut window = TensorRow::new(w);

    horizontal_tensor_row(&gx[row(0)], &gy[row(0)], &mut products, &mut ring[0]);
    for y in 0..h {
        if y + 1 < h {
            let next = &mut ring[(y + 1) % 3];
            horizontal_tensor_row(&gx[row(y + 1)], &gy[row(y + 1)], &mut products, next);
        }
        let above = (y > 0).then(|| &ring[(y + 2) % 3]);
        let below = (y + 1 < h).then(|| &ring[(y + 1) % 3]);
        window.sum_rows(above, &ring[y % 3], below);
        score_row(&window, y as u32, height, &mut scores[row(y)]);
    }

    scores
}

/// Normalized minimum eigenvalues of one row of 3x3 window sums.
fn score_row(window: &TensorRow, y: u32, height: u32, scores: &mut [f32]) {
    let width = scores.len();
    let rows = 1 + u32::from(y > 0) + u32::from(y + 1 < height);
    // Eigenvalues scale linearly with a common tensor scale. Divide once by
    // Scharr gain squared and the actual sample count, including edges.
    let normalization = |cols: u32| (32 * 32 * cols * rows) as f32;
    let interior = normalization(3);
    let sums = window.xx.iter().zip(&window.yy).zip(&window.xy);
    for (score, ((&a, &b), &c)) in scores.iter_mut().zip(sums) {
        *score = scaled_min_eigenvalue(a, b, c, interior);
    }
    // Boundary columns average fewer samples; a single column has none beside it.
    let edge = normalization(if width == 1 { 1 } else { 2 });
    for x in [0, width - 1] {
        scores[x] = scaled_min_eigenvalue(window.xx[x], window.yy[x], window.xy[x], edge);
    }
}

/// Small eigenvalue of [[a, c], [c, b]] for summed gradient products, divided
/// by `normalization`.
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
fn quality_floor(scores: &[f32], width: usize, height: usize, quality_level: f32) -> f32 {
    let (mut interior, mut border) = (0.0f32, 0.0f32);
    for (y, row) in scores.chunks_exact(width).enumerate() {
        if y == 0 || y + 1 == height {
            border = row.iter().fold(border, |max, &score| max.max(score));
        } else {
            let inner = &row[1..width - 1];
            interior = inner.iter().fold(interior, |max, &score| max.max(score));
            border = border.max(row[0]).max(row[width - 1]);
        }
    }
    if interior >= border {
        quality_level * interior
    } else {
        0.0
    }
}

/// Positive interior local maxima at or above `floor`, in row-major order. A
/// pixel is kept when no 8-neighbor has a greater score, so plateaus survive;
/// it equals the maximum of its 3x3 neighborhood, computed separably.
fn local_maxima(scores: &[f32], width: usize, height: usize, floor: f32) -> Vec<(u32, u32, f32)> {
    let mut maxima = Vec::new();
    if width < 3 || height < 3 {
        return maxima;
    }
    let row = |y: usize| &scores[y * width..(y + 1) * width];
    // Horizontal 3-pixel maxima for the interior columns of rows y - 1, y and
    // y + 1, stored at row % 3.
    let mut ring: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0; width - 2]);
    max_row_3(row(0), &mut ring[0]);
    max_row_3(row(1), &mut ring[1]);
    for y in 1..height - 1 {
        max_row_3(row(y + 1), &mut ring[(y + 1) % 3]);
        let (above, center, below) = (&ring[(y + 2) % 3], &ring[y % 3], &ring[(y + 1) % 3]);
        for (i, &score) in row(y)[1..width - 1].iter().enumerate() {
            let neighborhood = above[i].max(center[i]).max(below[i]);
            if score > 0.0 && score >= floor && score == neighborhood {
                maxima.push((i as u32 + 1, y as u32, score));
            }
        }
    }
    maxima
}

/// `dst[i] = max(src[i], src[i + 1], src[i + 2])`, one value per interior column.
fn max_row_3(src: &[f32], dst: &mut [f32]) {
    let windows = src.iter().zip(&src[1..]).zip(&src[2..]);
    for (value, ((&left, &center), &right)) in dst.iter_mut().zip(windows) {
        *value = left.max(center).max(right);
    }
}

fn filter_by_quality(features: &mut Vec<(u32, u32, f32)>, quality_level: f32) {
    let max_quality = features
        .iter()
        .map(|&(_, _, q)| q)
        .fold(0.0f32, |a, b| a.max(b));
    let threshold = quality_level * max_quality;
    // A zero response is not a corner, including when the entire image is flat.
    features.retain(|&(_, _, q)| q > 0.0 && q >= threshold);
}

fn filter_by_distance(
    features: &[(u32, u32, f32)],
    min_distance: u32,
    width: u32,
    height: u32,
) -> Vec<(u32, u32, f32)> {
    if features.is_empty() || min_distance == 0 {
        return features.to_vec();
    }
    let mut occupancy = OccupancyGrid::new(width, height, min_distance);
    let mut result = Vec::new();
    for &(x, y, q) in features {
        let (xf, yf) = (x as f32, y as f32);
        if !occupancy.has_neighbor_within(xf, yf, min_distance as f32) {
            // Multiple accepted points can occupy one cell (diagonally). Keep
            // every one so later candidates cannot hide an earlier neighbor.
            occupancy.insert(xf, yf);
            result.push((x, y, q));
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::box_filter_3x3::box_sum_3x3_in_place;

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
        let mut window = TensorRow::new(width as usize);
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
        let mut products = TensorRow::new(5);
        gradient_products_row(&gx, &gy, &mut products);
        assert_eq!(products.xx, [1, 961, 1024, 16_646_400, 16_646_400]);
        assert_eq!(products.yy, [1, 1, 961, 16_646_400, 16_646_400]);
        assert_eq!(products.xy, [-1, -31, 992, -16_646_400, 16_646_400]);
    }

    #[test]
    fn row_pipeline_matches_full_plane_box_sums_for_small_and_odd_sizes() {
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
                assert_eq!(
                    min_eigenvalue_scores(&gx, &gy),
                    expected,
                    "{width}x{height}"
                );
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
        for (width, height) in [(3, 3), (4, 3), (3, 7), (8, 5), (17, 11)] {
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
                assert_eq!(local_maxima(&scores, width, height, 0.0), expected);
                for floor in [0.5, 1.0, 3.0, 999.0] {
                    let mut above = expected.clone();
                    above.retain(|p| p.2 >= floor);
                    assert_eq!(local_maxima(&scores, width, height, floor), above);
                }
            }
        }
    }

    #[test]
    fn plateau_pixels_are_all_kept() {
        #[rustfmt::skip]
        let scores = [
            0.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 2.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(local_maxima(&scores, 4, 3, 0.0), [(1, 1, 2.0), (2, 1, 2.0)]);
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
        assert_eq!(local_maxima(&scores, 5, 4, 0.0), [(3, 2, 2.0)]);

        let mut interior = scores;
        interior[0] = 1.0;
        assert_eq!(quality_floor(&interior, 5, 4, 0.5), 2.5);
        assert_eq!(local_maxima(&interior, 5, 4, 2.5), [(1, 1, 5.0)]);
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
        assert_eq!(filter_by_distance(&candidates, 10, 30, 30), candidates[..2]);
    }

    #[test]
    fn zero_spacing_keeps_all_candidates_and_large_spacing_does_not_overflow() {
        let candidates = [(1, 1, 3.0), (1, 2, 2.0), (20, 20, 1.0)];
        assert_eq!(filter_by_distance(&candidates, 0, 30, 30), candidates);
        assert_eq!(
            filter_by_distance(&candidates, u32::MAX, 30, 30),
            candidates[..1]
        );
    }
}
