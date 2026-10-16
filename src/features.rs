use image::{GrayImage, ImageBuffer, Luma};
use std::cmp::Ordering;

use crate::utils::{box_filter_3x3::box_sum_3x3_in_place, fast_gradients::compute_gradients};

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

    // Compute squared gradients and their product
    let (mut ix_sq, mut iy_sq, mut ix_iy) = compute_gradient_products(&gx, &gy);

    // Sum without division; min_eigenvalue_scores applies the mean and
    // Scharr normalization together, preserving fractional gradient responses.
    box_sum_3x3_in_place(&mut ix_sq);
    box_sum_3x3_in_place(&mut iy_sq);
    box_sum_3x3_in_place(&mut ix_iy);

    // Compute minimum eigenvalues
    let mut features = min_eigenvalue_scores(&ix_sq, &iy_sq, &ix_iy);

    // Non-maximum suppression
    non_maximum_suppression(&mut features, image.width(), image.height());

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

type GradientProduct = (
    ImageBuffer<Luma<i32>, Vec<i32>>,
    ImageBuffer<Luma<i32>, Vec<i32>>,
    ImageBuffer<Luma<i32>, Vec<i32>>,
);

fn compute_gradient_products(
    gx: &ImageBuffer<Luma<i16>, Vec<i16>>,
    gy: &ImageBuffer<Luma<i16>, Vec<i16>>,
) -> GradientProduct {
    let (width, height) = gx.dimensions();
    let gx_data = gx.as_raw();
    let gy_data = gy.as_raw();
    let n = gx_data.len();

    // Build the three product planes as flat buffers, avoiding per-pixel
    // bounds-checked `put_pixel` (a noticeable win under WASM).
    let mut ix_sq = vec![0i32; n];
    let mut iy_sq = vec![0i32; n];
    let mut ix_iy = vec![0i32; n];

    // For u8 images, |Scharr| <= 4080: products <= 16_646_400 and 3x3
    // sums <= 149_817_600 fit in i32. Deferring the /32 normalization avoids
    // discarding sub-grey-level gradients before squaring them.
    for i in 0..n {
        let ix = gx_data[i] as i32;
        let iy = gy_data[i] as i32;
        ix_sq[i] = ix * ix;
        iy_sq[i] = iy * iy;
        ix_iy[i] = ix * iy;
    }

    (
        ImageBuffer::from_vec(width, height, ix_sq).unwrap(),
        ImageBuffer::from_vec(width, height, iy_sq).unwrap(),
        ImageBuffer::from_vec(width, height, ix_iy).unwrap(),
    )
}

fn min_eigenvalue_scores(
    a: &ImageBuffer<Luma<i32>, Vec<i32>>,
    b: &ImageBuffer<Luma<i32>, Vec<i32>>,
    c: &ImageBuffer<Luma<i32>, Vec<i32>>,
) -> Vec<(u32, u32, f32)> {
    let (width, height) = a.dimensions();
    let (a_data, b_data, c_data) = (a.as_raw(), b.as_raw(), c.as_raw());
    let mut features = Vec::with_capacity(a_data.len());

    for i in 0..a_data.len() {
        let x = i as u32 % width;
        let y = i as u32 / width;
        let cols = 1 + u32::from(x > 0) + u32::from(x + 1 < width);
        let rows = 1 + u32::from(y > 0) + u32::from(y + 1 < height);
        // Eigenvalues scale linearly with a common tensor scale. Divide once
        // by Scharr gain squared and the actual sample count, including edges.
        let normalization = (32 * 32 * cols * rows) as f64;
        let min_eigen =
            (tensor_min_eigenvalue(a_data[i], b_data[i], c_data[i]) / normalization) as f32;
        features.push((x, y, min_eigen));
    }

    features
}

/// Small eigenvalue of [[a, c], [c, b]], for summed gradient products.
fn tensor_min_eigenvalue(a: i32, b: i32, c: i32) -> f64 {
    // i32 tensor sums can have a discriminant larger than i32. Compute the
    // determinant exactly in i64 before conversion to f64; det/lambda_max
    // avoids cancellation in trace - sqrt(discriminant) for nearly rank-one
    // tensors. Valid u8 tensors have |entry| <= 149_817_600, so i64 is ample.
    let determinant = a as i64 * b as i64 - c as i64 * c as i64;
    if determinant <= 0 {
        return 0.0;
    }
    let (a, b, c) = (a as f64, b as f64, c as f64);
    let discriminant_sqrt = ((a - b).powi(2) + 4.0 * c.powi(2)).sqrt();
    let largest = (a + b + discriminant_sqrt) * 0.5;
    determinant as f64 / largest
}

fn non_maximum_suppression(features: &mut Vec<(u32, u32, f32)>, width: u32, height: u32) {
    let mut is_local_max = vec![false; features.len()];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = (y * width + x) as usize;
            let current = features[idx].2;

            let mut is_max = true;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                        continue;
                    }
                    let neighbor_idx = (ny as u32 * width + nx as u32) as usize;
                    if features[neighbor_idx].2 > current {
                        is_max = false;
                        break;
                    }
                }
                if !is_max {
                    break;
                }
            }
            is_local_max[idx] = is_max;
        }
    }

    features.retain(|(x, y, _)| {
        let idx = (y * width + x) as usize;
        is_local_max[idx]
    });
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
            let score = tensor_min_eigenvalue(tensor.0, tensor.1, tensor.2);
            assert!(score.is_finite());
            assert!(
                (score - tensor.3).abs() < 1e-5,
                "wrong score for {tensor:?}"
            );
        }
    }

    #[test]
    fn raw_products_preserve_fractional_normalized_gradients_and_extremes() {
        let gx = ImageBuffer::from_vec(5, 1, vec![1i16, -31, 32, 4080, -4080]).unwrap();
        let gy = ImageBuffer::from_vec(5, 1, vec![-1i16, 1, 31, -4080, -4080]).unwrap();
        let (xx, yy, xy) = compute_gradient_products(&gx, &gy);
        assert_eq!(xx.as_raw(), &[1, 961, 1024, 16_646_400, 16_646_400]);
        assert_eq!(yy.as_raw(), &[1, 1, 961, 16_646_400, 16_646_400]);
        assert_eq!(xy.as_raw(), &[-1, -31, 992, -16_646_400, 16_646_400]);
    }

    #[test]
    fn normalization_preserves_tensor_mean_at_borders_and_small_dimensions() {
        for (width, height) in [(1, 1), (1, 5), (5, 1), (2, 2), (3, 3), (7, 9)] {
            let mut a = ImageBuffer::from_pixel(width, height, Luma([1024]));
            let mut b = ImageBuffer::from_pixel(width, height, Luma([2048]));
            let c = ImageBuffer::new(width, height);
            box_sum_3x3_in_place(&mut a);
            box_sum_3x3_in_place(&mut b);
            let scores = min_eigenvalue_scores(&a, &b, &c);
            assert_eq!(scores.len(), (width * height) as usize);
            assert!(scores.iter().all(|p| p.2 == 1.0));
        }
    }

    #[test]
    fn fractional_window_mean_is_not_truncated() {
        let a = ImageBuffer::from_pixel(3, 3, Luma([1]));
        let b = a.clone();
        let c = ImageBuffer::new(3, 3);
        let scores = min_eigenvalue_scores(&a, &b, &c);
        let expected = 1.0 / (32.0 * 32.0 * 9.0);
        assert_eq!(scores[4], (1, 1, expected));
    }

    #[test]
    fn nearly_rank_one_tensor_keeps_its_small_positive_eigenvalue() {
        // Outer products of (4080, 4079) and (4079, 4078) have determinant 1.
        let (a, b, c) = (33_284_641, 33_268_325, 33_276_482);
        assert_eq!(a as i64 * b as i64 - c as i64 * c as i64, 1);
        let score = tensor_min_eigenvalue(a, b, c);
        let expected = 1.0 / (a as f64 + b as f64);
        assert!((score - expected).abs() <= expected * 1e-14);
    }

    #[test]
    fn maximum_product_sum_is_finite_and_rank_one_stays_zero() {
        let max = 9 * 4080 * 4080;
        assert_eq!(tensor_min_eigenvalue(max, max, 0), max as f64);
        assert_eq!(tensor_min_eigenvalue(max, max, max), 0.0);
        assert_eq!(tensor_min_eigenvalue(max, max, -max), 0.0);
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
