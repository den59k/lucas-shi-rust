use image::{GrayImage, ImageBuffer, Luma};
use std::cmp::Ordering;

use crate::utils::{box_filter_3x3::box_filter_3x3_in_place, fast_gradients::compute_gradients};

/// Finds good features points using the Shi-Tomasi algorithm
///
/// # Arguments
/// * `image` - Target image (grayscale)
/// * `quality_level` - Quality level. 0.4 is a good value
/// * `min_distance` - Filter points by distance between
///
///
/// # Returns
/// Vector of features with eigenvalue. Points sorted in descending order of quality
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

    let (width, height) = image.dimensions();
    let candidates = detect_candidates(image, quality_level);

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
    // Compute gradients
    let (gx, gy) = compute_gradients(image);

    // Compute squared gradients and their product
    let (mut ix_sq, mut iy_sq, mut ix_iy) = compute_gradient_products(&gx, &gy);

    // Smooth with 3x3 filters
    box_filter_3x3_in_place(&mut ix_sq);
    box_filter_3x3_in_place(&mut iy_sq);
    box_filter_3x3_in_place(&mut ix_iy);

    // Compute minimum eigenvalues
    let mut features = compute_min_eigenvalues(&ix_sq, &iy_sq, &ix_iy);

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
    ImageBuffer<Luma<i16>, Vec<i16>>,
    ImageBuffer<Luma<i16>, Vec<i16>>,
    ImageBuffer<Luma<i16>, Vec<i16>>,
);

fn compute_gradient_products(
    gx: &ImageBuffer<Luma<i16>, Vec<i16>>,
    gy: &ImageBuffer<Luma<i16>, Vec<i16>>,
) -> GradientProduct {
    let mut ix_sq: ImageBuffer<Luma<i16>, Vec<i16>> = ImageBuffer::new(gx.width(), gx.height());
    let mut iy_sq: ImageBuffer<Luma<i16>, Vec<i16>> = ImageBuffer::new(gx.width(), gx.height());
    let mut ix_iy: ImageBuffer<Luma<i16>, Vec<i16>> = ImageBuffer::new(gx.width(), gx.height());

    for ((x, y, gx_val), gy_val) in gx.enumerate_pixels().zip(gy.pixels()) {
        let ix = gx_val[0];
        let iy = gy_val[0];

        ix_sq.put_pixel(x, y, Luma([(ix / 32 * (ix / 32))]));
        iy_sq.put_pixel(x, y, Luma([(iy / 32 * (iy / 32))]));
        ix_iy.put_pixel(x, y, Luma([(ix / 32 * (iy / 32))]));
    }

    (ix_sq, iy_sq, ix_iy)
}

fn compute_min_eigenvalues(
    a: &ImageBuffer<Luma<i16>, Vec<i16>>,
    b: &ImageBuffer<Luma<i16>, Vec<i16>>,
    c: &ImageBuffer<Luma<i16>, Vec<i16>>,
) -> Vec<(u32, u32, f32)> {
    let mut features = Vec::with_capacity((a.width() * a.height()) as usize);

    for y in 0..a.height() {
        for x in 0..a.width() {
            let a_val = a.get_pixel(x, y)[0] as i32;
            let b_val = b.get_pixel(x, y)[0] as i32;
            let c_val = c.get_pixel(x, y)[0] as i32;

            let trace = a_val + b_val;
            let discriminant = (a_val - b_val).pow(2) + 4 * c_val.pow(2);
            let min_eigen = (((trace - discriminant) as f32).sqrt()) / 2.0;

            features.push((x, y, min_eigen));
        }
    }

    features
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
    features.retain(|&(_, _, q)| q >= threshold);
}

fn filter_by_distance(
    features: &[(u32, u32, f32)],
    min_distance: u32,
    width: u32,
    height: u32,
) -> Vec<(u32, u32, f32)> {
    let cell_size = min_distance;
    let grid_width = width.div_ceil(cell_size);
    let grid_height = height.div_ceil(cell_size);
    let mut grid = vec![vec![None; grid_height as usize]; grid_width as usize];
    let mut result = Vec::new();

    let min_dist_sq = (min_distance * min_distance) as i32;

    for &(x, y, q) in features {
        let cell_x = x / cell_size;
        let cell_y = y / cell_size;
        let mut too_close = false;

        for dx in -1..=1 {
            for dy in -1..=1 {
                let check_x = cell_x as i32 + dx;
                let check_y = cell_y as i32 + dy;

                if check_x < 0
                    || check_y < 0
                    || check_x >= grid_width as i32
                    || check_y >= grid_height as i32
                {
                    continue;
                }

                if let Some((px, py)) = grid[check_x as usize][check_y as usize] {
                    let dist_sq: i32 =
                        (x as i32 - px as i32).pow(2) + (y as i32 - py as i32).pow(2);
                    if dist_sq < min_dist_sq {
                        too_close = true;
                        break;
                    }
                }
            }
            if too_close {
                break;
            }
        }

        if !too_close {
            grid[cell_x as usize][cell_y as usize] = Some((x, y));
            result.push((x, y, q));
        }
    }

    result
}
