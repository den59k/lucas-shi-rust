// Main findings from this benchmark:
// - This benchmark isolates the Lucas-Kanade tracking loop itself, not the full pipeline.
// - Both variants use the same image pyramids, input points, and precomputed gradient pyramids,
//   so the timing difference focuses on the LK loop changes rather than gradient generation.
// - `calc_optical_flow_old_loop` models the historical implementation:
//   dynamic `Vec` allocation, dynamic `nalgebra` matrices, and per-iteration SVD solves.
// - `calc_optical_flow_new_loop` models the optimized implementation:
//   direct 2x2 accumulation/solve plus per-point precomputation of the previous patch and
//   gradient patch outside the inner iteration loop.
// - On the Windows dev machine, the optimized loop measured about 3.05x faster than the old loop
//   on the example images with 100 tracked points, a 21x21 window, and 30 max iterations.
//
// Remaining hotspot:
// - Bilinear interpolation is still done with checked pixel access in both variants, so it is
//   likely the next meaningful optimization target inside the LK path.

use std::hint::black_box;
use std::time::{Duration, Instant};

use image::{GrayImage, ImageBuffer, Luma, open};
use imageproc::gradients::{horizontal_scharr, vertical_scharr};
use nalgebra::{DMatrix, DVector, SVD};
use optical_flow_lk::{build_pyramid, good_features_to_track};

type GrayI16Image = ImageBuffer<Luma<i16>, Vec<i16>>;
type GradientPyramid = Vec<(GrayI16Image, GrayI16Image)>;
type LkCase = (
    Vec<GrayImage>,
    Vec<GrayImage>,
    GradientPyramid,
    Vec<(f32, f32)>,
);

fn calc_optical_flow_old_loop(
    prev_pyramid: &[GrayImage],
    curr_pyramid: &[GrayImage],
    grad_pyramid: &GradientPyramid,
    prev_points: &[(f32, f32)],
    window_size: usize,
    max_iterations: usize,
) -> Vec<(f32, f32)> {
    assert_eq!(prev_pyramid.len(), curr_pyramid.len());
    assert_eq!(prev_pyramid.len(), grad_pyramid.len());
    assert!(window_size % 2 == 1, "Window size must be odd");

    let n_levels = prev_pyramid.len();
    let radius = window_size / 2;
    let epsilon = 1e-3;
    let mut displacements: Vec<(f32, f32)> = prev_points.iter().map(|_| (0.0, 0.0)).collect();

    for level in (0..n_levels).rev() {
        let scale = 2f32.powi(level as i32);
        let prev_img = &prev_pyramid[level];
        let curr_img = &curr_pyramid[level];
        let (grad_x, grad_y) = (&grad_pyramid[level].0, &grad_pyramid[level].1);

        for ((prev_x, prev_y), disp) in prev_points.iter().zip(displacements.iter_mut()) {
            let x = *prev_x / scale;
            let y = *prev_y / scale;
            let mut dx = disp.0 / scale;
            let mut dy = disp.1 / scale;

            if !in_bounds(prev_img, x, y, radius) {
                continue;
            }

            let mut converged = false;
            for _ in 0..max_iterations {
                if converged {
                    break;
                }

                let curr_x = x + dx;
                let curr_y = y + dy;

                if !in_bounds(curr_img, curr_x, curr_y, radius) {
                    break;
                }

                let mut a_data = Vec::with_capacity(window_size * window_size * 2);
                let mut b_data = Vec::with_capacity(window_size * window_size);

                for j in -(radius as i32)..=radius as i32 {
                    for i in -(radius as i32)..=radius as i32 {
                        let px_prev = interpolate(prev_img, x + i as f32, y + j as f32);
                        let px_curr = interpolate(curr_img, curr_x + i as f32, curr_y + j as f32);
                        let ix = interpolate_alt(grad_x, x + i as f32, y + j as f32) / 32.0;
                        let iy = interpolate_alt(grad_y, x + i as f32, y + j as f32) / 32.0;

                        a_data.push(ix);
                        a_data.push(iy);
                        b_data.push(px_prev - px_curr);
                    }
                }

                let n_pixels = window_size * window_size;
                let a_matrix = DMatrix::from_row_slice(n_pixels, 2, &a_data);
                let b_vector = DVector::from_vec(b_data);
                let ata = a_matrix.transpose() * &a_matrix;
                let atb = a_matrix.transpose() * &b_vector;

                let svd = SVD::new(ata, true, true);
                if let Ok(solution) = svd.solve(&atb, 1e-6) {
                    let (ddx, ddy) = (solution[0], solution[1]);
                    dx += ddx;
                    dy += ddy;

                    if ddx.abs() < epsilon && ddy.abs() < epsilon {
                        converged = true;
                    }
                } else {
                    break;
                }
            }

            *disp = (dx * scale, dy * scale);
        }
    }

    prev_points
        .iter()
        .zip(displacements.iter())
        .map(|((x, y), (dx, dy))| (x + dx, y + dy))
        .collect()
}

fn calc_optical_flow_new_loop(
    prev_pyramid: &[GrayImage],
    curr_pyramid: &[GrayImage],
    grad_pyramid: &GradientPyramid,
    prev_points: &[(f32, f32)],
    window_size: usize,
    max_iterations: usize,
) -> Vec<(f32, f32)> {
    assert_eq!(prev_pyramid.len(), curr_pyramid.len());
    assert_eq!(prev_pyramid.len(), grad_pyramid.len());
    assert!(window_size % 2 == 1, "Window size must be odd");

    let n_levels = prev_pyramid.len();
    let radius = window_size / 2;
    let n_pixels = window_size * window_size;
    let epsilon = 1e-3;
    let det_epsilon = 1e-6;
    let offsets = build_window_offsets(radius);
    let mut displacements: Vec<(f32, f32)> = prev_points.iter().map(|_| (0.0, 0.0)).collect();

    for level in (0..n_levels).rev() {
        let scale = 2f32.powi(level as i32);
        let prev_img = &prev_pyramid[level];
        let curr_img = &curr_pyramid[level];
        let (grad_x, grad_y) = (&grad_pyramid[level].0, &grad_pyramid[level].1);
        let mut prev_patch = vec![0.0f32; n_pixels];
        let mut ix_patch = vec![0.0f32; n_pixels];
        let mut iy_patch = vec![0.0f32; n_pixels];

        for ((prev_x, prev_y), disp) in prev_points.iter().zip(displacements.iter_mut()) {
            let x = *prev_x / scale;
            let y = *prev_y / scale;
            let mut dx = disp.0 / scale;
            let mut dy = disp.1 / scale;

            if !in_bounds(prev_img, x, y, radius) {
                continue;
            }

            let mut gxx = 0.0f32;
            let mut gxy = 0.0f32;
            let mut gyy = 0.0f32;

            for (idx, (ox, oy)) in offsets.iter().enumerate() {
                let sample_x = x + ox;
                let sample_y = y + oy;
                let ix = interpolate_alt(grad_x, sample_x, sample_y) / 32.0;
                let iy = interpolate_alt(grad_y, sample_x, sample_y) / 32.0;

                prev_patch[idx] = interpolate(prev_img, sample_x, sample_y);
                ix_patch[idx] = ix;
                iy_patch[idx] = iy;
                gxx += ix * ix;
                gxy += ix * iy;
                gyy += iy * iy;
            }

            let Some((inv_h00, inv_h01, inv_h11)) = invert_2x2(gxx, gxy, gyy, det_epsilon) else {
                continue;
            };

            for _ in 0..max_iterations {
                let curr_x = x + dx;
                let curr_y = y + dy;

                if !in_bounds(curr_img, curr_x, curr_y, radius) {
                    break;
                }

                let mut bx = 0.0f32;
                let mut by = 0.0f32;

                for (idx, (ox, oy)) in offsets.iter().enumerate() {
                    let curr = interpolate(curr_img, curr_x + ox, curr_y + oy);
                    let error = prev_patch[idx] - curr;
                    bx += ix_patch[idx] * error;
                    by += iy_patch[idx] * error;
                }

                let ddx = inv_h00 * bx + inv_h01 * by;
                let ddy = inv_h01 * bx + inv_h11 * by;
                dx += ddx;
                dy += ddy;

                if ddx.abs() < epsilon && ddy.abs() < epsilon {
                    break;
                }
            }

            *disp = (dx * scale, dy * scale);
        }
    }

    prev_points
        .iter()
        .zip(displacements.iter())
        .map(|((x, y), (dx, dy))| (x + dx, y + dy))
        .collect()
}

fn build_gradient_pyramid(prev_pyramid: &[GrayImage]) -> GradientPyramid {
    prev_pyramid
        .iter()
        .map(|img| (horizontal_scharr(img), vertical_scharr(img)))
        .collect()
}

fn build_window_offsets(radius: usize) -> Vec<(f32, f32)> {
    let mut offsets = Vec::with_capacity((2 * radius + 1) * (2 * radius + 1));

    for j in -(radius as i32)..=radius as i32 {
        for i in -(radius as i32)..=radius as i32 {
            offsets.push((i as f32, j as f32));
        }
    }

    offsets
}

fn invert_2x2(a00: f32, a01: f32, a11: f32, det_epsilon: f32) -> Option<(f32, f32, f32)> {
    let det = a00 * a11 - a01 * a01;
    if det.abs() <= det_epsilon {
        return None;
    }

    let inv_det = 1.0 / det;
    Some((a11 * inv_det, -a01 * inv_det, a00 * inv_det))
}

fn in_bounds(img: &GrayImage, x: f32, y: f32, radius: usize) -> bool {
    let (w, h) = (img.width() as f32, img.height() as f32);
    x >= radius as f32 && x < w - radius as f32 && y >= radius as f32 && y < h - radius as f32
}

fn interpolate(img: &GrayImage, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    let mut sum = 0.0;
    for (sx, sy) in &[(x0, y0), (x0, y1), (x1, y0), (x1, y1)] {
        let px = img
            .get_pixel_checked(*sx as u32, *sy as u32)
            .map(|p| p[0] as f32)
            .unwrap_or(0.0);
        let wx = if sx == &x0 { 1.0 - dx } else { dx };
        let wy = if sy == &y0 { 1.0 - dy } else { dy };
        sum += px * wx * wy;
    }

    sum
}

fn interpolate_alt(img: &GrayI16Image, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    let mut sum = 0.0;
    for (sx, sy) in &[(x0, y0), (x0, y1), (x1, y0), (x1, y1)] {
        let px = img
            .get_pixel_checked(*sx as u32, *sy as u32)
            .map(|p| p[0] as f32)
            .unwrap_or(0.0);
        let wx = if sx == &x0 { 1.0 - dx } else { dx };
        let wy = if sy == &y0 { 1.0 - dy } else { dy };
        sum += px * wx * wy;
    }

    sum
}

fn load_case() -> LkCase {
    let prev_frame: GrayImage = open("examples/input1.png").unwrap().into_luma8();
    let next_frame: GrayImage = open("examples/input2.png").unwrap().into_luma8();
    let prev_pyramid = build_pyramid(&prev_frame, 4);
    let next_pyramid = build_pyramid(&next_frame, 4);
    let grad_pyramid = build_gradient_pyramid(&prev_pyramid);
    let mut points = good_features_to_track(&prev_frame, 0.1, 5);
    points.truncate(100);
    let prev_points = points
        .into_iter()
        .map(|(x, y, _)| (x as f32, y as f32))
        .collect();

    (prev_pyramid, next_pyramid, grad_pyramid, prev_points)
}

fn assert_points_close(expected: &[(f32, f32)], actual: &[(f32, f32)]) {
    assert_eq!(expected.len(), actual.len());

    for (idx, (expected_pt, actual_pt)) in expected.iter().zip(actual.iter()).enumerate() {
        let dx = (expected_pt.0 - actual_pt.0).abs();
        let dy = (expected_pt.1 - actual_pt.1).abs();
        assert!(
            dx <= 1e-2 && dy <= 1e-2,
            "point {idx} differs too much: expected {:?}, actual {:?}",
            expected_pt,
            actual_pt
        );
    }
}

fn validate_implementations(
    prev_pyramid: &[GrayImage],
    next_pyramid: &[GrayImage],
    grad_pyramid: &GradientPyramid,
    prev_points: &[(f32, f32)],
) {
    let old = calc_optical_flow_old_loop(
        prev_pyramid,
        next_pyramid,
        grad_pyramid,
        prev_points,
        21,
        30,
    );
    let new = calc_optical_flow_new_loop(
        prev_pyramid,
        next_pyramid,
        grad_pyramid,
        prev_points,
        21,
        30,
    );
    assert_points_close(&old, &new);
}

fn time_it<T, F>(iterations: usize, mut f: F) -> Duration
where
    F: FnMut() -> T,
{
    for _ in 0..2 {
        black_box(f());
    }

    let start = Instant::now();
    for _ in 0..iterations {
        black_box(f());
    }
    start.elapsed()
}

fn print_result(label: &str, iterations: usize, elapsed: Duration) {
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let per_iter_ms = total_ms / iterations as f64;
    println!("{label:<28} total: {total_ms:>10.3} ms   per-iter: {per_iter_ms:>8.3} ms");
}

fn main() {
    let (prev_pyramid, next_pyramid, grad_pyramid, prev_points) = load_case();

    println!("Lucas-Kanade loop benchmark: old vs optimized");
    validate_implementations(&prev_pyramid, &next_pyramid, &grad_pyramid, &prev_points);
    println!("Correctness check passed");

    let iterations = 10;
    println!("Points: {}  iterations: {}", prev_points.len(), iterations);

    let old_elapsed = time_it(iterations, || {
        calc_optical_flow_old_loop(
            black_box(&prev_pyramid),
            black_box(&next_pyramid),
            black_box(&grad_pyramid),
            black_box(&prev_points),
            21,
            30,
        )
    });
    print_result("calc_optical_flow_old", iterations, old_elapsed);

    let new_elapsed = time_it(iterations, || {
        calc_optical_flow_new_loop(
            black_box(&prev_pyramid),
            black_box(&next_pyramid),
            black_box(&grad_pyramid),
            black_box(&prev_points),
            21,
            30,
        )
    });
    print_result("calc_optical_flow_new", iterations, new_elapsed);

    let speedup = old_elapsed.as_secs_f64() / new_elapsed.as_secs_f64();
    println!("ratio old/new: {speedup:.3}x");
}
