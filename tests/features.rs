//! Independent scalar/eigensolver reference for the corrected 8-bit detector.

use image::{GrayImage, Luma};
use nalgebra::{Matrix2, SymmetricEigen};
use optical_flow_lk::{good_features_to_track, good_features_to_track_grid};

type Feature = (u32, u32, f32);

fn reference_candidates(image: &GrayImage, quality: f32) -> Vec<Feature> {
    let (w, h) = (image.width() as usize, image.height() as usize);
    if w < 3 || h < 3 {
        return Vec::new();
    }
    let mut products = vec![[0i16; 3]; w * h];
    let kx = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]];
    let ky = [[-3, -10, -3], [0, 0, 0], [3, 10, 3]];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let (mut gx, mut gy) = (0i32, 0i32);
            for j in 0..3 {
                for i in 0..3 {
                    let value = image.get_pixel((x + i - 1) as u32, (y + j - 1) as u32)[0] as i32;
                    gx += value * kx[j][i];
                    gy += value * ky[j][i];
                }
            }
            let (gx, gy) = (gx / 32, gy / 32);
            products[y * w + x] = [(gx * gx) as i16, (gy * gy) as i16, (gx * gy) as i16];
        }
    }
    // Separate source/destination buffers and integer rounding after each pass
    // model the existing detector's quantization independently of its filter.
    let mut horizontal = products.clone();
    let mut smooth = products.clone();
    for y in 0..h {
        for x in 0..w {
            let (lo, hi) = (x.saturating_sub(1), (x + 1).min(w - 1));
            for plane in 0..3 {
                let sum: i32 = (lo..=hi).map(|sx| products[y * w + sx][plane] as i32).sum();
                horizontal[y * w + x][plane] = (sum / (hi - lo + 1) as i32) as i16;
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            let (lo, hi) = (y.saturating_sub(1), (y + 1).min(h - 1));
            for plane in 0..3 {
                let sum: i32 = (lo..=hi)
                    .map(|sy| horizontal[sy * w + x][plane] as i32)
                    .sum();
                smooth[y * w + x][plane] = (sum / (hi - lo + 1) as i32) as i16;
            }
        }
    }
    let scores: Vec<f32> = smooth
        .iter()
        .map(|&[a, b, c]| {
            let eigenvalues =
                SymmetricEigen::new(Matrix2::new(a as f64, c as f64, c as f64, b as f64))
                    .eigenvalues;
            eigenvalues[0].min(eigenvalues[1]).max(0.0) as f32
        })
        .collect();
    let mut candidates = Vec::new();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let score = scores[y * w + x];
            let mut maximum = true;
            for sy in y - 1..=y + 1 {
                for sx in x - 1..=x + 1 {
                    if scores[sy * w + sx] > score {
                        maximum = false;
                    }
                }
            }
            if maximum && score > 0.0 {
                candidates.push((x as u32, y as u32, score));
            }
        }
    }
    let maximum = candidates.iter().map(|p| p.2).fold(0.0, f32::max);
    candidates.retain(|p| p.2 >= maximum * quality);
    candidates.sort_by(|a, b| b.2.total_cmp(&a.2));
    candidates
}

fn reference_spacing(candidates: &[Feature], distance: u32) -> Vec<Feature> {
    let mut accepted: Vec<Feature> = Vec::new();
    let distance_sq = (distance as f64).powi(2);
    for &candidate in candidates {
        if accepted.iter().all(|p| {
            (p.0 as f64 - candidate.0 as f64).powi(2) + (p.1 as f64 - candidate.1 as f64).powi(2)
                >= distance_sq
        }) {
            accepted.push(candidate);
        }
    }
    accepted
}

fn texture(width: u32, height: u32) -> GrayImage {
    GrayImage::from_fn(width, height, |x, y| {
        Luma([((x * 31 + y * 17 + (x ^ y) * 13) & 255) as u8])
    })
}

fn assert_features_match(actual: &[Feature], expected: &[Feature]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert_eq!((a.0, a.1), (e.0, e.1));
        assert!(
            (a.2 - e.2).abs() <= 1e-4 * e.2.max(1.0),
            "score mismatch: {a:?} vs {e:?}"
        );
    }
}

#[test]
fn detector_matches_independent_scalar_reference() {
    for (width, height) in [(3, 3), (4, 5), (17, 19), (48, 32), (65, 49)] {
        let image = texture(width, height);
        for quality in [0.0, 0.01, 0.5, 1.0] {
            let candidates = reference_candidates(&image, quality);
            for distance in [0, 1, 5, 100, u32::MAX] {
                let expected = reference_spacing(&candidates, distance);
                let actual = good_features_to_track(&image, quality, distance);
                assert_features_match(&actual, &expected);
            }
        }
    }
}

#[test]
fn flat_and_tiny_images_do_not_produce_corners_or_panic() {
    for (width, height) in [
        (0, 0),
        (0, 5),
        (5, 0),
        (1, 1),
        (1, 5),
        (5, 1),
        (2, 9),
        (9, 2),
        (3, 3),
        (16, 16),
    ] {
        for value in [0, 127, 255] {
            let image = GrayImage::from_pixel(width, height, Luma([value]));
            assert!(good_features_to_track(&image, 0.01, 0).is_empty());
            assert!(good_features_to_track_grid(&image, 2, 2, 3, 0.01, 0, &[]).is_empty());
        }
    }
}

#[test]
fn grid_detector_matches_reference_with_existing_points() {
    let image = texture(65, 49);
    let existing = [(4.5, 4.5), (18.0, 15.0), (48.0, 34.0)];
    let (cols, rows, budget, distance) = (4, 3, 3, 7u32);
    let cell = |x: f32, y: f32| {
        let col = ((x / 65.0) * cols as f32) as u32;
        let row = ((y / 49.0) * rows as f32) as u32;
        (row.min(rows - 1) * cols + col.min(cols - 1)) as usize
    };
    let mut counts = vec![0; (cols * rows) as usize];
    let mut occupied = existing.to_vec();
    for &(x, y) in &existing {
        counts[cell(x, y)] += 1;
    }
    let mut expected = Vec::new();
    for candidate in reference_candidates(&image, 0.01) {
        let (x, y) = (candidate.0 as f32, candidate.1 as f32);
        let index = cell(x, y);
        if counts[index] < budget
            && occupied
                .iter()
                .all(|&(px, py)| (x - px).powi(2) + (y - py).powi(2) >= (distance as f32).powi(2))
        {
            counts[index] += 1;
            occupied.push((x, y));
            expected.push(candidate);
        }
    }
    let actual = good_features_to_track_grid(&image, cols, rows, budget, 0.01, distance, &existing);
    assert_features_match(&actual, &expected);
    assert_eq!(
        actual,
        good_features_to_track_grid(&image, cols, rows, budget, 0.01, distance, &existing)
    );
}

#[test]
fn step_edge_has_no_corner_response_away_from_its_endpoints() {
    let image = GrayImage::from_fn(31, 27, |x, _| Luma([if x < 15 { 0 } else { 255 }]));
    let features = good_features_to_track(&image, 0.01, 1);
    assert!(
        features
            .iter()
            .all(|&(_, y, _)| y == 1 || y == image.height() - 2)
    );
}

#[test]
fn checkerboard_corner_is_detected_and_unchanged_by_an_intensity_offset() {
    let image = GrayImage::from_fn(31, 31, |x, y| {
        Luma([if (x < 15) == (y < 15) { 96 } else { 32 }])
    });
    let offset_image = GrayImage::from_fn(31, 31, |x, y| Luma([image.get_pixel(x, y)[0] + 64]));
    let features = good_features_to_track(&image, 0.1, 0);
    assert!(
        features
            .iter()
            .any(|&(x, y, _)| (13..=16).contains(&x) && (13..=16).contains(&y))
    );
    assert!(features.iter().all(|p| p.2.is_finite() && p.2 > 0.0));
    assert_eq!(features, good_features_to_track(&offset_image, 0.1, 0));
}
