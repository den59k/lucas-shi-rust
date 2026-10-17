//! f32 detector: equality with the 8-bit detector on integer images, physical
//! units, invariances, thresholds, ordering and degenerate inputs.

use image::{GrayImage, Luma};
use optical_flow_lk::{
    FloatDetectorContext, FloatImageView, ScoreThreshold, good_features_to_track,
    good_features_to_track_grid,
};

#[path = "../benches/support/feature_inputs.rs"]
#[allow(dead_code)]
mod feature_inputs;

type Corner = (u32, u32, f32);

fn as_f32(image: &GrayImage) -> Vec<f32> {
    image.as_raw().iter().map(|&v| v as f32).collect()
}

fn view(pixels: &[f32], width: u32, height: u32) -> FloatImageView<'_> {
    FloatImageView::new(pixels, width, height).expect("valid test image")
}

fn detect(
    pixels: &[f32],
    width: u32,
    height: u32,
    threshold: ScoreThreshold,
    d: u32,
) -> Vec<Corner> {
    FloatDetectorContext::new()
        .detect(view(pixels, width, height), threshold, d)
        .to_vec()
}

fn relative(quality: f32) -> ScoreThreshold {
    ScoreThreshold::absolute(0.0).with_relative(quality)
}

fn noise(w: u32, h: u32, seed: u32) -> GrayImage {
    GrayImage::from_fn(w, h, |x, y| {
        let mut s = (y * w + x)
            .wrapping_mul(2654435761)
            .wrapping_add(seed.wrapping_mul(40503))
            ^ 0x9e3779b9;
        s ^= s >> 15;
        s = s.wrapping_mul(0x85ebca6b);
        Luma([(s & 0xff) as u8])
    })
}

fn integer_images() -> Vec<(String, GrayImage)> {
    let mut images: Vec<(String, GrayImage)> = feature_inputs::cases()
        .into_iter()
        .map(|case| (case.name.to_string(), case.image))
        .collect();
    for (w, h) in [(3, 3), (4, 5), (17, 19), (33, 1), (65, 49), (127, 61)] {
        images.push((format!("noise_{w}x{h}"), noise(w, h, w * 31 + h)));
    }
    for contrast in [1u8, 7, 255] {
        let checker = GrayImage::from_fn(31, 31, |x, y| {
            Luma([if (x < 15) == (y < 15) { contrast } else { 0 }])
        });
        images.push((format!("checker_{contrast}"), checker));
    }
    images.push((
        "binary_noise".into(),
        GrayImage::from_fn(64, 48, |x, y| {
            Luma([if ((x * 7 + y * 13) ^ (x * y)) % 3 == 0 {
                255
            } else {
                0
            }])
        }),
    ));
    images
}

#[test]
fn integer_images_give_the_8_bit_result_bit_for_bit() {
    let mut context = FloatDetectorContext::new();
    for (name, image) in integer_images() {
        let (w, h) = image.dimensions();
        let pixels = as_f32(&image);
        for quality in [0.0f32, 0.01, 0.05, 0.3, 1.0] {
            for distance in [0u32, 1, 8] {
                let expected = good_features_to_track(&image, quality, distance);
                let actual = context.detect(view(&pixels, w, h), relative(quality), distance);
                assert_eq!(actual, expected, "{name} q={quality} d={distance}");
            }
        }
        let existing: Vec<(f32, f32)> = (0..6)
            .map(|i| ((i * 37 % w) as f32 + 0.5, (i * 23 % h) as f32))
            .collect();
        let expected = good_features_to_track_grid(&image, 6, 4, 5, 0.01, 6, &existing);
        let actual =
            context.detect_grid(view(&pixels, w, h), 6, 4, 5, relative(0.01), 6, &existing);
        assert_eq!(actual, expected, "{name} grid");
    }
}

#[test]
fn constant_offset_changes_nothing() {
    let image = feature_inputs::texture(160, 120);
    let base = as_f32(&image);
    // Integer values plus an integer offset stay exact.
    let shifted: Vec<f32> = base.iter().map(|&v| v + 1000.0).collect();
    for threshold in [relative(0.05), ScoreThreshold::absolute(0.5)] {
        assert_eq!(
            detect(&shifted, 160, 120, threshold, 6),
            detect(&base, 160, 120, threshold, 6)
        );
    }
}

#[test]
fn scores_scale_with_squared_contrast() {
    let image = feature_inputs::texture(160, 120);
    let base = as_f32(&image);
    let scaled: Vec<f32> = base.iter().map(|&v| v * 4.0).collect();
    let low: Vec<f32> = base.iter().map(|&v| v / 64.0).collect();
    let reference = detect(&base, 160, 120, relative(0.05), 6);
    assert!(reference.len() > 20);
    for (pixels, factor) in [(&scaled, 16.0f32), (&low, 1.0 / 4096.0)] {
        let corners = detect(pixels, 160, 120, relative(0.05), 6);
        let expected: Vec<Corner> = reference
            .iter()
            .map(|&(x, y, s)| (x, y, s * factor))
            .collect();
        assert_eq!(corners, expected, "factor {factor}");
        // An absolute threshold scales the same way.
        let absolute =
            |pixels: &[f32], t: f32| detect(pixels, 160, 120, ScoreThreshold::absolute(t), 6);
        assert_eq!(absolute(pixels, 2.0 * factor), {
            absolute(&base, 2.0)
                .into_iter()
                .map(|(x, y, s)| (x, y, s * factor))
                .collect::<Vec<_>>()
        });
    }
}

/// A 64x48 frame in °C: 24.5 °C background with rectangles of `contrast` kelvin.
fn thermal_scene(contrast: f32) -> Vec<f32> {
    let (w, h) = (64u32, 48u32);
    (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let inside = |x0, y0, x1, y1| x >= x0 && x < x1 && y >= y0 && y < y1;
            let hot = inside(8, 8, 24, 20) || inside(36, 26, 56, 40);
            24.5 + if hot { contrast } else { 0.0 }
        })
        .collect()
}

/// The 8-bit mapping of a 20-220 °C range (0.78 K per grey level).
fn to_u8(celsius: &[f32], w: u32, h: u32) -> GrayImage {
    let pixels = celsius
        .iter()
        .map(|&t| ((t - 20.0) / 200.0 * 255.0).round().clamp(0.0, 255.0) as u8)
        .collect();
    GrayImage::from_raw(w, h, pixels).unwrap()
}

#[test]
fn weak_thermal_contrast_is_found_in_physical_units() {
    // 0.25 K is a third of a grey level: both temperatures map to level 6.
    let scene = thermal_scene(0.25);
    let gray = to_u8(&scene, 64, 48);
    assert!(good_features_to_track(&gray, 0.05, 4).is_empty());

    // The corner score of a step of height c is proportional to c^2; with an
    // absolute threshold in (°C / px)^2 the eight rectangle corners pass.
    let corners = detect(&scene, 64, 48, ScoreThreshold::absolute(1e-3), 4);
    let rectangle_corners = [
        (8, 8),
        (23, 8),
        (8, 19),
        (23, 19),
        (36, 26),
        (55, 26),
        (36, 39),
        (55, 39),
    ];
    for (cx, cy) in rectangle_corners {
        assert!(
            corners
                .iter()
                .any(|&(x, y, _)| x.abs_diff(cx) <= 1 && y.abs_diff(cy) <= 1),
            "no corner near ({cx}, {cy}) in {corners:?}"
        );
    }
    assert_eq!(corners.len(), rectangle_corners.len(), "{corners:?}");

    // Twice the contrast gives four times the score.
    let doubled = detect(
        &thermal_scene(0.5),
        64,
        48,
        ScoreThreshold::absolute(1e-3),
        4,
    );
    let quadrupled: Vec<Corner> = corners.iter().map(|&(x, y, s)| (x, y, 4.0 * s)).collect();
    assert_eq!(doubled, quadrupled);
}

#[test]
fn relative_reference_is_the_strongest_corner_of_the_whole_image() {
    // Strong rectangle on the left, weak one on the right.
    let (w, h) = (64u32, 32u32);
    let pixels: Vec<f32> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let inside = |x0, x1| x >= x0 && x < x1 && (8..24).contains(&y);
            if inside(6, 22) {
                10.0
            } else if inside(40, 56) {
                0.5
            } else {
                0.0
            }
        })
        .collect();
    let mut context = FloatDetectorContext::new();
    let image = view(&pixels, w, h);
    let right_half = |corners: &[Corner]| corners.iter().filter(|c| c.0 >= 32).count();

    let global = context.detect_grid(image, 2, 1, 10, relative(0.05), 3, &[]);
    assert!(global.iter().all(|c| c.0 < 32) && !global.is_empty());

    let absolute = context.detect_grid(image, 2, 1, 10, ScoreThreshold::absolute(1e-3), 3, &[]);
    assert!(right_half(absolute) >= 4, "{absolute:?}");
}

#[test]
fn results_are_ordered_and_deterministic_across_reuse() {
    let a = as_f32(&feature_inputs::texture(200, 150));
    let b = as_f32(&noise(120, 90, 3));
    let threshold = ScoreThreshold::absolute(0.2).with_relative(0.01);
    let fresh_a = detect(&a, 200, 150, threshold, 5);
    let fresh_b = detect(&b, 120, 90, threshold, 5);
    let mut context = FloatDetectorContext::new();
    for _ in 0..3 {
        assert_eq!(context.detect(view(&a, 200, 150), threshold, 5), fresh_a);
        assert_eq!(context.detect(view(&b, 120, 90), threshold, 5), fresh_b);
    }
    for corners in [&fresh_a, &fresh_b] {
        assert!(corners.windows(2).all(|pair| {
            let ((x0, y0, s0), (x1, y1, s1)) = (pair[0], pair[1]);
            s0 > s1 || (s0 == s1 && (y0, x0) < (y1, x1))
        }));
    }
}

#[test]
fn plateau_ties_follow_row_major_order() {
    // Two identical, separated squares: mirrored corners score exactly equal.
    let (w, h) = (40u32, 20u32);
    let pixels: Vec<f32> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let square = |x0: u32| x >= x0 && x < x0 + 8 && (6..14).contains(&y);
            if square(6) || square(26) { 2.0 } else { 0.0 }
        })
        .collect();
    let corners = detect(&pixels, w, h, ScoreThreshold::absolute(1e-3), 0);
    let strongest = corners[0].2;
    let tied: Vec<(u32, u32)> = corners
        .iter()
        .filter(|c| c.2 == strongest)
        .map(|c| (c.0, c.1))
        .collect();
    assert!(tied.len() >= 2, "{corners:?}");
    let mut row_major = tied.clone();
    row_major.sort_by_key(|&(x, y)| (y, x));
    assert_eq!(tied, row_major);
}

#[test]
fn degenerate_images_yield_no_corners() {
    let threshold = ScoreThreshold::absolute(0.0);
    for (w, h) in [(0, 0), (0, 5), (1, 1), (2, 9), (9, 2)] {
        let pixels = vec![1.5f32; (w * h) as usize];
        assert!(detect(&pixels, w, h, threshold, 3).is_empty(), "{w}x{h}");
    }
    let flat = vec![300.0f32; 50 * 40];
    assert!(detect(&flat, 50, 40, threshold, 0).is_empty());
    let mut context = FloatDetectorContext::new();
    assert!(
        context
            .detect_grid(view(&flat, 50, 40), 3, 3, 5, threshold, 2, &[(10.0, 10.0)])
            .is_empty()
    );
}

#[test]
fn grid_budget_counts_existing_points_and_keeps_their_distance() {
    let pixels = as_f32(&feature_inputs::texture(160, 120));
    let image = view(&pixels, 160, 120);
    let threshold = ScoreThreshold::absolute(0.01);
    let mut context = FloatDetectorContext::new();
    assert!(
        context
            .detect_grid(image, 4, 3, 0, threshold, 5, &[])
            .is_empty()
    );

    let existing = [(20.0, 20.0), (21.0, 22.0), (100.0, 60.0)];
    let corners = context
        .detect_grid(image, 4, 3, 3, threshold, 8, &existing)
        .to_vec();
    let cell = |x: f32, y: f32| {
        ((y / 120.0 * 3.0) as usize).min(2) * 4 + ((x / 160.0 * 4.0) as usize).min(3)
    };
    let mut counts = [0usize; 12];
    for &(x, y) in &existing {
        counts[cell(x, y)] += 1;
    }
    for &(x, y, _) in &corners {
        counts[cell(x as f32, y as f32)] += 1;
        for &(ex, ey) in &existing {
            let distance = ((x as f32 - ex).powi(2) + (y as f32 - ey).powi(2)).sqrt();
            assert!(distance >= 8.0, "({x}, {y}) too close to ({ex}, {ey})");
        }
    }
    assert!(counts.iter().all(|&count| count <= 3), "{counts:?}");
    assert!(corners.len() >= 9, "{corners:?}");
}

#[test]
#[should_panic(expected = "grid must be non-empty")]
fn empty_grid_panics() {
    let pixels = vec![0.0f32; 16];
    FloatDetectorContext::new().detect_grid(
        view(&pixels, 4, 4),
        0,
        2,
        1,
        ScoreThreshold::absolute(0.0),
        0,
        &[],
    );
}
