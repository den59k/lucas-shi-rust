//! f32 Lucas-Kanade: accuracy, agreement with the 8-bit tracker, physical
//! units, status codes and invalid inputs.

use image::GrayImage;
use optical_flow_lk::{
    DEFAULT_FB_THRESHOLD, DEFAULT_MIN_EIGEN_THRESHOLD, FloatImageView, FloatTrackerContext,
    TrackKernel, TrackOptions, TrackParams, TrackResult, TrackStatus, TrackerContext,
};

const WIN: usize = 21;
const ITERS: usize = 30;

/// Band-limited f32 texture with values in about 0..255: hash noise smoothed by
/// three 5x5 means, without quantization.
fn texture(w: u32, h: u32, seed: u32) -> Vec<f32> {
    let mut buf: Vec<f32> = (0..w * h)
        .map(|i| {
            let mut s = i.wrapping_mul(2654435761).wrapping_add(seed) ^ 0x9e3779b9;
            s ^= s >> 15;
            s = s.wrapping_mul(0x85ebca6b);
            s ^= s >> 13;
            (s & 0xff) as f32
        })
        .collect();
    for _ in 0..3 {
        let src = buf.clone();
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let (mut sum, mut n) = (0.0, 0.0);
                for dy in -2..=2 {
                    for dx in -2..=2 {
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
    // Stretch the smoothed noise back to roughly the full 8-bit range.
    let (min, max) = buf
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    buf.iter()
        .map(|&v| (v - min) / (max - min) * 255.0)
        .collect()
}

/// Bilinear sample with edge clamping.
fn sample(src: &[f32], w: u32, h: u32, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (x.floor() as u32, y.floor() as u32);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f32, y - y0 as f32);
    let p = |x: u32, y: u32| src[(y * w + x) as usize];
    p(x0, y0) * (1.0 - fx) * (1.0 - fy)
        + p(x1, y0) * fx * (1.0 - fy)
        + p(x0, y1) * (1.0 - fx) * fy
        + p(x1, y1) * fx * fy
}

/// `src` warped so that content moves by `(sx, sy)` and rotates by `angle`
/// radians around `(cx, cy)`.
fn warp(src: &[f32], w: u32, h: u32, (sx, sy): (f32, f32), angle: f32) -> Vec<f32> {
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let (sin, cos) = angle.sin_cos();
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32 - sx - cx, (i / w) as f32 - sy - cy);
            // Inverse rotation of the output position into the source.
            let (rx, ry) = (cos * x + sin * y + cx, -sin * x + cos * y + cy);
            sample(src, w, h, rx, ry)
        })
        .collect()
}

fn view(pixels: &[f32], w: u32, h: u32) -> FloatImageView<'_> {
    FloatImageView::new(pixels, w, h).expect("valid test frame")
}

fn lattice(w: u32, h: u32, margin: f32, n: usize) -> Vec<(f32, f32)> {
    let cols = (n as f32).sqrt().ceil() as usize;
    let rows = n.div_ceil(cols);
    (0..n)
        .map(|i| {
            let (c, r) = ((i % cols) as f32 + 0.5, (i / cols) as f32 + 0.5);
            (
                margin + c * (w as f32 - 2.0 * margin) / cols as f32,
                margin + r * (h as f32 - 2.0 * margin) / rows as f32,
            )
        })
        .collect()
}

fn track(
    prev: &[f32],
    next: &[f32],
    (w, h): (u32, u32),
    levels: usize,
    points: &[(f32, f32)],
    predicted: Option<&[(f32, f32)]>,
    threshold: f32,
) -> Vec<TrackResult> {
    let mut ctx = FloatTrackerContext::new();
    ctx.prepare(view(prev, w, h), view(next, w, h), levels);
    ctx.track(
        points,
        predicted,
        TrackParams {
            window_size: WIN,
            max_iterations: ITERS,
            min_eigen_threshold: threshold,
        },
    )
    .to_vec()
}

fn distance(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

const SIZE: (u32, u32) = (240, 180);

#[test]
fn subpixel_and_integer_shifts_are_recovered() {
    let (w, h) = SIZE;
    let prev = texture(w, h, 1);
    let points = lattice(w, h, 40.0, 30);
    for shift in [(0.0f32, 0.0f32), (0.3, -0.45), (1.0, 2.0), (2.7, 1.6)] {
        let next = warp(&prev, w, h, shift, 0.0);
        for result in &track(&prev, &next, SIZE, 3, &points, None, 1e-3) {
            assert_eq!(result.status, TrackStatus::Tracked, "shift {shift:?}");
        }
        for (result, &(x, y)) in track(&prev, &next, SIZE, 3, &points, None, 1e-3)
            .iter()
            .zip(&points)
        {
            let error = distance(result.pos, (x + shift.0, y + shift.1));
            assert!(error < 0.05, "shift {shift:?}: {error} px at ({x}, {y})");
        }
    }
}

#[test]
fn small_rotation_is_tracked() {
    let (w, h) = SIZE;
    let prev = texture(w, h, 2);
    let angle = 2.0f32.to_radians();
    let next = warp(&prev, w, h, (0.0, 0.0), angle);
    let points = lattice(w, h, 50.0, 25);
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let (sin, cos) = angle.sin_cos();
    for (result, &(x, y)) in track(&prev, &next, SIZE, 3, &points, None, 1e-3)
        .iter()
        .zip(&points)
    {
        let expected = (
            cos * (x - cx) - sin * (y - cy) + cx,
            sin * (x - cx) + cos * (y - cy) + cy,
        );
        assert_eq!(result.status, TrackStatus::Tracked);
        // A translational window model drifts under rotation; the 8-bit test
        // allows 0.5 px at 1.5 degrees.
        assert!(
            distance(result.pos, expected) < 0.25,
            "{:?} vs {expected:?}",
            result.pos
        );
    }
}

fn to_u8(pixels: &[f32], w: u32, h: u32) -> GrayImage {
    let bytes = pixels
        .iter()
        .map(|&v| v.round().clamp(0.0, 255.0) as u8)
        .collect();
    GrayImage::from_raw(w, h, bytes).unwrap()
}

#[test]
fn agrees_with_the_8_bit_tracker_on_8_bit_frames() {
    let (w, h) = SIZE;
    let prev_u8 = to_u8(&texture(w, h, 3), w, h);
    let next_u8 = to_u8(&warp(&texture(w, h, 3), w, h, (1.4, -0.8), 0.0), w, h);
    let as_f32 = |image: &GrayImage| image.as_raw().iter().map(|&v| v as f32).collect::<Vec<_>>();
    let (prev, next) = (as_f32(&prev_u8), as_f32(&next_u8));
    // Includes points close to and across the border.
    let mut points = lattice(w, h, 12.0, 49);
    points.extend([(3.0, 90.0), (236.5, 20.0), (120.0, 178.0)]);

    for levels in [1, 3] {
        let mut ctx = TrackerContext::new();
        ctx.prepare(&prev_u8, &next_u8, levels);
        let expected = ctx
            .track(
                &points,
                None,
                TrackParams {
                    window_size: WIN,
                    max_iterations: ITERS,
                    min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
                },
            )
            .to_vec();
        let actual = track(
            &prev,
            &next,
            SIZE,
            levels,
            &points,
            None,
            DEFAULT_MIN_EIGEN_THRESHOLD,
        );
        // One level shares every input value; several levels differ by the
        // 8-bit pyramid's truncation.
        let tolerance = if levels == 1 { 1e-3 } else { 0.05 };
        for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(a.status, e.status, "levels {levels}, point {i}");
            if e.status == TrackStatus::Tracked {
                assert!(
                    distance(a.pos, e.pos) < tolerance,
                    "levels {levels}: {a:?} vs {e:?}"
                );
                assert!(
                    (a.error - e.error).abs() < 0.1 + 0.05 * e.error,
                    "{a:?} vs {e:?}"
                );
            }
        }
    }
}

#[test]
fn tiny_physical_contrast_tracks_like_full_contrast() {
    // A thermal-like frame: 25 °C with 0.4 K of texture, and the same texture
    // at full 8-bit contrast. Displacements must agree; eigenvalues scale with
    // the contrast squared, residuals linearly.
    let (w, h) = SIZE;
    let shift = (1.3f32, 0.6f32);
    let base = texture(w, h, 4);
    let base_next = warp(&base, w, h, shift, 0.0);
    let scale = 0.4 / 255.0;
    let celsius = |pixels: &[f32]| pixels.iter().map(|&v| 25.0 + v * scale).collect::<Vec<_>>();
    let (prev, next) = (celsius(&base), celsius(&base_next));
    let points = lattice(w, h, 40.0, 30);

    let full = track(&base, &base_next, SIZE, 3, &points, None, 1e-3);
    let tiny = track(&prev, &next, SIZE, 3, &points, None, 1e-3 * scale * scale);
    for (f, t) in full.iter().zip(&tiny) {
        assert_eq!(
            (f.status, t.status),
            (TrackStatus::Tracked, TrackStatus::Tracked)
        );
        assert!(distance(f.pos, t.pos) < 0.01, "{f:?} vs {t:?}");
        assert!(
            (t.error - f.error * scale).abs() <= 0.02 * scale * 255.0,
            "{f:?} vs {t:?}"
        );
    }

    // Mapped to 8 bits at 0.78 K per grey level, the texture is about two
    // grey levels deep: fewer points track, and those that do are far less
    // accurate (about 0.25 px instead of 0.013 px mean error here).
    let gray = |pixels: &[f32]| {
        to_u8(
            &pixels
                .iter()
                .map(|&t| (t - 20.0) / 200.0 * 255.0)
                .collect::<Vec<_>>(),
            w,
            h,
        )
    };
    let mut ctx = TrackerContext::new();
    ctx.prepare(&gray(&prev), &gray(&next), 3);
    let quantized = ctx.track(
        &points,
        None,
        TrackParams {
            window_size: WIN,
            max_iterations: ITERS,
            min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
        },
    );
    let mean_error = |results: &[TrackResult]| {
        let errors: Vec<f32> = results
            .iter()
            .zip(&points)
            .filter(|(r, _)| r.status == TrackStatus::Tracked)
            .map(|(r, &(x, y))| distance(r.pos, (x + shift.0, y + shift.1)))
            .collect();
        (
            errors.len(),
            errors.iter().sum::<f32>() / errors.len().max(1) as f32,
        )
    };
    let ((f32_tracked, f32_error), (u8_tracked, u8_error)) =
        (mean_error(&tiny), mean_error(quantized));
    assert_eq!(f32_tracked, points.len());
    assert!(f32_error < 0.03, "f32 mean error {f32_error}");
    assert!(
        u8_tracked < points.len() && u8_error > 5.0 * f32_error,
        "8-bit: {u8_tracked} tracked, {u8_error} px"
    );
}

#[test]
fn eigenvalue_threshold_is_in_squared_input_units() {
    let (w, h) = SIZE;
    let prev = texture(w, h, 5);
    let next = warp(&prev, w, h, (0.5, 0.5), 0.0);
    let points = lattice(w, h, 40.0, 16);
    let statuses = |threshold: f32, factor: f32| {
        let scaled = |pixels: &[f32]| pixels.iter().map(|&v| v * factor).collect::<Vec<_>>();
        track(
            &scaled(&prev),
            &scaled(&next),
            SIZE,
            2,
            &points,
            None,
            threshold,
        )
        .iter()
        .map(|r| r.status)
        .collect::<Vec<_>>()
    };
    // Huge thresholds reject everything. Pick one that splits the windows;
    // scaling the image by 10 scales the eigenvalues by 100, so the same split
    // needs a threshold scaled the same way.
    assert!(
        statuses(1e9, 1.0)
            .iter()
            .all(|&s| s == TrackStatus::LowTexture)
    );
    let mixed = |s: &[TrackStatus]| {
        s.contains(&TrackStatus::Tracked) && s.contains(&TrackStatus::LowTexture)
    };
    let threshold = (0..60)
        .map(|k| 10.0 * 1.05f32.powi(k))
        .find(|&t| mixed(&statuses(t, 1.0)))
        .expect("some threshold splits the windows");
    let base = statuses(threshold, 1.0);
    assert_eq!(statuses(threshold * 100.0, 10.0), base);
    assert_eq!(statuses(threshold / 64.0, 0.125), base);
    // A zero threshold still tracks tiny but well-conditioned texture.
    assert!(
        statuses(0.0, 1e-5)
            .iter()
            .all(|&s| s == TrackStatus::Tracked)
    );
}

#[test]
fn flat_out_of_bounds_and_empty_inputs() {
    let (w, h) = SIZE;
    let flat = vec![300.0f32; (w * h) as usize];
    let points = [(100.0, 80.0), (2.0, 2.0), (-5.0, 50.0), (f32::NAN, 10.0)];
    let results = track(&flat, &flat, SIZE, 3, &points, None, 0.0);
    assert_eq!(results[0].status, TrackStatus::LowTexture);
    assert_eq!(results[0].error, 0.0);
    for result in &results[1..] {
        assert_eq!(result.status, TrackStatus::OutOfBounds);
        assert_eq!(result.error, f32::INFINITY);
    }
    assert!(track(&flat, &flat, SIZE, 3, &[], None, 0.0).is_empty());
}

#[test]
fn prediction_recovers_large_motion() {
    let (w, h) = SIZE;
    let prev = texture(w, h, 6);
    let shift = (19.0f32, -13.0f32);
    let next = warp(&prev, w, h, shift, 0.0);
    let points = lattice(w, h, 50.0, 20);
    let predicted: Vec<(f32, f32)> = points.iter().map(|&(x, y)| (x + 18.0, y - 12.0)).collect();
    let results = track(&prev, &next, SIZE, 2, &points, Some(&predicted), 1e-3);
    for (result, &(x, y)) in results.iter().zip(&points) {
        assert_eq!(result.status, TrackStatus::Tracked);
        assert!(distance(result.pos, (x + shift.0, y + shift.1)) < 0.05);
    }
}

#[test]
fn forward_backward_check_flags_occlusion() {
    let (w, h) = SIZE;
    let prev = texture(w, h, 7);
    let mut next = warp(&prev, w, h, (1.0, 1.0), 0.0);
    // Paste unrelated texture over a square around (60, 60).
    let other = texture(w, h, 99);
    for y in 40..80 {
        for x in 40..80 {
            next[(y * w + x) as usize] = other[(y * w + x) as usize];
        }
    }
    let points = [(60.0, 60.0), (170.0, 120.0)];
    let mut ctx = FloatTrackerContext::new();
    ctx.prepare(view(&prev, w, h), view(&next, w, h), 3);
    let results = ctx.track_fb(
        &points,
        None,
        TrackParams {
            window_size: WIN,
            max_iterations: ITERS,
            min_eigen_threshold: 1e-3,
        },
        DEFAULT_FB_THRESHOLD,
    );
    assert_ne!(results[0].status, TrackStatus::Tracked);
    assert_eq!(results[1].status, TrackStatus::Tracked);
}

#[test]
fn repeated_and_reused_tracking_is_deterministic() {
    let (w, h) = SIZE;
    let frames: Vec<Vec<f32>> = (0..3)
        .map(|i| warp(&texture(w, h, 8), w, h, (i as f32 * 0.7, 0.3), 0.0))
        .collect();
    let points = lattice(w, h, 20.0, 40);
    let mut ctx = FloatTrackerContext::new();
    let mut first = Vec::new();
    for round in 0..2 {
        for pair in frames.windows(2) {
            ctx.prepare(view(&pair[0], w, h), view(&pair[1], w, h), 3);
            let results = ctx
                .track_fb(
                    &points,
                    None,
                    TrackParams {
                        window_size: 15,
                        max_iterations: 15,
                        min_eigen_threshold: 1e-3,
                    },
                    DEFAULT_FB_THRESHOLD,
                )
                .to_vec();
            let mut fresh = FloatTrackerContext::new();
            fresh.prepare(view(&pair[0], w, h), view(&pair[1], w, h), 3);
            assert_eq!(
                fresh.track_fb(
                    &points,
                    None,
                    TrackParams {
                        window_size: 15,
                        max_iterations: 15,
                        min_eigen_threshold: 1e-3
                    },
                    DEFAULT_FB_THRESHOLD
                ),
                results
            );
            if round == 0 {
                first.push(results);
            } else {
                assert!(first.contains(&results));
            }
        }
    }
}

#[test]
fn advance_matches_prepare_on_a_frame_sequence() {
    let (w, h) = SIZE;
    let base = texture(w, h, 10);
    let frames: Vec<Vec<f32>> = [((0.0, 0.0), 0.0), ((1.6, -0.9), 0.0), ((3.1, 0.4), 0.015)]
        .into_iter()
        .map(|(motion, angle)| {
            // Small physical contrast, as in a thermal frame in °C.
            warp(&base, w, h, motion, angle)
                .iter()
                .map(|v| 21.0 + v * 0.01)
                .collect()
        })
        .collect();
    let params = TrackParams {
        window_size: 15,
        max_iterations: 20,
        min_eigen_threshold: 1e-6,
    };
    let bits = |results: &[TrackResult]| -> Vec<(u32, u32, TrackStatus, u32)> {
        results
            .iter()
            .map(|r| {
                (
                    r.pos.0.to_bits(),
                    r.pos.1.to_bits(),
                    r.status,
                    r.error.to_bits(),
                )
            })
            .collect()
    };

    let mut prepared = FloatTrackerContext::new();
    let mut prepared_points = lattice(w, h, 12.0, 30);
    let mut advanced = FloatTrackerContext::new();
    let mut advanced_points = prepared_points.clone();
    advanced.advance(view(&frames[0], w, h), 3);

    for pair in frames.windows(2) {
        prepared.prepare(view(&pair[0], w, h), view(&pair[1], w, h), 3);
        advanced.advance(view(&pair[1], w, h), 3);
        assert_eq!(advanced.prev_pyramid(), prepared.prev_pyramid());
        assert_eq!(advanced.next_pyramid(), prepared.next_pyramid());

        let expected = prepared
            .track_fb(&prepared_points, None, params, DEFAULT_FB_THRESHOLD)
            .to_vec();
        let results = advanced
            .track_fb(&advanced_points, None, params, DEFAULT_FB_THRESHOLD)
            .to_vec();
        assert_eq!(bits(&results), bits(&expected));
        assert!(results.iter().any(|r| r.status == TrackStatus::Tracked));

        prepared_points = expected.iter().map(|r| r.pos).collect();
        advanced_points = results.iter().map(|r| r.pos).collect();
    }

    // A change of `levels` rebuilds the previous pyramid from the last frame,
    // a change of size starts over from the new frame.
    let mut reference = FloatTrackerContext::new();
    advanced.advance(view(&frames[0], w, h), 2);
    reference.prepare(view(&frames[2], w, h), view(&frames[0], w, h), 2);
    assert_eq!(advanced.prev_pyramid(), reference.prev_pyramid());
    assert_eq!(advanced.next_pyramid(), reference.next_pyramid());
    let small = texture(40, 30, 11);
    advanced.advance(view(&small, 40, 30), 2);
    reference.prepare(view(&small, 40, 30), view(&small, 40, 30), 2);
    assert_eq!(advanced.prev_pyramid(), reference.prev_pyramid());
    assert_eq!(advanced.next_pyramid(), reference.next_pyramid());
}

#[test]
fn cached_gradients_and_options_match_a_fresh_context() {
    // The context keeps each frame's gradients across passes and frames; a
    // fresh context computes them again. The options alternate so that a
    // frame becomes the previous frame with only some levels computed.
    let (w, h) = SIZE;
    let base = texture(w, h, 12);
    let frames: Vec<Vec<f32>> = [(0.0, 0.0), (1.6, -0.9), (3.1, 0.4), (-2.2, 1.5)]
        .into_iter()
        .map(|motion| {
            warp(&base, w, h, motion, 0.0)
                .iter()
                .map(|v| 21.0 + v * 0.01)
                .collect()
        })
        .collect();
    let params = TrackParams {
        window_size: 15,
        max_iterations: 20,
        min_eigen_threshold: 1e-6,
    };
    let bits = |results: &[TrackResult]| -> Vec<(u32, u32, TrackStatus, u32)> {
        results
            .iter()
            .map(|r| {
                (
                    r.pos.0.to_bits(),
                    r.pos.1.to_bits(),
                    r.status,
                    r.error.to_bits(),
                )
            })
            .collect()
    };
    let points = lattice(w, h, 6.0, 40);
    let option_sets = [
        TrackOptions::new().with_fb_levels(1),
        TrackOptions::new().with_start_level(1),
        TrackOptions::new(),
    ];
    let mut ctx = FloatTrackerContext::new();
    ctx.advance(view(&frames[0], w, h), 3);
    for (step, pair) in frames.windows(2).enumerate() {
        ctx.advance(view(&pair[1], w, h), 3);
        let mut fresh = FloatTrackerContext::new();
        fresh.prepare(view(&pair[0], w, h), view(&pair[1], w, h), 3);
        let options = option_sets[step % option_sets.len()];
        for _ in 0..2 {
            let cached = ctx
                .track_fb_with(&points, None, params, DEFAULT_FB_THRESHOLD, options)
                .to_vec();
            let recomputed =
                fresh.track_fb_with(&points, None, params, DEFAULT_FB_THRESHOLD, options);
            assert_eq!(bits(&cached), bits(recomputed), "step {step}");
            let cached = ctx.track_with(&points, None, params, options).to_vec();
            assert_eq!(
                bits(&cached),
                bits(fresh.track_with(&points, None, params, options))
            );
        }
        // The kernel option does not apply to f32 frames.
        let reference = TrackOptions::new().with_kernel(TrackKernel::Reference);
        let fixed = TrackOptions::new().with_kernel(TrackKernel::FixedPoint);
        let a = ctx.track_with(&points, None, params, reference).to_vec();
        assert_eq!(bits(&a), bits(ctx.track_with(&points, None, params, fixed)));
    }
}

#[test]
#[should_panic(expected = "same size")]
fn frames_of_different_size_panic() {
    let a = vec![0.0f32; 20 * 10];
    let b = vec![0.0f32; 10 * 20];
    FloatTrackerContext::new().prepare(view(&a, 20, 10), view(&b, 10, 20), 2);
}

#[test]
#[should_panic(expected = "one entry per prev_point")]
fn prediction_length_mismatch_panics() {
    let (w, h) = SIZE;
    let frame = texture(w, h, 9);
    track(&frame, &frame, SIZE, 1, &[(50.0, 50.0)], Some(&[]), 1e-3);
}

#[test]
#[should_panic(expected = "Window size must be odd")]
fn even_window_panics() {
    let frame = vec![1.0f32; 32 * 32];
    let mut ctx = FloatTrackerContext::new();
    ctx.prepare(view(&frame, 32, 32), view(&frame, 32, 32), 1);
    ctx.track(
        &[(16.0, 16.0)],
        None,
        TrackParams {
            window_size: 8,
            max_iterations: 10,
            min_eigen_threshold: 0.0,
        },
    );
}
