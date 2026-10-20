//! End-to-end synthetic tests for detection, tracking, status codes,
//! prediction, the forward-backward check and grid detection.

use image::{GrayImage, Luma};
use optical_flow_lk::{
    Corner, DEFAULT_FB_THRESHOLD, DEFAULT_MIN_EIGEN_THRESHOLD, Grid, TrackKernel, TrackOptions,
    TrackParams, TrackResult, TrackStatus, TrackerCapacity, TrackerContext, build_pyramid,
    calc_optical_flow_ex, calc_optical_flow_fb, good_features_to_track_grid,
};

const WIN: usize = 21;
const ITERS: usize = 30;
const PARAMS: TrackParams = TrackParams {
    window_size: WIN,
    max_iterations: ITERS,
    min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
};

/// Deterministic, band-limited texture (noise smoothed with a few box blurs) so
/// the pyramid does not alias and LK behaves like it would on a real image.
fn textured(w: u32, h: u32) -> GrayImage {
    let mut buf: Vec<f32> = (0..(w * h))
        .map(|i| {
            let mut s = i.wrapping_mul(2654435761) ^ 0x9e3779b9;
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
    let mut img = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img.put_pixel(x, y, Luma([buf[(y * w + x) as usize] as u8]));
        }
    }
    img
}

/// Bilinear resample of `src` at `(x, y)`; out-of-frame reads as 0.
fn sample(src: &GrayImage, x: f32, y: f32) -> f32 {
    let (w, h) = src.dimensions();
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let (dx, dy) = (x - x0 as f32, y - y0 as f32);
    let g = |xx: i32, yy: i32| {
        if xx >= 0 && yy >= 0 && xx < w as i32 && yy < h as i32 {
            src.get_pixel(xx as u32, yy as u32)[0] as f32
        } else {
            0.0
        }
    };
    g(x0, y0) * (1.0 - dx) * (1.0 - dy)
        + g(x0 + 1, y0) * dx * (1.0 - dy)
        + g(x0, y0 + 1) * (1.0 - dx) * dy
        + g(x0 + 1, y0 + 1) * dx * dy
}

/// Shift the image so that content moves by `(sx, sy)`.
fn shift(src: &GrayImage, sx: f32, sy: f32) -> GrayImage {
    let (w, h) = src.dimensions();
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = sample(src, x as f32 - sx, y as f32 - sy);
            out.put_pixel(x, y, Luma([v as u8]));
        }
    }
    out
}

/// Rotate the image by `angle` radians about `(cx, cy)`.
fn rotate(src: &GrayImage, angle: f32, cx: f32, cy: f32) -> GrayImage {
    let (w, h) = src.dimensions();
    let (s, c) = angle.sin_cos();
    let mut out = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            // Map output pixel back into the source frame.
            let (ox, oy) = (x as f32 - cx, y as f32 - cy);
            let srcx = c * ox + s * oy + cx;
            let srcy = -s * ox + c * oy + cy;
            out.put_pixel(x, y, Luma([sample(src, srcx, srcy) as u8]));
        }
    }
    out
}

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

#[test]
fn integer_and_subpixel_shift_under_0_2px() {
    let prev = textured(320, 240);
    let pts = vec![
        (160.0f32, 120.0),
        (90.0, 80.0),
        (220.0, 150.0),
        (130.0, 175.0),
    ];

    for (sx, sy) in [(3.0f32, 2.0f32), (0.4, -0.7), (5.0, 0.0), (-0.35, 0.85)] {
        let next = shift(&prev, sx, sy);
        let pp = build_pyramid(&prev, 4);
        let np = build_pyramid(&next, 4);
        let res = calc_optical_flow_ex(&pp, &np, &pts, None, PARAMS);

        for (i, r) in res.iter().enumerate() {
            let exp = (pts[i].0 + sx, pts[i].1 + sy);
            assert_eq!(r.status, TrackStatus::Tracked);
            let e = dist(r.pos, exp);
            assert!(e < 0.2, "shift ({sx},{sy}) pt{i}: err {e} >= 0.2");
        }
    }
}

#[test]
fn small_rotation_tracks_accurately() {
    let prev = textured(320, 240);
    let (cx, cy) = (160.0f32, 120.0f32);
    let angle = 1.5f32.to_radians();
    let next = rotate(&prev, angle, cx, cy);

    // Points moderately close to the center so the rotation is near-translational
    // within the window.
    let pts = vec![
        (130.0f32, 110.0),
        (190.0, 130.0),
        (150.0, 150.0),
        (175.0, 95.0),
    ];
    let pp = build_pyramid(&prev, 4);
    let np = build_pyramid(&next, 4);
    let res = calc_optical_flow_ex(&pp, &np, &pts, None, PARAMS);

    let (s, c) = angle.sin_cos();
    for (i, r) in res.iter().enumerate() {
        // Forward rotation of the point about the center.
        let (ox, oy) = (pts[i].0 - cx, pts[i].1 - cy);
        let exp = (c * ox - s * oy + cx, s * ox + c * oy + cy);
        assert_eq!(r.status, TrackStatus::Tracked, "pt{i}");
        let e = dist(r.pos, exp);
        assert!(e < 0.5, "rotation pt{i}: err {e} >= 0.5");
    }
}

#[test]
fn point_leaving_frame_is_out_of_bounds() {
    let prev = textured(200, 200);
    let next = shift(&prev, 4.0, 0.0);
    let pts = vec![(100.0f32, 100.0)];
    // A prediction that lands the search window across the right edge: the
    // window leaves the image, so tracking must report OutOfBounds.
    let predicted = vec![(196.0f32, 100.0)];
    let pp = build_pyramid(&prev, 4);
    let np = build_pyramid(&next, 4);
    let res = calc_optical_flow_ex(&pp, &np, &pts, Some(&predicted), PARAMS);

    assert_eq!(res[0].status, TrackStatus::OutOfBounds);
    assert!(res[0].error.is_infinite());
}

#[test]
fn flat_region_is_low_texture() {
    // Constant image -> zero gradients -> minimum eigenvalue 0 everywhere.
    let prev = GrayImage::from_pixel(200, 200, Luma([128]));
    let next = GrayImage::from_pixel(200, 200, Luma([128]));
    let pts = vec![(100.0f32, 100.0)];
    let pp = build_pyramid(&prev, 4);
    let np = build_pyramid(&next, 4);
    let res = calc_optical_flow_ex(&pp, &np, &pts, None, PARAMS);

    assert_eq!(res[0].status, TrackStatus::LowTexture);
}

#[test]
fn prediction_improves_large_displacement() {
    let prev = textured(320, 240);
    let (sx, sy) = (26.0f32, 23.0f32);
    let next = shift(&prev, sx, sy);
    let pts = vec![(160.0f32, 120.0), (110.0, 95.0), (205.0, 150.0)];
    let predicted: Vec<(f32, f32)> = pts.iter().map(|&(x, y)| (x + sx, y + sy)).collect();

    let pp = build_pyramid(&prev, 4);
    let np = build_pyramid(&next, 4);

    let with = calc_optical_flow_ex(&pp, &np, &pts, Some(&predicted), PARAMS);
    let without = calc_optical_flow_ex(&pp, &np, &pts, None, PARAMS);

    let mut improved = 0;
    for (i, w) in with.iter().enumerate() {
        let exp = (pts[i].0 + sx, pts[i].1 + sy);
        assert_eq!(
            w.status,
            TrackStatus::Tracked,
            "pt{i} should track with guess"
        );
        assert!(dist(w.pos, exp) < 0.2, "pt{i} guided err too large");
        if dist(without[i].pos, exp) > 1.0 {
            improved += 1;
        }
    }
    assert!(
        improved > 0,
        "prediction should rescue points the zero-init path loses"
    );
}

/// Stamp a textured occluder (copied from a distant region) so the forward pass
/// can confidently latch onto a *wrong* match.
fn occlude_textured(img: &mut GrayImage, src: &GrayImage, cx: i32, cy: i32, half: i32) {
    let (w, h) = img.dimensions();
    let (ox, oy) = (90, 70);
    for y in (cy - half)..=(cy + half) {
        for x in (cx - half)..=(cx + half) {
            let (sx, sy) = (x + ox, y + oy);
            if x >= 0
                && y >= 0
                && x < w as i32
                && y < h as i32
                && sx >= 0
                && sy >= 0
                && sx < w as i32
                && sy < h as i32
            {
                let v = src.get_pixel(sx as u32, sy as u32)[0];
                img.put_pixel(x as u32, y as u32, Luma([v]));
            }
        }
    }
}

#[test]
fn fb_check_catches_occlusion() {
    let prev = textured(320, 240);
    let (sx, sy) = (3.0f32, 2.0f32);
    let mut next = shift(&prev, sx, sy);

    let open = (80.0f32, 70.0f32);
    let hidden = (200.0f32, 150.0f32);
    occlude_textured(&mut next, &prev, 200 + sx as i32, 150 + sy as i32, 22);

    let pp = build_pyramid(&prev, 4);
    let np = build_pyramid(&next, 4);
    let pts = vec![open, hidden];

    let res = calc_optical_flow_fb(&pp, &np, &pts, None, PARAMS, DEFAULT_FB_THRESHOLD);

    assert_eq!(
        res[0].status,
        TrackStatus::Tracked,
        "open point should survive FB"
    );
    assert_ne!(
        res[1].status,
        TrackStatus::Tracked,
        "occluded point must be rejected"
    );
}

#[test]
fn context_matches_free_functions() {
    let prev = textured(320, 240);
    let next = shift(&prev, 2.0, -1.5);
    let pts = vec![(160.0f32, 120.0), (90.0, 80.0), (210.0, 160.0)];

    let pp = build_pyramid(&prev, 4);
    let np = build_pyramid(&next, 4);
    let free = calc_optical_flow_ex(&pp, &np, &pts, None, PARAMS);

    let mut ctx = TrackerContext::new();
    ctx.prepare(&prev, &next, 4);
    let ctx_res = ctx.track(&pts, None, PARAMS);

    for (a, b) in free.iter().zip(ctx_res.iter()) {
        assert_eq!(a.pos, b.pos);
        assert_eq!(a.status, b.status);
    }
}

/// Bit patterns of every field, so that equal results are identical, not
/// merely equal as floats.
fn bits(results: &[TrackResult]) -> Vec<(u32, u32, TrackStatus, u32)> {
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
}

#[test]
fn advance_matches_prepare_on_a_frame_sequence() {
    let base = textured(320, 240);
    let first = shift(&base, 2.5, -1.25);
    let frames = [
        base.clone(),
        first.clone(),
        rotate(&first, 0.02, 150.0, 110.0),
    ];
    let start = vec![
        (160.0f32, 120.0),
        (90.0, 80.0),
        (210.0, 160.0),
        (40.0, 200.0),
        (300.0, 30.0),
        (5.0, 5.0),
    ];
    let predicted_shift = (2.0, -1.0);

    // Per-frame user: the last frame is kept and both pyramids are rebuilt.
    let mut prepared = TrackerContext::new();
    let mut prepared_points = start.clone();
    // Per-frame user of `advance`: only the new frame is passed.
    let mut advanced = TrackerContext::new();
    let mut advanced_points = start.clone();
    advanced.advance(&frames[0], 4);

    for pair in frames.windows(2) {
        prepared.prepare(&pair[0], &pair[1], 4);
        advanced.advance(&pair[1], 4);
        assert_eq!(advanced.prev_pyramid(), prepared.prev_pyramid());
        assert_eq!(advanced.next_pyramid(), prepared.next_pyramid());

        let predicted: Vec<(f32, f32)> = prepared_points
            .iter()
            .map(|&(x, y)| (x + predicted_shift.0, y + predicted_shift.1))
            .collect();
        let expected = prepared
            .track_fb(&prepared_points, None, PARAMS, DEFAULT_FB_THRESHOLD)
            .to_vec();
        let expected_guessed = bits(prepared.track(&prepared_points, Some(&predicted), PARAMS));
        let results = advanced
            .track_fb(&advanced_points, None, PARAMS, DEFAULT_FB_THRESHOLD)
            .to_vec();
        assert_eq!(bits(&results), bits(&expected));
        assert_eq!(
            bits(advanced.track(&advanced_points, Some(&predicted), PARAMS)),
            expected_guessed
        );
        assert!(
            results.iter().any(|r| r.status == TrackStatus::Tracked),
            "nothing tracked: {results:?}"
        );

        prepared_points = expected.iter().map(|r| r.pos).collect();
        advanced_points = results.iter().map(|r| r.pos).collect();
    }
}

#[test]
fn advance_rebuilds_what_it_cannot_reuse() {
    let a = textured(160, 120);
    let b = shift(&a, 1.5, 0.5);
    let c = textured(200, 100);
    let assert_pyramids = |ctx: &TrackerContext, prev: &GrayImage, next: &GrayImage, levels| {
        assert_eq!(ctx.prev_pyramid(), build_pyramid(prev, levels));
        assert_eq!(ctx.next_pyramid(), build_pyramid(next, levels));
    };

    let mut ctx = TrackerContext::new();
    // No last frame: both pyramids show the first frame.
    ctx.advance(&a, 3);
    assert_pyramids(&ctx, &a, &a, 3);
    // Fewer and more levels: the previous pyramid is rebuilt from the last frame.
    ctx.advance(&b, 2);
    assert_pyramids(&ctx, &a, &b, 2);
    ctx.advance(&a, 4);
    assert_pyramids(&ctx, &b, &a, 4);
    // More levels than the frame size allows behave like the largest count.
    ctx.advance(&b, 40);
    assert_pyramids(&ctx, &a, &b, 40);
    ctx.advance(&a, 30);
    assert_pyramids(&ctx, &b, &a, 30);
    // A frame of another size starts over.
    ctx.advance(&c, 3);
    assert_pyramids(&ctx, &c, &c, 3);
    // `prepare` sets the last frame as well.
    ctx.prepare(&a, &b, 3);
    ctx.advance(&a, 3);
    assert_pyramids(&ctx, &b, &a, 3);

    // Reserved pyramids hold no frame yet.
    let capacity = TrackerCapacity {
        width: 160,
        height: 120,
        levels: 3,
        window_size: WIN,
        points: 10,
    };
    let mut reserved = TrackerContext::with_capacity(capacity);
    reserved.advance(&b, 3);
    assert_pyramids(&reserved, &b, &b, 3);
    // Reserving more levels keeps the last frame.
    reserved.reserve(TrackerCapacity {
        levels: 5,
        ..capacity
    });
    reserved.advance(&a, 5);
    assert_pyramids(&reserved, &b, &a, 5);
    // Reserving another frame size discards it, also for frames of that size.
    reserved.reserve(TrackerCapacity {
        width: 200,
        height: 100,
        ..capacity
    });
    reserved.advance(&c, 3);
    assert_pyramids(&reserved, &c, &c, 3);
}

#[test]
fn grid_detection_is_uniform_and_respects_occupancy() {
    let img = textured(320, 240);
    let (cols, rows, max_per_cell, min_dist) = (4u32, 3u32, 3u32, 8u32);

    let pts = good_features_to_track_grid(
        &img,
        Grid {
            cols,
            rows,
            max_per_cell,
        },
        0.05,
        min_dist,
        &[],
    );

    // Spacing respected.
    for i in 0..pts.len() {
        for j in (i + 1)..pts.len() {
            let d2 = (pts[i].x as i32 - pts[j].x as i32).pow(2)
                + (pts[i].y as i32 - pts[j].y as i32).pow(2);
            assert!(
                d2 >= (min_dist * min_dist) as i32,
                "points closer than min_distance"
            );
        }
    }

    // Per-cell budget respected and coverage is broad.
    let mut counts = vec![0u32; (cols * rows) as usize];
    for &Corner { x, y, .. } in &pts {
        let c = (x * cols / 320).min(cols - 1);
        let r = (y * rows / 240).min(rows - 1);
        counts[(r * cols + c) as usize] += 1;
    }
    assert!(counts.iter().all(|&c| c <= max_per_cell));
    assert!(
        counts.iter().filter(|&&c| c > 0).count() >= 10,
        "should cover most cells"
    );

    // A cell pre-filled by existing points yields no new detections there.
    let existing: Vec<(f32, f32)> = vec![(20.0, 20.0), (40.0, 30.0), (60.0, 10.0)];
    let pts2 = good_features_to_track_grid(
        &img,
        Grid {
            cols,
            rows,
            max_per_cell,
        },
        0.05,
        min_dist,
        &existing,
    );
    let in_cell0 = pts2
        .iter()
        .filter(|corner| corner.x * cols / 320 == 0 && corner.y * rows / 240 == 0)
        .count();
    assert_eq!(
        in_cell0, 0,
        "cell filled by existing_points must be skipped"
    );
}

#[test]
#[should_panic(expected = "frames must have the same size")]
fn context_frames_of_different_size_panic() {
    TrackerContext::new().prepare(&GrayImage::new(40, 30), &GrayImage::new(30, 40), 2);
}

#[test]
#[should_panic(expected = "same level sizes")]
fn larger_next_pyramid_panics_instead_of_misreading_it() {
    // A larger next frame used to be read with the previous frame's width.
    let prev = build_pyramid(&textured(64, 48), 2);
    let next = build_pyramid(&textured(80, 48), 2);
    calc_optical_flow_ex(&prev, &next, &[(30.0, 20.0)], None, PARAMS);
}

/// A frame sequence with motion between every pair and points that reach the
/// borders.
fn sequence() -> (Vec<GrayImage>, Vec<(f32, f32)>) {
    let base = textured(320, 240);
    let first = shift(&base, 2.5, -1.25);
    let frames = vec![
        base.clone(),
        first.clone(),
        rotate(&first, 0.02, 150.0, 110.0),
        shift(&base, -4.0, 3.5),
        base,
    ];
    let points = (0..60)
        .map(|i| (6.0 + (i % 10) as f32 * 34.0, 5.0 + (i / 10) as f32 * 45.0))
        .collect();
    (frames, points)
}

#[test]
fn cached_gradients_give_the_results_of_recomputed_ones() {
    // A context computes each frame's gradients once and keeps them across
    // passes and frames; the free functions and a fresh context compute them
    // again. Option sets alternate so that a frame becomes the previous frame
    // with only some of its levels computed (the backward pass over the
    // finest level computes only level 0).
    let (frames, points) = sequence();
    let option_sets = [
        TrackOptions::new(),
        TrackOptions::new().with_fb_levels(1),
        TrackOptions::new().with_start_level(1),
        TrackOptions::new()
            .with_fb_levels(2)
            .with_kernel(TrackKernel::Reference),
        TrackOptions::new().with_kernel(TrackKernel::FixedPoint),
    ];
    let mut ctx = TrackerContext::new();
    ctx.advance(&frames[0], 4);
    for (step, pair) in frames.windows(2).enumerate() {
        ctx.advance(&pair[1], 4);
        let options = option_sets[step % option_sets.len()];
        let mut fresh = TrackerContext::new();
        fresh.prepare(&pair[0], &pair[1], 4);
        for _ in 0..2 {
            let cached = ctx
                .track_fb_with(&points, None, PARAMS, DEFAULT_FB_THRESHOLD, options)
                .to_vec();
            let recomputed =
                fresh.track_fb_with(&points, None, PARAMS, DEFAULT_FB_THRESHOLD, options);
            assert_eq!(bits(&cached), bits(recomputed), "step {step}, {options:?}");
            let cached = ctx.track_with(&points, None, PARAMS, options).to_vec();
            let recomputed = fresh.track_with(&points, None, PARAMS, options);
            assert_eq!(bits(&cached), bits(recomputed), "step {step}, {options:?}");
        }
        // The free functions recompute every gradient per call.
        let free = calc_optical_flow_fb(
            ctx.prev_pyramid(),
            ctx.next_pyramid(),
            &points,
            None,
            PARAMS,
            DEFAULT_FB_THRESHOLD,
        );
        let cached = ctx.track_fb(&points, None, PARAMS, DEFAULT_FB_THRESHOLD);
        assert_eq!(bits(cached), bits(&free), "step {step}");
        let free = calc_optical_flow_ex(
            ctx.prev_pyramid(),
            ctx.next_pyramid(),
            &points,
            None,
            PARAMS,
        );
        assert_eq!(
            bits(ctx.track(&points, None, PARAMS)),
            bits(&free),
            "step {step}"
        );
    }
}

#[test]
fn free_functions_use_the_automatic_kernel() {
    let (frames, points) = sequence();
    let mut ctx = TrackerContext::new();
    ctx.prepare(&frames[0], &frames[1], 4);
    let free = calc_optical_flow_ex(
        ctx.prev_pyramid(),
        ctx.next_pyramid(),
        &points,
        None,
        PARAMS,
    );
    let auto = ctx.track_with(
        &points,
        None,
        PARAMS,
        TrackOptions::new().with_kernel(TrackKernel::Auto),
    );
    assert_eq!(bits(auto), bits(&free));
    // Auto is one of the two kernels.
    let fixed = bits(ctx.track_with(
        &points,
        None,
        PARAMS,
        TrackOptions::new().with_kernel(TrackKernel::FixedPoint),
    ));
    let reference = bits(ctx.track_with(
        &points,
        None,
        PARAMS,
        TrackOptions::new().with_kernel(TrackKernel::Reference),
    ));
    let free = bits(&free);
    assert!(free == fixed || free == reference);
    // Windows above the fixed-point limit always use the reference kernel.
    let wide = TrackParams {
        window_size: 129,
        ..PARAMS
    };
    let points = [(160.0, 120.0)];
    let results: Vec<_> = [TrackKernel::FixedPoint, TrackKernel::Reference]
        .map(|kernel| {
            bits(ctx.track_with(&points, None, wide, TrackOptions::new().with_kernel(kernel)))
        })
        .into();
    assert_eq!(results[0], results[1]);
}

#[test]
fn start_level_skips_the_coarse_levels() {
    let prev = textured(320, 240);
    let (sx, sy) = (26.0f32, 23.0f32);
    let next = shift(&prev, sx, sy);
    let points = vec![(160.0f32, 120.0), (110.0, 95.0), (205.0, 150.0)];
    let exact: Vec<(f32, f32)> = points.iter().map(|&(x, y)| (x + sx, y + sy)).collect();
    // A prediction off by about a pixel, as from a motion model.
    let predicted: Vec<(f32, f32)> = exact.iter().map(|&(x, y)| (x - 0.8, y + 0.6)).collect();
    let mut ctx = TrackerContext::new();
    ctx.prepare(&prev, &next, 4);

    // The top level, or any level above it, is the default.
    let default = bits(ctx.track(&points, Some(&predicted), PARAMS));
    for level in [3, 4, 100] {
        let options = TrackOptions::new().with_start_level(level);
        let results = ctx.track_with(&points, Some(&predicted), PARAMS, options);
        assert_eq!(bits(results), default, "start level {level}");
    }

    // From level 0 the prediction alone carries the motion.
    for level in [0, 1] {
        let options = TrackOptions::new().with_start_level(level);
        let results = ctx.track_with(&points, Some(&predicted), PARAMS, options);
        for (r, &e) in results.iter().zip(&exact) {
            assert_eq!(r.status, TrackStatus::Tracked, "start level {level}");
            assert!(
                dist(r.pos, e) < 0.2,
                "start level {level}: {:?} vs {e:?}",
                r.pos
            );
        }
    }
    // Without a prediction, level 0 cannot recover 35 px of motion.
    let options = TrackOptions::new().with_start_level(0);
    let results = ctx.track_with(&points, None, PARAMS, options);
    assert!(
        results
            .iter()
            .zip(&exact)
            .all(|(r, &e)| r.status != TrackStatus::Tracked || dist(r.pos, e) > 1.0)
    );
}

#[test]
fn fb_levels_limit_the_backward_pass() {
    let (frames, points) = sequence();
    let mut ctx = TrackerContext::new();
    ctx.prepare(&frames[0], &frames[1], 4);
    let all = bits(ctx.track_fb(&points, None, PARAMS, DEFAULT_FB_THRESHOLD));
    // All levels, or more, are the default.
    for levels in [4, 5, 100] {
        let options = TrackOptions::new().with_fb_levels(levels);
        let results = ctx.track_fb_with(&points, None, PARAMS, DEFAULT_FB_THRESHOLD, options);
        assert_eq!(bits(results), all, "fb_levels {levels}");
    }
    // The forward pass is the same with any backward pass: only points the
    // forward pass tracks can change status, and only to or from
    // `FbInconsistent`.
    let forward = ctx.track(&points, None, PARAMS).to_vec();
    for levels in [1, 2, 3] {
        let options = TrackOptions::new().with_fb_levels(levels);
        let results = ctx.track_fb_with(&points, None, PARAMS, DEFAULT_FB_THRESHOLD, options);
        let mut tracked = 0;
        for (r, f) in results.iter().zip(&forward) {
            assert_eq!(r.pos, f.pos);
            assert_eq!(r.error.to_bits(), f.error.to_bits());
            if f.status == TrackStatus::Tracked {
                assert!(matches!(
                    r.status,
                    TrackStatus::Tracked | TrackStatus::FbInconsistent
                ));
                tracked += (r.status == TrackStatus::Tracked) as usize;
            } else {
                assert_eq!(r.status, f.status);
            }
        }
        assert!(
            tracked > points.len() / 2,
            "fb_levels {levels}: {tracked} tracked"
        );
    }
}

#[test]
fn fb_check_over_the_finest_level_still_catches_occlusion() {
    let prev = textured(320, 240);
    let (sx, sy) = (3.0f32, 2.0f32);
    let mut next = shift(&prev, sx, sy);
    occlude_textured(&mut next, &prev, 200 + sx as i32, 150 + sy as i32, 22);
    let mut ctx = TrackerContext::new();
    ctx.prepare(&prev, &next, 4);
    let options = TrackOptions::new().with_fb_levels(1);
    let points = [(80.0f32, 70.0f32), (200.0, 150.0)];
    let results = ctx.track_fb_with(&points, None, PARAMS, DEFAULT_FB_THRESHOLD, options);
    assert_eq!(results[0].status, TrackStatus::Tracked);
    assert_ne!(results[1].status, TrackStatus::Tracked);
}

#[test]
#[should_panic(expected = "fb_levels must be at least 1")]
fn zero_fb_levels_panic() {
    let image = textured(64, 48);
    let mut ctx = TrackerContext::new();
    ctx.prepare(&image, &image, 2);
    let options = TrackOptions::new().with_fb_levels(0);
    ctx.track_fb_with(&[(30.0, 20.0)], None, PARAMS, DEFAULT_FB_THRESHOLD, options);
}
