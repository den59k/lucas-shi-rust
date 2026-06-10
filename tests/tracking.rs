//! End-to-end synthetic tests for detection, tracking, status codes,
//! prediction, the forward-backward check and grid detection.

use image::{GrayImage, Luma};
use optical_flow_lk::{
    DEFAULT_FB_THRESHOLD, DEFAULT_MIN_EIGEN_THRESHOLD, TrackStatus, TrackerContext, build_pyramid,
    calc_optical_flow_ex, calc_optical_flow_fb, good_features_to_track_grid,
};

const WIN: usize = 21;
const ITERS: usize = 30;

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
        let res = calc_optical_flow_ex(
            &pp,
            &np,
            &pts,
            None,
            WIN,
            ITERS,
            DEFAULT_MIN_EIGEN_THRESHOLD,
        );

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
    let res = calc_optical_flow_ex(
        &pp,
        &np,
        &pts,
        None,
        WIN,
        ITERS,
        DEFAULT_MIN_EIGEN_THRESHOLD,
    );

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
    let res = calc_optical_flow_ex(
        &pp,
        &np,
        &pts,
        Some(&predicted),
        WIN,
        ITERS,
        DEFAULT_MIN_EIGEN_THRESHOLD,
    );

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
    let res = calc_optical_flow_ex(
        &pp,
        &np,
        &pts,
        None,
        WIN,
        ITERS,
        DEFAULT_MIN_EIGEN_THRESHOLD,
    );

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

    let with = calc_optical_flow_ex(
        &pp,
        &np,
        &pts,
        Some(&predicted),
        WIN,
        ITERS,
        DEFAULT_MIN_EIGEN_THRESHOLD,
    );
    let without = calc_optical_flow_ex(
        &pp,
        &np,
        &pts,
        None,
        WIN,
        ITERS,
        DEFAULT_MIN_EIGEN_THRESHOLD,
    );

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

    let res = calc_optical_flow_fb(
        &pp,
        &np,
        &pts,
        None,
        WIN,
        ITERS,
        DEFAULT_MIN_EIGEN_THRESHOLD,
        DEFAULT_FB_THRESHOLD,
    );

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
    let free = calc_optical_flow_ex(
        &pp,
        &np,
        &pts,
        None,
        WIN,
        ITERS,
        DEFAULT_MIN_EIGEN_THRESHOLD,
    );

    let mut ctx = TrackerContext::new();
    ctx.prepare(&prev, &next, 4);
    let ctx_res = ctx.track(&pts, None, WIN, ITERS, DEFAULT_MIN_EIGEN_THRESHOLD);

    for (a, b) in free.iter().zip(ctx_res.iter()) {
        assert_eq!(a.pos, b.pos);
        assert_eq!(a.status, b.status);
    }
}

#[test]
fn grid_detection_is_uniform_and_respects_occupancy() {
    let img = textured(320, 240);
    let (cols, rows, max_per_cell, min_dist) = (4u32, 3u32, 3u32, 8u32);

    let pts = good_features_to_track_grid(&img, cols, rows, max_per_cell, 0.05, min_dist, &[]);

    // Spacing respected.
    for i in 0..pts.len() {
        for j in (i + 1)..pts.len() {
            let d2 = (pts[i].0 as i32 - pts[j].0 as i32).pow(2)
                + (pts[i].1 as i32 - pts[j].1 as i32).pow(2);
            assert!(
                d2 >= (min_dist * min_dist) as i32,
                "points closer than min_distance"
            );
        }
    }

    // Per-cell budget respected and coverage is broad.
    let mut counts = vec![0u32; (cols * rows) as usize];
    for &(x, y, _) in &pts {
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
    let pts2 =
        good_features_to_track_grid(&img, cols, rows, max_per_cell, 0.05, min_dist, &existing);
    let in_cell0 = pts2
        .iter()
        .filter(|&&(x, y, _)| x * cols / 320 == 0 && y * rows / 240 == 0)
        .count();
    assert_eq!(
        in_cell0, 0,
        "cell filled by existing_points must be skipped"
    );
}
