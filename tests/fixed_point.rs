//! The fixed-point kernel of the 8-bit tracker against the f32 reference
//! kernel, a golden record of its results on every target, and the
//! forward-backward level option on the bundled example pair.
//!
//! Run with `--nocapture` to see the measured agreement.

use image::{GrayImage, Luma};
use optical_flow_lk::{
    Corner, DEFAULT_FB_THRESHOLD, DEFAULT_MIN_EIGEN_THRESHOLD, TrackKernel, TrackOptions,
    TrackParams, TrackResult, TrackStatus, TrackerContext, good_features_to_track,
};

const LEVELS: usize = 4;

fn params(window_size: usize) -> TrackParams {
    TrackParams {
        window_size,
        max_iterations: 30,
        min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
    }
}

fn example_pair() -> (GrayImage, GrayImage) {
    let decode = |bytes: &[u8]| image::load_from_memory(bytes).unwrap().into_luma8();
    (
        decode(include_bytes!("../examples/input1.png")),
        decode(include_bytes!("../examples/input2.png")),
    )
}

/// The detector's corners on the first example frame, as the OpenCV
/// comparison picks them (quality 0.01, at least 5 px apart).
fn example_points(prev: &GrayImage) -> Vec<(f32, f32)> {
    good_features_to_track(prev, 0.01, 5)
        .iter()
        .map(Corner::position)
        .collect()
}

/// Band-limited noise: hash noise smoothed by three 5x5 means.
fn textured(w: u32, h: u32) -> GrayImage {
    let mut buf: Vec<f32> = (0..w * h)
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
    // Stretch to the full 8-bit range.
    let (lo, hi) = buf
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    GrayImage::from_fn(w, h, |x, y| {
        Luma([((buf[(y * w + x) as usize] - lo) / (hi - lo) * 255.0) as u8])
    })
}

/// `src` resampled at `map(x, y)` for every output pixel; outside reads 0.
fn warp(src: &GrayImage, map: impl Fn(f32, f32) -> (f32, f32)) -> GrayImage {
    let (w, h) = src.dimensions();
    let pixel = |x: i32, y: i32| {
        if x >= 0 && y >= 0 && x < w as i32 && y < h as i32 {
            src.get_pixel(x as u32, y as u32)[0] as f32
        } else {
            0.0
        }
    };
    GrayImage::from_fn(w, h, |x, y| {
        let (sx, sy) = map(x as f32, y as f32);
        let (x0, y0) = (sx.floor(), sy.floor());
        let (fx, fy) = (sx - x0, sy - y0);
        let (x0, y0) = (x0 as i32, y0 as i32);
        let value = pixel(x0, y0) * (1.0 - fx) * (1.0 - fy)
            + pixel(x0 + 1, y0) * fx * (1.0 - fy)
            + pixel(x0, y0 + 1) * (1.0 - fx) * fy
            + pixel(x0 + 1, y0 + 1) * fx * fy;
        Luma([(value + 0.5) as u8])
    })
}

/// Synthetic frame pairs: sub-pixel and larger shifts, a rotation and an
/// affine warp, each as (name, prev, next).
fn synthetic_pairs() -> Vec<(String, GrayImage, GrayImage)> {
    let prev = textured(320, 240);
    let mut pairs = Vec::new();
    for (sx, sy) in [(0.4f32, -0.7f32), (0.25, 0.5), (3.25, 2.5), (12.5, -7.25)] {
        let next = warp(&prev, |x, y| (x - sx, y - sy));
        pairs.push((format!("shift ({sx}, {sy})"), prev.clone(), next));
    }
    let (s, c) = 2f32.to_radians().sin_cos();
    let rotated = warp(&prev, |x, y| {
        let (ox, oy) = (x - 160.0, y - 120.0);
        (c * ox + s * oy + 160.0, -s * ox + c * oy + 120.0)
    });
    pairs.push(("rotation 2 deg".into(), prev.clone(), rotated));
    let affine = warp(&prev, |x, y| {
        let (ox, oy) = (x - 160.0, y - 120.0);
        (
            0.98 * ox - 0.02 * oy + 158.7,
            0.01 * ox + 0.985 * oy + 121.4,
        )
    });
    pairs.push(("affine".into(), prev, affine));
    pairs
}

/// Corners plus a lattice reaching close to the borders.
fn synthetic_points(prev: &GrayImage) -> Vec<(f32, f32)> {
    let mut points: Vec<(f32, f32)> = good_features_to_track(prev, 0.05, 6)
        .iter()
        .take(150)
        .map(Corner::position)
        .collect();
    points.extend((0..60).map(|i| (6.0 + (i % 10) as f32 * 34.1, 5.0 + (i / 10) as f32 * 45.3)));
    points
}

fn distance(a: (f32, f32), b: (f32, f32)) -> f32 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

/// How two result sets agree.
#[derive(Debug)]
struct Agreement {
    points: usize,
    same_status: usize,
    /// Points the reference tracks stably (see [`unstable`]), and how many of
    /// them have the same status in both.
    stable: usize,
    same_status_stable: usize,
    both_tracked: usize,
    median: f32,
    max: f32,
    /// Largest deviation on both-tracked points the reference tracks stably.
    max_stable: f32,
    /// Both-tracked points the reference does not track stably that deviate
    /// by more than 0.1 px: index and deviation.
    excused: Vec<(usize, f32)>,
}

impl Agreement {
    fn of(reference: &[TrackResult], fixed: &[TrackResult], unstable: &[bool]) -> Self {
        let both: Vec<(usize, f32)> = reference
            .iter()
            .zip(fixed)
            .enumerate()
            .filter(|(_, (r, f))| {
                r.status == TrackStatus::Tracked && f.status == TrackStatus::Tracked
            })
            .map(|(i, (r, f))| (i, distance(r.pos, f.pos)))
            .collect();
        let mut distances: Vec<f32> = both.iter().map(|&(_, d)| d).collect();
        distances.sort_by(f32::total_cmp);
        let same = |i: usize| reference[i].status == fixed[i].status;
        Agreement {
            points: reference.len(),
            same_status: (0..reference.len()).filter(|&i| same(i)).count(),
            stable: unstable.iter().filter(|&&u| !u).count(),
            same_status_stable: (0..reference.len())
                .filter(|&i| !unstable[i] && same(i))
                .count(),
            both_tracked: distances.len(),
            median: distances.get(distances.len() / 2).copied().unwrap_or(0.0),
            max: distances.last().copied().unwrap_or(0.0),
            max_stable: both
                .iter()
                .filter(|&&(i, _)| !unstable[i])
                .map(|&(_, d)| d)
                .fold(0.0, f32::max),
            excused: both
                .iter()
                .copied()
                .filter(|&(i, d)| unstable[i] && d > 0.1)
                .collect(),
        }
    }

    /// The accuracy gate of the fixed-point kernel. Statuses agree on 99% of
    /// the points; with the forward-backward check, whose threshold and two
    /// iteration caps add decisions that tip either way on borderline
    /// points, on 98% of all points and 99% of the stably tracked ones. On
    /// points both kernels track, positions agree within 0.02 px in the
    /// median and 0.1 px at most, except on the few points (at most 2%) where
    /// the reference result itself is unstable.
    fn check(&self, case: &str, round_trip: bool) {
        let percent = |part: usize, whole: usize| 100.0 * part as f32 / whole.max(1) as f32;
        println!(
            "{case}: status {}/{} ({:.1}%, {:.1}% of {} stable), both tracked {}, \
             median {:.5} px, max {:.5} px, max on stable points {:.5} px, \
             unstable points beyond 0.1 px {:?}",
            self.same_status,
            self.points,
            percent(self.same_status, self.points),
            percent(self.same_status_stable, self.stable),
            self.stable,
            self.both_tracked,
            self.median,
            self.max,
            self.max_stable,
            self.excused
        );
        let all = if round_trip { 98 } else { 99 };
        assert!(
            self.same_status * 100 >= self.points * all,
            "{case}: statuses agree on fewer than {all}% of points: {self:?}"
        );
        assert!(
            self.same_status_stable * 100 >= self.stable * 99,
            "{case}: statuses agree on fewer than 99% of stable points: {self:?}"
        );
        assert!(self.median <= 0.02, "{case}: median {self:?}");
        assert!(self.max_stable <= 0.1, "{case}: max {self:?}");
        assert!(
            self.excused.len() * 50 <= self.both_tracked,
            "{case}: {self:?}"
        );
        assert!(self.both_tracked > 0, "{case}: nothing tracked");
    }
}

/// Points whose reference result moves by more than 0.1 px, or changes
/// status, when the input point moves by 1e-4 px in one of 8 directions:
/// iterations that end near the boundary between two basins of attraction,
/// where any change in arithmetic can land them in either.
fn unstable(
    ctx: &mut TrackerContext,
    points: &[(f32, f32)],
    reference: &[TrackResult],
    track: impl Fn(&mut TrackerContext, &[(f32, f32)]) -> Vec<TrackResult>,
) -> Vec<bool> {
    let mut unstable = vec![false; points.len()];
    for k in 0..8 {
        let angle = k as f32 * std::f32::consts::FRAC_PI_4;
        let (dx, dy) = (1e-4 * angle.cos(), 1e-4 * angle.sin());
        let moved: Vec<(f32, f32)> = points.iter().map(|&(x, y)| (x + dx, y + dy)).collect();
        for (i, (m, r)) in track(ctx, &moved).iter().zip(reference).enumerate() {
            if m.status != r.status || distance(m.pos, r.pos) > 0.1 {
                unstable[i] = true;
            }
        }
    }
    unstable
}

/// `track` and `track_fb` with the reference and the fixed-point kernel.
fn compare_kernels(case: &str, prev: &GrayImage, next: &GrayImage, points: &[(f32, f32)]) {
    let mut ctx = TrackerContext::new();
    ctx.prepare(prev, next, LEVELS);
    let reference = TrackOptions::new().with_kernel(TrackKernel::Reference);
    let fixed = TrackOptions::new().with_kernel(TrackKernel::FixedPoint);
    for window in [13, 21] {
        let p = params(window);
        let forward = |ctx: &mut TrackerContext, points: &[(f32, f32)], options| {
            ctx.track_with(points, None, p, options).to_vec()
        };
        let round_trip = |ctx: &mut TrackerContext, points: &[(f32, f32)], options| {
            ctx.track_fb_with(points, None, p, DEFAULT_FB_THRESHOLD, options)
                .to_vec()
        };
        let r = forward(&mut ctx, points, reference);
        let f = forward(&mut ctx, points, fixed);
        let shaky = unstable(&mut ctx, points, &r, |ctx, moved| {
            forward(ctx, moved, reference)
        });
        Agreement::of(&r, &f, &shaky).check(&format!("{case}, w{window}, track"), false);
        let r = round_trip(&mut ctx, points, reference);
        let f = round_trip(&mut ctx, points, fixed);
        let shaky = unstable(&mut ctx, points, &r, |ctx, moved| {
            round_trip(ctx, moved, reference)
        });
        Agreement::of(&r, &f, &shaky).check(&format!("{case}, w{window}, track_fb"), true);
    }
}

const GOLDEN: &str = include_str!("fixtures/fixed_point_example.txt");

fn status_letter(status: TrackStatus) -> char {
    match status {
        TrackStatus::Tracked => 'T',
        TrackStatus::OutOfBounds => 'O',
        TrackStatus::Diverged => 'D',
        TrackStatus::LowTexture => 'L',
        TrackStatus::FbInconsistent => 'F',
    }
}

/// Every fixed-point result on the example pair for the first 200 corners:
/// `track`, `track_fb` and `track_fb` over the finest level, windows 13 and
/// 21, as exact bits.
fn golden_lines() -> Vec<String> {
    let (prev, next) = example_pair();
    let points: Vec<(f32, f32)> = example_points(&prev).into_iter().take(200).collect();
    let mut ctx = TrackerContext::new();
    ctx.prepare(&prev, &next, LEVELS);
    let fixed = TrackOptions::new().with_kernel(TrackKernel::FixedPoint);
    let mut lines = Vec::new();
    for window in [13, 21] {
        let p = params(window);
        let cases = [
            ("track", ctx.track_with(&points, None, p, fixed).to_vec()),
            (
                "track_fb",
                ctx.track_fb_with(&points, None, p, DEFAULT_FB_THRESHOLD, fixed)
                    .to_vec(),
            ),
            (
                "track_fb_1",
                ctx.track_fb_with(
                    &points,
                    None,
                    p,
                    DEFAULT_FB_THRESHOLD,
                    fixed.with_fb_levels(1),
                )
                .to_vec(),
            ),
        ];
        for (name, results) in cases {
            for (i, r) in results.iter().enumerate() {
                lines.push(format!(
                    "w{window} {name} {i} {:08x} {:08x} {} {:08x}",
                    r.pos.0.to_bits(),
                    r.pos.1.to_bits(),
                    status_letter(r.status),
                    r.error.to_bits()
                ));
            }
        }
    }
    lines
}

/// The fixed-point kernel computes the same integers on every instruction
/// set, so its results are a property of the input alone: this record of
/// them must match bit for bit natively (scalar, SSE2, AVX2, NEON) and on
/// wasm32 with or without `simd128`. Regenerate with
/// `GOLDEN_BLESS=1 cargo test --test fixed_point` after an intended change.
#[test]
fn fixed_point_results_match_the_golden_record() {
    let lines = golden_lines();
    if std::env::var_os("GOLDEN_BLESS").is_some() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/fixed_point_example.txt"
        );
        let mut text = String::from(
            "# Fixed-point tracking of the first 200 corners of examples/input1.png into\n\
             # input2.png, 4 levels, 30 iterations: window, case, point, then the bits\n\
             # of x and y, the status and the bits of the error. See tests/fixed_point.rs.\n",
        );
        for line in &lines {
            text.push_str(line);
            text.push('\n');
        }
        std::fs::write(path, text).expect("write the golden record");
        return;
    }
    let expected: Vec<&str> = GOLDEN.lines().filter(|l| !l.starts_with('#')).collect();
    let mismatches: Vec<String> = lines
        .iter()
        .zip(&expected)
        .filter(|(a, e)| a != *e)
        .map(|(a, e)| format!("  expected {e}\n  actual   {a}"))
        .collect();
    assert!(
        mismatches.is_empty() && lines.len() == expected.len(),
        "{} of {} results differ from the golden record ({} expected):\n{}",
        mismatches.len(),
        lines.len(),
        expected.len(),
        mismatches[..mismatches.len().min(20)].join("\n")
    );
}

/// How many points the backward pass over the finest `fb_levels` levels
/// judges differently from the backward pass over all levels, on the
/// example pair (median motion about 19 px): `(accepted, rejected)`, the
/// points it accepts that the full check rejects and the reverse.
fn fb_level_changes(window: usize, kernel: TrackKernel) -> Vec<(usize, usize, usize)> {
    let (prev, next) = example_pair();
    let points = example_points(&prev);
    let mut ctx = TrackerContext::new();
    ctx.prepare(&prev, &next, LEVELS);
    let p = params(window);
    let options = TrackOptions::new().with_kernel(kernel);
    let all = ctx
        .track_fb_with(&points, None, p, DEFAULT_FB_THRESHOLD, options)
        .to_vec();
    let consistent = |r: &&TrackResult| r.status == TrackStatus::Tracked;
    println!(
        "w{window} {kernel:?}: {} points, {} tracked forward, {} pass the full check",
        points.len(),
        ctx.track_with(&points, None, p, options)
            .iter()
            .filter(consistent)
            .count(),
        all.iter().filter(consistent).count()
    );
    (1..=LEVELS)
        .map(|levels| {
            let some = ctx.track_fb_with(
                &points,
                None,
                p,
                DEFAULT_FB_THRESHOLD,
                options.with_fb_levels(levels),
            );
            let flips = |from: TrackStatus, to: TrackStatus| {
                all.iter()
                    .zip(some)
                    .filter(|(a, s)| a.status == from && s.status == to)
                    .count()
            };
            let accepted = flips(TrackStatus::FbInconsistent, TrackStatus::Tracked);
            let rejected = flips(TrackStatus::Tracked, TrackStatus::FbInconsistent);
            let changed = all
                .iter()
                .zip(some)
                .filter(|(a, s)| a.status != s.status)
                .count();
            assert_eq!(changed, accepted + rejected, "only the check can change");
            println!("  fb_levels {levels}: {accepted} more accepted, {rejected} more rejected");
            (levels, accepted, rejected)
        })
        .collect()
}

#[test]
fn fb_levels_trade_on_the_example_pair() {
    for window in [13, 21] {
        for kernel in [TrackKernel::FixedPoint, TrackKernel::Reference] {
            let changes = fb_level_changes(window, kernel);
            // All levels are the full check; one level still agrees with it
            // on all but a few percent of the points.
            assert_eq!(changes[LEVELS - 1], (LEVELS, 0, 0));
            let (_, accepted, rejected) = changes[0];
            assert!(accepted + rejected <= 20, "w{window}: {changes:?}");
        }
    }
}

#[test]
fn fixed_point_agrees_with_the_reference_on_the_example_pair() {
    let (prev, next) = example_pair();
    let points = example_points(&prev);
    assert!(points.len() > 300);
    compare_kernels("example", &prev, &next, &points);
}

#[test]
fn fixed_point_agrees_with_the_reference_on_synthetic_motion() {
    for (name, prev, next) in synthetic_pairs() {
        compare_kernels(&name, &prev, &next, &synthetic_points(&prev));
    }
}
