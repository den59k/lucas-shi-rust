//! Bit-exact regression fixtures for detection and tracking.
//!
//! Every case is reduced to its output count and a 64-bit FNV-1a hash of the
//! exact output bits, stored one line per case in `fixtures/regression.txt`.
//! Outputs must be identical on every target (native, AVX2, WASM). After an
//! intended change in results, regenerate the file with
//! `REGRESSION_BLESS=1 cargo test --test regression` and review the diff.

use image::{GrayImage, Luma};
use optical_flow_lk::{
    DEFAULT_FB_THRESHOLD, DEFAULT_MIN_EIGEN_THRESHOLD, FloatDetectorContext, FloatImageView,
    FloatTrackerContext, ScoreThreshold, TrackResult, TrackStatus, build_pyramid,
    calc_optical_flow_ex, calc_optical_flow_fb, good_features_to_track,
    good_features_to_track_grid,
};

#[path = "../benches/support/feature_inputs.rs"]
#[allow(dead_code)]
mod feature_inputs;

const FIXTURE: &str = include_str!("fixtures/regression.txt");

/// 64-bit FNV-1a over the exact bits of a case's output.
struct Hash(u64);

impl Hash {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 ^ byte as u64).wrapping_mul(0x0100_0000_01b3);
        }
    }

    fn u32(&mut self, value: u32) {
        self.bytes(&value.to_le_bytes());
    }
}

fn corners_line(id: String, corners: &[(u32, u32, f32)]) -> String {
    let mut hash = Hash::new();
    for &(x, y, score) in corners {
        hash.u32(x);
        hash.u32(y);
        hash.u32(score.to_bits());
    }
    format!("{id} n={} {:016x}", corners.len(), hash.0)
}

fn tracks_line(id: String, results: &[TrackResult]) -> String {
    let mut hash = Hash::new();
    for result in results {
        hash.u32(result.pos.0.to_bits());
        hash.u32(result.pos.1.to_bits());
        hash.u32(result.error.to_bits());
        hash.u32(match result.status {
            TrackStatus::Tracked => 0,
            TrackStatus::OutOfBounds => 1,
            TrackStatus::Diverged => 2,
            TrackStatus::LowTexture => 3,
            TrackStatus::FbInconsistent => 4,
        });
    }
    let tracked = results
        .iter()
        .filter(|r| r.status == TrackStatus::Tracked)
        .count();
    format!("{id} n={} tracked={tracked} {:016x}", results.len(), hash.0)
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

/// Detector inputs: the benchmark images, small and odd sizes, low and
/// saturated contrast, and a global maximum on the border.
fn detection_images() -> Vec<(String, GrayImage)> {
    let mut images: Vec<(String, GrayImage)> = feature_inputs::cases()
        .into_iter()
        .map(|case| (case.name.to_string(), case.image))
        .collect();
    let example2 = image::load_from_memory(include_bytes!("../examples/input2.png"))
        .expect("bundled example image must decode")
        .into_luma8();
    images.push(("example2".into(), example2));
    for (w, h) in [
        (3, 3),
        (4, 5),
        (5, 4),
        (7, 3),
        (3, 9),
        (17, 19),
        (33, 1),
        (1, 33),
        (2, 2),
        (65, 49),
        (127, 61),
    ] {
        images.push((format!("noise_{w}x{h}"), noise(w, h, w * 31 + h)));
    }
    for contrast in [1u8, 2, 7, 64, 255] {
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
    images.push((
        "border_max".into(),
        GrayImage::from_fn(40, 30, |x, y| {
            Luma([if x < 2 && y < 2 {
                255
            } else if (x / 5 + y / 5) % 2 == 0 {
                20
            } else {
                0
            }])
        }),
    ));
    images
}

fn detection_lines() -> Vec<String> {
    let mut lines = Vec::new();
    for (name, image) in detection_images() {
        let (w, h) = image.dimensions();
        for quality in [0.0f32, 0.01, 0.05, 0.3, 1.0] {
            for distance in [0u32, 1, 6, 8, 100] {
                let corners = good_features_to_track(&image, quality, distance);
                lines.push(corners_line(
                    format!("detect {name} q={quality} d={distance}"),
                    &corners,
                ));
            }
        }
        let existing: Vec<(f32, f32)> = (0..6)
            .map(|i| ((i * 37 % w) as f32 + 0.5, (i * 23 % h) as f32))
            .collect();
        for (cols, rows, budget) in [(1, 1, 10), (6, 4, 5), (3, 2, 1)] {
            for points in [&[][..], &existing[..]] {
                let corners =
                    good_features_to_track_grid(&image, cols, rows, budget, 0.01, 6, points);
                lines.push(corners_line(
                    format!("grid {name} {cols}x{rows} b={budget} e={}", points.len()),
                    &corners,
                ));
            }
        }
        // The 8-bit API does not validate quality_level; pin what it does
        // outside 0..=1.
        for quality in [f32::NAN, -1.0, 2.0, f32::INFINITY] {
            let corners = good_features_to_track(&image, quality, 6);
            lines.push(corners_line(
                format!("detect {name} q={quality} d=6"),
                &corners,
            ));
            let corners = good_features_to_track_grid(&image, 6, 4, 5, quality, 6, &existing);
            lines.push(corners_line(
                format!("grid {name} 6x4 b=5 e=6 q={quality}"),
                &corners,
            ));
        }
    }
    lines
}

/// Bilinear shift of `src` by `(sx, sy)`, truncated to u8; outside reads as 0.
fn shift(src: &GrayImage, sx: f32, sy: f32) -> GrayImage {
    let (w, h) = src.dimensions();
    let pixel = |x: i32, y: i32| {
        if x >= 0 && y >= 0 && x < w as i32 && y < h as i32 {
            src.get_pixel(x as u32, y as u32)[0] as f32
        } else {
            0.0
        }
    };
    GrayImage::from_fn(w, h, |x, y| {
        let (fx, fy) = (x as f32 - sx, y as f32 - sy);
        let (x0, y0) = (fx.floor(), fy.floor());
        let (dx, dy) = (fx - x0, fy - y0);
        let (x0, y0) = (x0 as i32, y0 as i32);
        let value = pixel(x0, y0) * (1.0 - dx) * (1.0 - dy)
            + pixel(x0 + 1, y0) * dx * (1.0 - dy)
            + pixel(x0, y0 + 1) * (1.0 - dx) * dy
            + pixel(x0 + 1, y0 + 1) * dx * dy;
        Luma([value as u8])
    })
}

fn tracking_lines() -> Vec<String> {
    let prev = feature_inputs::texture(320, 240);
    // A lattice reaching into the borders plus the detector's own corners.
    let mut points: Vec<(f32, f32)> = (0..48)
        .map(|i| (4.0 + (i % 8) as f32 * 44.5, 3.0 + (i / 8) as f32 * 46.7))
        .collect();
    points.extend(
        good_features_to_track(&prev, 0.05, 8)
            .iter()
            .take(60)
            .map(|&(x, y, _)| (x as f32, y as f32)),
    );

    let mut lines = Vec::new();
    for (sx, sy) in [(0.4f32, -0.3f32), (3.0, 2.0), (12.5, -7.25)] {
        let next = shift(&prev, sx, sy);
        let predicted: Vec<(f32, f32)> = points.iter().map(|&(x, y)| (x + sx, y + sy)).collect();
        for (levels, window, iterations) in [(1usize, 7usize, 10usize), (3, 15, 15), (4, 21, 30)] {
            let prev_pyramid = build_pyramid(&prev, levels);
            let next_pyramid = build_pyramid(&next, levels);
            let id =
                |kind: &str| format!("{kind} s=({sx},{sy}) l={levels} w={window} i={iterations}");
            for (guess, label) in [(None, "track"), (Some(predicted.as_slice()), "track_guess")] {
                let results = calc_optical_flow_ex(
                    &prev_pyramid,
                    &next_pyramid,
                    &points,
                    guess,
                    window,
                    iterations,
                    DEFAULT_MIN_EIGEN_THRESHOLD,
                );
                lines.push(tracks_line(id(label), &results));
            }
            let results = calc_optical_flow_fb(
                &prev_pyramid,
                &next_pyramid,
                &points,
                None,
                window,
                iterations,
                DEFAULT_MIN_EIGEN_THRESHOLD,
                DEFAULT_FB_THRESHOLD,
            );
            lines.push(tracks_line(id("track_fb"), &results));
        }
    }
    lines
}

/// A real-valued thermal-like frame in °C: smooth gradients, two warm bodies
/// and fine texture of a few hundredths of a kelvin, shifted by `(sx, sy)`.
fn thermal_frame(w: u32, h: u32, (sx, sy): (f32, f32)) -> Vec<f32> {
    (0..w * h)
        .map(|i| {
            let x = (i % w) as f32 - sx;
            let y = (i / w) as f32 - sy;
            let blob = |cx: f32, cy: f32, r: f32, t: f32| {
                t / (1.0 + ((x - cx).powi(2) + (y - cy).powi(2)) / (r * r))
            };
            let texture =
                (x * 0.37).sin() * (y * 0.29).cos() * 0.04 + (x * 0.11 + y * 0.07).sin() * 0.02;
            22.5 + 0.01 * x + blob(80.0, 60.0, 14.0, 6.0) + blob(200.0, 150.0, 25.0, 1.5) + texture
        })
        .collect()
}

/// f32 detector and tracker on real-valued frames, which no 8-bit fixture
/// covers.
fn float_lines() -> Vec<String> {
    let (w, h) = (256u32, 192u32);
    let prev = thermal_frame(w, h, (0.0, 0.0));
    let prev_view = FloatImageView::new(&prev, w, h).unwrap();
    let mut lines = Vec::new();
    let mut detector = FloatDetectorContext::new();
    let mut points = Vec::new();
    for threshold in [
        ScoreThreshold::absolute(0.0),
        ScoreThreshold::absolute(1e-5),
        ScoreThreshold::absolute(1e-4).with_relative(0.05),
    ] {
        for distance in [0u32, 6] {
            let corners = detector.detect(prev_view, threshold, distance);
            lines.push(corners_line(
                format!(
                    "float_detect a={} r={} d={distance}",
                    threshold.absolute_part(),
                    threshold.relative_part()
                ),
                corners,
            ));
            if distance == 6 && threshold == ScoreThreshold::absolute(1e-5) {
                let corners = corners.iter().take(80);
                points = corners.map(|&(x, y, _)| (x as f32, y as f32)).collect();
            }
        }
    }
    let existing = [(80.0, 60.0), (10.5, 180.0)];
    let threshold = ScoreThreshold::absolute(1e-5);
    let corners = detector.detect_grid(prev_view, 4, 3, 6, threshold, 8, &existing);
    lines.push(corners_line(
        "float_detect grid 4x3 b=6 e=2".into(),
        corners,
    ));

    let mut tracker = FloatTrackerContext::new();
    for shift in [(0.35f32, -0.6f32), (4.0, 2.5)] {
        let next = thermal_frame(w, h, shift);
        let next_view = FloatImageView::new(&next, w, h).unwrap();
        for (levels, window, iterations) in [(1usize, 9usize, 10usize), (3, 15, 15)] {
            tracker.prepare(prev_view, next_view, levels);
            let id = |kind: &str| {
                format!(
                    "{kind} s=({},{}) l={levels} w={window} i={iterations}",
                    shift.0, shift.1
                )
            };
            let results = tracker.track(&points, None, window, iterations, 1e-6);
            lines.push(tracks_line(id("float_track"), results));
            let results = tracker.track_fb(
                &points,
                None,
                window,
                iterations,
                1e-6,
                DEFAULT_FB_THRESHOLD,
            );
            lines.push(tracks_line(id("float_track_fb"), results));
        }
    }
    lines
}

fn check(section: &str, actual: Vec<String>) {
    let expected: Vec<&str> = FIXTURE
        .lines()
        .filter(|line| {
            line.split(' ')
                .next()
                .is_some_and(|kind| kind_section(kind) == section)
        })
        .collect();
    let mismatches: Vec<String> = actual
        .iter()
        .zip(&expected)
        .filter(|(a, e)| a != *e)
        .map(|(a, e)| format!("  expected {e}\n  actual   {a}"))
        .collect();
    assert!(
        mismatches.is_empty() && actual.len() == expected.len(),
        "{} of {} {section} cases differ ({} expected); \
         run with REGRESSION_BLESS=1 after an intended change:\n{}",
        mismatches.len(),
        actual.len(),
        expected.len(),
        mismatches[..mismatches.len().min(20)].join("\n")
    );
}

fn kind_section(kind: &str) -> &'static str {
    if kind.starts_with("float") {
        "float"
    } else if kind.starts_with("track") {
        "tracking"
    } else {
        "detection"
    }
}

#[test]
fn outputs_match_fixtures() {
    let detection = detection_lines();
    let tracking = tracking_lines();
    let float = float_lines();
    if std::env::var_os("REGRESSION_BLESS").is_some() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/regression.txt");
        let mut text = String::new();
        for line in detection.iter().chain(&tracking).chain(&float) {
            text.push_str(line);
            text.push('\n');
        }
        std::fs::write(path, text).expect("write regression fixtures");
        return;
    }
    check("detection", detection);
    check("tracking", tracking);
    check("float", float);
}
