//! Writes the grayscale inputs as PGM, the crate's corners and tracks as JSON,
//! and native timings, into the output directory given as the first argument.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Instant;

use image::GrayImage;
use optical_flow_lk::{
    DEFAULT_MIN_EIGEN_THRESHOLD, TrackParams, TrackStatus, TrackerContext, good_features_to_track,
};

const QUALITY: f32 = 0.01;
const MIN_DISTANCE: u32 = 5;
const MAX_POINTS: usize = 200;
const LEVELS: usize = 4;
const WINDOW: usize = 21;
const ITERATIONS: usize = 30;
const PARAMS: TrackParams = TrackParams {
    window_size: WINDOW,
    max_iterations: ITERATIONS,
    min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
};
const WARMUP: usize = 10;
const REPS: usize = 100;

fn status_code(status: TrackStatus) -> u8 {
    match status {
        TrackStatus::Tracked => 0,
        TrackStatus::OutOfBounds => 1,
        TrackStatus::Diverged => 2,
        TrackStatus::LowTexture => 3,
        TrackStatus::FbInconsistent => 4,
    }
}

fn write_pgm(path: &Path, img: &GrayImage) {
    let mut bytes = format!("P5\n{} {}\n255\n", img.width(), img.height()).into_bytes();
    bytes.extend_from_slice(img.as_raw());
    std::fs::write(path, bytes).unwrap();
}

fn stats(mut samples: Vec<f64>) -> (f64, f64, f64) {
    samples.sort_by(f64::total_cmp);
    let pick = |q: f64| samples[((samples.len() - 1) as f64 * q).round() as usize];
    (pick(0.5), pick(0.9), samples[0])
}

fn bench(mut f: impl FnMut()) -> (f64, f64, f64) {
    for _ in 0..WARMUP {
        f();
    }
    let samples = (0..REPS)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    stats(samples)
}

fn json_f32(v: f32) -> String {
    if v.is_finite() { format!("{v:?}") } else { "null".into() }
}

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "out".into());
    let out = Path::new(&out);
    std::fs::create_dir_all(out).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
    let prev: GrayImage = image::open(root.join("examples/input1.png")).unwrap().into_luma8();
    let next: GrayImage = image::open(root.join("examples/input2.png")).unwrap().into_luma8();
    write_pgm(&out.join("input1.pgm"), &prev);
    write_pgm(&out.join("input2.pgm"), &next);

    let corners = good_features_to_track(&prev, QUALITY, MIN_DISTANCE);
    let points: Vec<(f32, f32)> = corners
        .iter()
        .take(MAX_POINTS)
        .map(|corner| corner.position())
        .collect();
    let mut ctx = TrackerContext::new();
    ctx.prepare(&prev, &next, LEVELS);
    let tracks = ctx
        .track(&points, None, PARAMS)
        .to_vec();

    let detect = bench(|| {
        std::hint::black_box(good_features_to_track(&prev, QUALITY, MIN_DISTANCE));
    });
    let track = bench(|| {
        ctx.prepare(&prev, &next, LEVELS);
        std::hint::black_box(ctx.track(&points, None, PARAMS));
    });

    let mut json = String::new();
    let _ = write!(
        json,
        "{{\n\"width\":{},\"height\":{},\n\"params\":{{\"quality\":{QUALITY},\"min_distance\":{MIN_DISTANCE},\"levels\":{LEVELS},\"window\":{WINDOW},\"iterations\":{ITERATIONS},\"max_points\":{MAX_POINTS}}},\n",
        prev.width(),
        prev.height()
    );
    json.push_str("\"corners\":[");
    for (i, c) in corners.iter().enumerate() {
        let (x, y) = (c.x, c.y);
        let _ = write!(json, "{}[{x},{y},{}]", if i > 0 { "," } else { "" }, json_f32(c.score));
    }
    json.push_str("],\n\"track_points\":[");
    for (i, &(x, y)) in points.iter().enumerate() {
        let _ = write!(json, "{}[{x},{y}]", if i > 0 { "," } else { "" });
    }
    json.push_str("],\n\"track\":[");
    for (i, r) in tracks.iter().enumerate() {
        let _ = write!(
            json,
            "{}[{},{},{},{}]",
            if i > 0 { "," } else { "" },
            json_f32(r.pos.0),
            json_f32(r.pos.1),
            status_code(r.status),
            json_f32(r.error)
        );
    }
    let _ = write!(
        json,
        "],\n\"native_ms\":{{\"detect\":{{\"median\":{:.4},\"p90\":{:.4},\"min\":{:.4}}},\"track\":{{\"median\":{:.4},\"p90\":{:.4},\"min\":{:.4}}}}},\n\"reps\":{REPS}\n}}\n",
        detect.0, detect.1, detect.2, track.0, track.1, track.2
    );
    std::fs::write(out.join("rust.json"), json).unwrap();

    let tracked = tracks.iter().filter(|r| r.status == TrackStatus::Tracked).count();
    println!(
        "native crate: {} corners, {} points tracked of {}; detect median {:.3} ms, track median {:.3} ms",
        corners.len(),
        tracked,
        points.len(),
        detect.0,
        track.0
    );
}
