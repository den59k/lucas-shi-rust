//! Stage costs and explicit rejection experiments, confined to this benchmark.
//! Includes the production source to reach private functions without adding a
//! profiling API to the library. See README.md for setup/timing boundaries.

use std::cmp::Ordering;
use std::hint::black_box;
use std::time::Duration;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

#[allow(dead_code)]
#[path = "support/feature_inputs.rs"]
mod feature_inputs;

// The included detector resolves its normal crate::utils paths here.
#[allow(dead_code)]
mod utils {
    pub mod fast_gradients {
        include!("../src/utils/fast_gradients.rs");
    }
    pub mod box_filter_3x3 {
        include!("../src/utils/box_filter_3x3.rs");
    }
}

// Included unit-test imports are unused in a harness=false benchmark build.
#[allow(dead_code, unused_imports)]
mod detector {
    include!("support/feature_stage_access.rs");
    include!("../src/features.rs");
}

type Points = Vec<(u32, u32, f32)>;
const QUALITY: f32 = 0.05;
const DISTANCE: u32 = 8;

fn sort(points: &mut [(u32, u32, f32)]) {
    points.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
}

// Same dense flag buffer and comparisons as production NMS, with an explicit
// supplied score floor before neighborhood work. Neighbor scores stay intact.
fn screened_nms(mut scores: Points, width: u32, height: u32, threshold: f32) -> Points {
    let mut keep = vec![false; scores.len()];
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let i = (y * width + x) as usize;
            let q = scores[i].2;
            if q <= 0.0 || q < threshold {
                continue;
            }
            let mut is_max = true;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let j = ((y as i32 + dy) as u32 * width + (x as i32 + dx) as u32) as usize;
                    if scores[j].2 > q {
                        is_max = false;
                        break;
                    }
                }
                if !is_max {
                    break;
                }
            }
            keep[i] = is_max;
        }
    }
    scores.retain(|&(x, y, _)| keep[(y * width + x) as usize]);
    scores
}

fn bench_stages(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_stages");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));
    group.nresamples(10_000);

    for case in feature_inputs::cases() {
        let (w, h) = case.image.dimensions();
        let gradients = utils::fast_gradients::compute_gradients(&case.image);
        let scores = detector::tensor_scores(&gradients.0, &gradients.1);
        // Dense per-pixel points for the bench-local screening baseline below.
        let score_points: Points = (0..w * h)
            .zip(&scores)
            .map(|(i, &q)| (i % w, i / w, q))
            .collect();
        let maxima = detector::nms(&scores, w, h);
        let mut floored = detector::floored_nms(&scores, w, h, QUALITY);
        detector::quality(&mut floored, QUALITY);
        let mut qualified = maxima.clone();
        detector::quality(&mut qualified, QUALITY);
        assert_eq!(floored, qualified);
        let mut sorted = qualified.clone();
        sort(&mut sorted);
        let result = detector::spacing(&sorted, DISTANCE, w, h);
        assert_eq!(
            result,
            detector::good_features_to_track(&case.image, QUALITY, DISTANCE)
        );
        assert_eq!(
            result,
            optical_flow_lk::good_features_to_track(&case.image, QUALITY, DISTANCE)
        );
        let max_score = maxima.iter().map(|p| p.2).fold(0.0f32, f32::max);
        let relative = QUALITY * max_score;
        eprintln!(
            "COUNTS {{\"case\":\"{}\",\"pixels\":{},\"positive_scores\":{},\"nms_all\":{},\"nms_positive\":{},\"qualified\":{},\"output\":{},\"max_score\":{},\"relative_threshold\":{}}}",
            case.name,
            scores.len(),
            scores.iter().filter(|&&q| q > 0.0).count(),
            maxima.len(),
            maxima.iter().filter(|p| p.2 > 0.0).count(),
            sorted.len(),
            result.len(),
            max_score,
            relative
        );

        macro_rules! stage {
            ($name:expr, $setup:expr, $run:expr) => {
                group.bench_function(BenchmarkId::new(case.name, $name), |b| {
                    b.iter_batched($setup, $run, BatchSize::PerIteration)
                });
            };
        }
        group.bench_function(BenchmarkId::new(case.name, "whole"), |b| {
            b.iter(|| detector::good_features_to_track(black_box(&case.image), QUALITY, DISTANCE));
        });
        group.bench_function(BenchmarkId::new(case.name, "gradients"), |b| {
            b.iter(|| utils::fast_gradients::compute_gradients(black_box(&case.image)));
        });
        group.bench_function(BenchmarkId::new(case.name, "tensor_scores"), |b| {
            b.iter(|| detector::tensor_scores(black_box(&gradients.0), black_box(&gradients.1)));
        });
        group.bench_function(BenchmarkId::new(case.name, "nms"), |b| {
            b.iter(|| detector::nms(black_box(&scores), w, h));
        });
        stage!("quality", || maxima.clone(), |mut p| {
            detector::quality(black_box(&mut p), QUALITY);
            p
        });
        stage!("sort", || qualified.clone(), |mut p| {
            sort(black_box(p.as_mut_slice()));
            p
        });
        group.bench_function(BenchmarkId::new(case.name, "spacing"), |b| {
            b.iter(|| detector::spacing(black_box(&sorted), DISTANCE, w, h));
        });

        // Exploratory floors in (grey levels / pixel)^2, not suggested defaults.
        for absolute in [0.0f32, 0.01, 0.1, 1.0, 4.0, 16.0] {
            let threshold = relative.max(absolute);
            let mut expected = maxima.clone();
            expected.retain(|p| p.2 > 0.0 && p.2 >= threshold);
            assert_eq!(
                screened_nms(score_points.clone(), w, h, threshold),
                expected
            );
            let before_spacing = expected.len();
            sort(&mut expected);
            let selected = detector::spacing(&expected, DISTANCE, w, h);
            eprintln!(
                "FLOOR {{\"case\":\"{}\",\"absolute\":{},\"effective\":{},\"eligible_pixels\":{},\"qualified\":{},\"output\":{}}}",
                case.name,
                absolute,
                threshold,
                scores
                    .iter()
                    .filter(|&&q| q > 0.0 && q >= threshold)
                    .count(),
                before_spacing,
                selected.len()
            );
        }
        group.bench_function(BenchmarkId::new(case.name, "nms_quality"), |b| {
            b.iter(|| {
                let mut p = detector::nms(black_box(&scores), w, h);
                detector::quality(&mut p, QUALITY);
                p
            });
        });
        // Production path: floor discovery, floored NMS and the quality filter.
        group.bench_function(BenchmarkId::new(case.name, "floored_nms_quality"), |b| {
            b.iter(|| {
                let mut p = detector::floored_nms(black_box(&scores), w, h, QUALITY);
                detector::quality(&mut p, QUALITY);
                p
            });
        });
        // Threshold supplied from the untimed reference NMS above. This isolates
        // screening cost; it does not measure discovering a relative threshold.
        stage!("screened_nms_rel", || score_points.clone(), |p| {
            screened_nms(black_box(p), w, h, relative)
        });
        stage!("screened_nms_abs1", || score_points.clone(), |p| {
            screened_nms(black_box(p), w, h, relative.max(1.0))
        });
    }

    // Revision-independent dense score fixture: unique-ish pseudo-random scores,
    // fixed coordinates and order. It controls workload, not detector quality.
    let (w, h) = (320u32, 240u32);
    let controlled: Points = (0..w * h)
        .map(|i| {
            let q = i.wrapping_mul(2_654_435_761).rotate_left(13) & 65_535;
            (i % w, i / w, q as f32 / 1024.0)
        })
        .collect();
    let controlled_scores: Vec<f32> = controlled.iter().map(|p| p.2).collect();
    let maxima = detector::nms(&controlled_scores, w, h);
    let mut qualified = maxima.clone();
    detector::quality(&mut qualified, QUALITY);
    let mut sorted = qualified.clone();
    sort(&mut sorted);
    let result = detector::spacing(&sorted, DISTANCE, w, h);
    let fingerprint = |p: &Points| {
        p.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, &(x, y, q)| {
            [x, y, q.to_bits()]
                .iter()
                .fold(hash, |a, &v| (a ^ v as u64).wrapping_mul(0x100_0000_01b3))
        })
    };
    eprintln!(
        "CONTROL {{\"pixels\":{},\"nms\":{},\"qualified\":{},\"output\":{},\"input_hash\":{},\"output_hash\":{}}}",
        controlled.len(),
        maxima.len(),
        sorted.len(),
        result.len(),
        fingerprint(&controlled),
        fingerprint(&result)
    );
    group.bench_function("controlled/nms", |b| {
        b.iter(|| detector::nms(black_box(&controlled_scores), w, h))
    });
    group.bench_function("controlled/quality", |b| {
        b.iter_batched(
            || maxima.clone(),
            |mut p| {
                detector::quality(black_box(&mut p), QUALITY);
                p
            },
            BatchSize::PerIteration,
        )
    });
    group.bench_function("controlled/sort", |b| {
        b.iter_batched(
            || qualified.clone(),
            |mut p| {
                sort(black_box(p.as_mut_slice()));
                p
            },
            BatchSize::PerIteration,
        )
    });
    group.bench_function("controlled/spacing", |b| {
        b.iter(|| detector::spacing(black_box(&sorted), DISTANCE, w, h))
    });
    group.finish();
}

criterion_group!(benches, bench_stages);
criterion_main!(benches);
