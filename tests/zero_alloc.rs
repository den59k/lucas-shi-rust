//! Asserts that the detection and tracking hot paths perform no heap
//! allocation, both after a warm-up and on the first call into a context
//! created with explicit capacities.
//!
//! Uses a counting global allocator (scoped to this test binary only, so it
//! does not affect the library or other tests).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use image::{GrayImage, Luma};
use optical_flow_lk::{
    DEFAULT_FB_THRESHOLD, DEFAULT_MIN_EIGEN_THRESHOLD, DetectorCapacity, DetectorContext,
    FloatDetectorContext, FloatImageView, FloatTrackerContext, Grid, ScoreThreshold, TrackKernel,
    TrackOptions, TrackParams, TrackerCapacity, TrackerContext,
};

struct CountingAllocator;

// Per thread, so tests running in parallel do not count each other's
// allocations. Const-initialized `Cell`s need no allocation to access.
thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

fn record() {
    let counting = COUNTING.try_with(Cell::get).unwrap_or(false);
    if counting {
        let _ = ALLOCS.try_with(|allocs| allocs.set(allocs.get() + 1));
    }
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record();
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Heap allocations `f` performs on the calling thread.
fn allocations<R>(f: impl FnOnce() -> R) -> (usize, R) {
    ALLOCS.with(|allocs| allocs.set(0));
    COUNTING.with(|counting| counting.set(true));
    let result = f();
    COUNTING.with(|counting| counting.set(false));
    (ALLOCS.with(Cell::get), result)
}

fn textured(w: u32, h: u32, seed: u32) -> GrayImage {
    let mut img = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let mut s = (y * w + x)
                .wrapping_mul(2654435761)
                .wrapping_add(seed.wrapping_mul(40503))
                ^ 0x9e3779b9;
            s ^= s >> 15;
            s = s.wrapping_mul(0x85ebca6b);
            img.put_pixel(x, y, Luma([(s & 0xff) as u8]));
        }
    }
    img
}

#[test]
fn steady_state_track_is_allocation_free() {
    let prev = textured(640, 480, 1);
    let next = textured(640, 480, 2);
    let points: Vec<(f32, f32)> = (0..150)
        .map(|i| (40.0 + (i % 15) as f32 * 38.0, 40.0 + (i / 15) as f32 * 40.0))
        .collect();
    let predicted: Vec<(f32, f32)> = points.iter().map(|&(x, y)| (x + 1.0, y + 0.5)).collect();

    let mut ctx = TrackerContext::new();

    // Warm-up: lets every buffer grow to its steady-state capacity.
    for _ in 0..3 {
        ctx.prepare(&prev, &next, 4);
        ctx.track(
            &points,
            Some(&predicted),
            TrackParams {
                window_size: 21,
                max_iterations: 30,
                min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
            },
        );
    }

    // Measured steady-state step.
    let (allocs, n) = allocations(|| {
        ctx.prepare(&prev, &next, 4);
        ctx.track(
            &points,
            Some(&predicted),
            TrackParams {
                window_size: 21,
                max_iterations: 30,
                min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
            },
        )
        .len()
    });
    assert_eq!(n, points.len());
    assert_eq!(
        allocs, 0,
        "steady-state prepare+track allocated {allocs} times"
    );
}

#[test]
fn steady_state_detection_is_allocation_free() {
    let img = textured(640, 480, 3);
    let existing: Vec<(f32, f32)> = (0..24)
        .map(|i| (50.0 + (i % 6) as f32 * 100.0, 60.0 + (i / 6) as f32 * 110.0))
        .collect();

    let mut ctx = DetectorContext::new();

    // Warm-up: lets every buffer grow to its steady-state capacity.
    for _ in 0..3 {
        ctx.detect(&img, 0.01, 8);
        ctx.detect_grid(
            &img,
            Grid {
                cols: 6,
                rows: 4,
                max_per_cell: 5,
            },
            0.01,
            8,
            &existing,
        );
    }

    let (allocs, (plain, grid)) = allocations(|| {
        let plain = ctx.detect(&img, 0.01, 8).len();
        let grid = ctx
            .detect_grid(
                &img,
                Grid {
                    cols: 6,
                    rows: 4,
                    max_per_cell: 5,
                },
                0.01,
                8,
                &existing,
            )
            .len();
        (plain, grid)
    });
    assert!(
        plain > 0 && grid > 0,
        "detector found nothing on a textured image"
    );
    assert_eq!(
        allocs, 0,
        "steady-state detect+detect_grid allocated {allocs} times"
    );
}

/// Lattice of `n` points spread over a 640x480 frame.
fn lattice(n: usize) -> Vec<(f32, f32)> {
    (0..n)
        .map(|i| (24.0 + (i % 20) as f32 * 30.0, 24.0 + (i / 20) as f32 * 28.0))
        .collect()
}

#[test]
fn reserved_detection_does_not_allocate_on_any_content() {
    let images = [
        textured(640, 480, 4),
        textured(320, 240, 5),
        textured(17, 9, 6),
        GrayImage::from_pixel(640, 480, Luma([90])),
    ];
    let existing = lattice(50);

    // Zero spacing reserves for every interior pixel being a result.
    let mut ctx = DetectorContext::with_capacity(DetectorCapacity {
        width: 640,
        height: 480,
        min_distance: 0,
        grid_cells: 24,
        existing_points: existing.len(),
    });
    for image in &images {
        let (allocs, counts) = allocations(|| {
            [
                ctx.detect(image, 0.0, 0).len(),
                ctx.detect(image, 0.01, 8).len(),
                ctx.detect_grid(
                    image,
                    Grid {
                        cols: 6,
                        rows: 4,
                        max_per_cell: 5,
                    },
                    0.0,
                    0,
                    &existing,
                )
                .len(),
                ctx.detect_grid(
                    image,
                    Grid {
                        cols: 4,
                        rows: 3,
                        max_per_cell: 50,
                    },
                    0.01,
                    6,
                    &existing[..10],
                )
                .len(),
            ]
        });
        assert_eq!(
            allocs,
            0,
            "{:?} allocated {allocs} times",
            image.dimensions()
        );
        if image.width() == 640 && image.as_raw()[0] != 90 {
            assert!(counts[0] > 10_000, "dense case found only {counts:?}");
        }
    }

    // With spacing, results are bounded by how many points fit at that
    // spacing, so the reservation is smaller.
    let mut ctx = DetectorContext::with_capacity(DetectorCapacity {
        width: 640,
        height: 480,
        min_distance: 8,
        grid_cells: 24,
        existing_points: existing.len(),
    });
    let (allocs, _) = allocations(|| {
        for image in &images {
            ctx.detect(image, 0.0, 8);
            ctx.detect(image, 0.0, 30);
            ctx.detect_grid(
                image,
                Grid {
                    cols: 6,
                    rows: 4,
                    max_per_cell: 1000,
                },
                0.0,
                8,
                &existing,
            );
        }
    });
    assert_eq!(allocs, 0, "spaced detection allocated {allocs} times");
}

#[test]
fn reserved_tracking_does_not_allocate_with_varying_points() {
    let prev = textured(640, 480, 7);
    let next = textured(640, 480, 8);
    let points = lattice(300);
    let predicted: Vec<(f32, f32)> = points.iter().map(|&(x, y)| (x + 0.5, y - 0.25)).collect();

    let mut ctx = TrackerContext::with_capacity(TrackerCapacity {
        width: 640,
        height: 480,
        levels: 4,
        window_size: 21,
        points: points.len(),
    });
    let (allocs, tracked) = allocations(|| {
        ctx.prepare(&prev, &next, 4);
        let mut tracked = ctx
            .track(
                &points,
                Some(&predicted),
                TrackParams {
                    window_size: 21,
                    max_iterations: 30,
                    min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
                },
            )
            .len();
        tracked += ctx
            .track(
                &points[..40],
                None,
                TrackParams {
                    window_size: 15,
                    max_iterations: 10,
                    min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
                },
            )
            .len();
        tracked += ctx
            .track_fb(
                &points[..150],
                Some(&predicted[..150]),
                TrackParams {
                    window_size: 21,
                    max_iterations: 30,
                    min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
                },
                DEFAULT_FB_THRESHOLD,
            )
            .len();
        ctx.prepare(&next, &prev, 4);
        tracked += ctx
            .track_fb(
                &points,
                None,
                TrackParams {
                    window_size: 7,
                    max_iterations: 30,
                    min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
                },
                DEFAULT_FB_THRESHOLD,
            )
            .len();
        tracked
    });
    assert_eq!(tracked, 300 + 40 + 150 + 300);
    assert_eq!(allocs, 0, "reserved tracking allocated {allocs} times");
}

#[test]
fn steady_state_track_fb_is_allocation_free() {
    let prev = textured(320, 240, 9);
    let next = textured(320, 240, 10);
    let points = lattice(80)
        .into_iter()
        .map(|(x, y)| (x * 0.5, y * 0.5))
        .collect::<Vec<_>>();
    let mut ctx = TrackerContext::new();
    for _ in 0..2 {
        ctx.prepare(&prev, &next, 3);
        ctx.track_fb(
            &points,
            None,
            TrackParams {
                window_size: 15,
                max_iterations: 15,
                min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
            },
            DEFAULT_FB_THRESHOLD,
        );
    }
    let (allocs, n) = allocations(|| {
        ctx.prepare(&prev, &next, 3);
        ctx.track_fb(
            &points,
            None,
            TrackParams {
                window_size: 15,
                max_iterations: 15,
                min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
            },
            DEFAULT_FB_THRESHOLD,
        )
        .len()
    });
    assert_eq!(n, points.len());
    assert_eq!(allocs, 0, "steady-state track_fb allocated {allocs} times");
}

const ADVANCE_PARAMS: TrackParams = TrackParams {
    window_size: 21,
    max_iterations: 30,
    min_eigen_threshold: DEFAULT_MIN_EIGEN_THRESHOLD,
};

/// Option sets every per-frame call cycles through: backward passes over
/// fewer levels (so that frames become previous frames with only some
/// gradient levels computed), a lower start level and both kernels.
const OPTION_SETS: [TrackOptions; 5] = [
    TrackOptions::new(),
    TrackOptions::new().with_fb_levels(1),
    TrackOptions::new().with_start_level(1).with_fb_levels(2),
    TrackOptions::new().with_kernel(TrackKernel::Reference),
    TrackOptions::new().with_kernel(TrackKernel::FixedPoint),
];

/// One frame step of a real-time front-end: `advance`, `track_fb_with` and a
/// `track_with` that reuses the cached gradients. Returns the result count.
fn frame_step(
    ctx: &mut TrackerContext,
    frame: &GrayImage,
    points: &[(f32, f32)],
    options: TrackOptions,
) -> usize {
    ctx.advance(frame, 4);
    let n = ctx
        .track_fb_with(points, None, ADVANCE_PARAMS, DEFAULT_FB_THRESHOLD, options)
        .len();
    n + ctx.track_with(points, None, ADVANCE_PARAMS, options).len()
}

#[test]
fn steady_state_options_and_kernels_are_allocation_free() {
    let frames = [25, 26, 27].map(|seed| textured(640, 480, seed));
    let points = lattice(200);
    let mut ctx = TrackerContext::new();
    // Warm-up: every option set and kernel once.
    for (frame, options) in frames.iter().cycle().zip(OPTION_SETS) {
        frame_step(&mut ctx, frame, &points, options);
    }
    let (allocs, tracked) = allocations(|| {
        let steps = frames
            .iter()
            .cycle()
            .zip(OPTION_SETS.iter().cycle())
            .take(10);
        steps
            .map(|(frame, &options)| frame_step(&mut ctx, frame, &points, options))
            .sum::<usize>()
    });
    assert_eq!(tracked, 10 * 2 * points.len());
    assert_eq!(
        allocs, 0,
        "steady-state option sets allocated {allocs} times"
    );
}

#[test]
fn steady_state_advance_is_allocation_free() {
    let frames = [16, 17, 18].map(|seed| textured(640, 480, seed));
    let points = lattice(200);
    let mut ctx = TrackerContext::new();
    // Warm-up: the first call builds both pyramids, the second swaps them.
    for frame in &frames[..2] {
        ctx.advance(frame, 4);
        ctx.track_fb(&points, None, ADVANCE_PARAMS, DEFAULT_FB_THRESHOLD);
    }
    let (allocs, tracked) = allocations(|| {
        let mut tracked = 0;
        for frame in frames.iter().cycle().take(6) {
            ctx.advance(frame, 4);
            tracked += ctx
                .track_fb(&points, None, ADVANCE_PARAMS, DEFAULT_FB_THRESHOLD)
                .len();
        }
        tracked
    });
    assert_eq!(tracked, 6 * points.len());
    assert_eq!(
        allocs, 0,
        "steady-state advance+track_fb allocated {allocs} times"
    );
}

#[test]
fn reserved_advance_does_not_allocate_from_the_first_frame() {
    let frames = [19, 20, 21].map(|seed| textured(640, 480, seed));
    let points = lattice(300);
    let capacity = TrackerCapacity {
        width: 640,
        height: 480,
        levels: 4,
        window_size: 21,
        points: points.len(),
    };
    let mut ctx = TrackerContext::with_capacity(capacity);
    let (allocs, tracked) = allocations(|| {
        // Every option set and both kernels from the first frame on.
        let steps = frames.iter().cycle().zip(OPTION_SETS.iter().cycle());
        steps
            .take(5)
            .map(|(frame, &options)| frame_step(&mut ctx, frame, &points, options))
            .sum::<usize>()
    });
    assert_eq!(tracked, 5 * 2 * points.len());
    assert_eq!(allocs, 0, "reserved advance allocated {allocs} times");

    let celsius: Vec<Vec<f32>> = [22, 23, 24]
        .into_iter()
        .map(|seed| {
            textured(640, 480, seed)
                .as_raw()
                .iter()
                .map(|&v| 30.0 + v as f32 * 0.004)
                .collect()
        })
        .collect();
    let mut ctx = FloatTrackerContext::with_capacity(capacity);
    let (allocs, tracked) = allocations(|| {
        let mut tracked = 0;
        for (frame, &options) in celsius
            .iter()
            .cycle()
            .zip(OPTION_SETS.iter().cycle())
            .take(5)
        {
            ctx.advance(FloatImageView::new(frame, 640, 480).unwrap(), 4);
            tracked += ctx
                .track_fb_with(
                    &points,
                    None,
                    TrackParams {
                        min_eigen_threshold: 1e-6,
                        ..ADVANCE_PARAMS
                    },
                    DEFAULT_FB_THRESHOLD,
                    options,
                )
                .len();
        }
        tracked
    });
    assert_eq!(tracked, 5 * points.len());
    assert_eq!(allocs, 0, "reserved f32 advance allocated {allocs} times");
}

#[test]
fn reserved_float_detection_does_not_allocate_on_any_content() {
    let frames: Vec<(u32, u32, Vec<f32>)> = [(640u32, 480u32, 11u32), (320, 240, 12), (5, 4, 13)]
        .into_iter()
        .map(|(w, h, seed)| {
            let pixels = textured(w, h, seed)
                .as_raw()
                .iter()
                .map(|&v| 20.0 + v as f32 * 0.01)
                .collect();
            (w, h, pixels)
        })
        .collect();
    let existing = lattice(50);
    let mut ctx = FloatDetectorContext::with_capacity(DetectorCapacity {
        width: 640,
        height: 480,
        min_distance: 0,
        grid_cells: 24,
        existing_points: existing.len(),
    });
    let threshold = ScoreThreshold::absolute(0.0);
    let (allocs, dense) = allocations(|| {
        let mut dense = 0;
        for (w, h, pixels) in &frames {
            let image = FloatImageView::new(pixels, *w, *h).unwrap();
            dense = dense.max(ctx.detect(image, threshold, 0).len());
            ctx.detect(image, threshold.with_relative(0.01), 8);
            ctx.detect_grid(
                image,
                Grid {
                    cols: 6,
                    rows: 4,
                    max_per_cell: 5,
                },
                threshold,
                0,
                &existing,
            );
        }
        dense
    });
    assert!(dense > 10_000, "dense case found only {dense}");
    assert_eq!(allocs, 0, "reserved f32 detection allocated {allocs} times");
}

#[test]
fn reserved_float_tracking_does_not_allocate_with_varying_points() {
    let celsius = |seed| -> Vec<f32> {
        textured(640, 480, seed)
            .as_raw()
            .iter()
            .map(|&v| 30.0 + v as f32 * 0.004)
            .collect()
    };
    let (prev, next) = (celsius(14), celsius(15));
    let points = lattice(300);
    let mut ctx = FloatTrackerContext::with_capacity(TrackerCapacity {
        width: 640,
        height: 480,
        levels: 4,
        window_size: 21,
        points: points.len(),
    });
    let (allocs, tracked) = allocations(|| {
        let (prev, next) = (
            FloatImageView::new(&prev, 640, 480).unwrap(),
            FloatImageView::new(&next, 640, 480).unwrap(),
        );
        ctx.prepare(prev, next, 4);
        let mut tracked = ctx
            .track(
                &points,
                None,
                TrackParams {
                    window_size: 21,
                    max_iterations: 30,
                    min_eigen_threshold: 1e-6,
                },
            )
            .len();
        tracked += ctx
            .track_fb(
                &points[..120],
                None,
                TrackParams {
                    window_size: 15,
                    max_iterations: 15,
                    min_eigen_threshold: 0.0,
                },
                DEFAULT_FB_THRESHOLD,
            )
            .len();
        ctx.prepare(next, prev, 4);
        tracked += ctx
            .track(
                &points[..7],
                None,
                TrackParams {
                    window_size: 7,
                    max_iterations: 5,
                    min_eigen_threshold: 1e-6,
                },
            )
            .len();
        tracked
    });
    assert_eq!(tracked, 300 + 120 + 7);
    assert_eq!(allocs, 0, "reserved f32 tracking allocated {allocs} times");
}
