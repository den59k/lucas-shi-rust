//! Asserts that the detection and tracking hot paths perform no heap
//! allocation, both after a warm-up and on the first call into a context
//! created with explicit capacities.
//!
//! Uses a counting global allocator (scoped to this test binary only, so it
//! does not affect the library or other tests).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use image::{GrayImage, Luma};
use optical_flow_lk::{
    DEFAULT_FB_THRESHOLD, DEFAULT_MIN_EIGEN_THRESHOLD, DetectorCapacity, DetectorContext,
    TrackerCapacity, TrackerContext,
};

/// The allocation counter is global, so measured sections must not overlap.
static SERIAL: Mutex<()> = Mutex::new(());

struct CountingAllocator;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);
static COUNTING: AtomicBool = AtomicBool::new(false);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Allocations performed by `f`. The caller holds `SERIAL`.
fn allocations<R>(f: impl FnOnce() -> R) -> (usize, R) {
    ALLOCS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    let result = f();
    COUNTING.store(false, Ordering::Relaxed);
    (ALLOCS.load(Ordering::Relaxed), result)
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

    // Hold the lock across warm-up too: the counter sees every thread's
    // allocations, so another test's warm-up must not overlap a measurement.
    let _serial = SERIAL.lock().unwrap();

    // Warm-up: lets every buffer grow to its steady-state capacity.
    for _ in 0..3 {
        ctx.prepare(&prev, &next, 4);
        ctx.track(
            &points,
            Some(&predicted),
            21,
            30,
            DEFAULT_MIN_EIGEN_THRESHOLD,
        );
    }

    // Measured steady-state step.
    ALLOCS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    ctx.prepare(&prev, &next, 4);
    let results = ctx.track(
        &points,
        Some(&predicted),
        21,
        30,
        DEFAULT_MIN_EIGEN_THRESHOLD,
    );
    let n = results.len();
    COUNTING.store(false, Ordering::Relaxed);

    let allocs = ALLOCS.load(Ordering::Relaxed);
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

    let _serial = SERIAL.lock().unwrap();

    // Warm-up: lets every buffer grow to its steady-state capacity.
    for _ in 0..3 {
        ctx.detect(&img, 0.01, 8);
        ctx.detect_grid(&img, 6, 4, 5, 0.01, 8, &existing);
    }

    ALLOCS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    let plain = ctx.detect(&img, 0.01, 8).len();
    let grid = ctx.detect_grid(&img, 6, 4, 5, 0.01, 8, &existing).len();
    COUNTING.store(false, Ordering::Relaxed);

    let allocs = ALLOCS.load(Ordering::Relaxed);
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
    let _serial = SERIAL.lock().unwrap();

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
                ctx.detect_grid(image, 6, 4, 5, 0.0, 0, &existing).len(),
                ctx.detect_grid(image, 4, 3, 50, 0.01, 6, &existing[..10])
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
            ctx.detect_grid(image, 6, 4, 1000, 0.0, 8, &existing);
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
    let _serial = SERIAL.lock().unwrap();

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
                21,
                30,
                DEFAULT_MIN_EIGEN_THRESHOLD,
            )
            .len();
        tracked += ctx
            .track(&points[..40], None, 15, 10, DEFAULT_MIN_EIGEN_THRESHOLD)
            .len();
        tracked += ctx
            .track_fb(
                &points[..150],
                Some(&predicted[..150]),
                21,
                30,
                DEFAULT_MIN_EIGEN_THRESHOLD,
                DEFAULT_FB_THRESHOLD,
            )
            .len();
        ctx.prepare(&next, &prev, 4);
        tracked += ctx
            .track_fb(
                &points,
                None,
                7,
                30,
                DEFAULT_MIN_EIGEN_THRESHOLD,
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
    let _serial = SERIAL.lock().unwrap();
    for _ in 0..2 {
        ctx.prepare(&prev, &next, 3);
        ctx.track_fb(
            &points,
            None,
            15,
            15,
            DEFAULT_MIN_EIGEN_THRESHOLD,
            DEFAULT_FB_THRESHOLD,
        );
    }
    let (allocs, n) = allocations(|| {
        ctx.prepare(&prev, &next, 3);
        ctx.track_fb(
            &points,
            None,
            15,
            15,
            DEFAULT_MIN_EIGEN_THRESHOLD,
            DEFAULT_FB_THRESHOLD,
        )
        .len()
    });
    assert_eq!(n, points.len());
    assert_eq!(allocs, 0, "steady-state track_fb allocated {allocs} times");
}
