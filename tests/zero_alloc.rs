//! Asserts the steady-state tracking hot path performs no heap allocation.
//!
//! Uses a counting global allocator (scoped to this test binary only, so it
//! does not affect the library or other tests) to count allocations across a
//! warmed-up `prepare` + `track` step.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use image::{GrayImage, Luma};
use optical_flow_lk::{DEFAULT_MIN_EIGEN_THRESHOLD, TrackerContext};

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
            21,
            30,
            DEFAULT_MIN_EIGEN_THRESHOLD,
        );
    }

    // Measured steady-state step.
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
