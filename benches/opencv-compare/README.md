# OpenCV comparison

Compares `optical-flow-lk` with OpenCV on the same inputs, in two parts:

- **Parity**: do both libraries find the same corners and track points to the
  same positions? Measured natively against OpenCV 4.11 in Python.
- **Speed in WebAssembly**: the deployment target this crate is built for.
  Measured in Node against OpenCV.js 4.11, with native rows for context.

Nothing here is used by the library. All timings are whole calls on preloaded
inputs: no image decoding, no copies into or out of wasm memory.

## Layout

| Path | Role |
|---|---|
| `native/` | Rust bin. Writes `out/input1.pgm`, `out/input2.pgm` (the bundled example pair as the crate sees it, converted to gray by the `image` crate), `out/rust.json` (the crate's corners, tracks and native timings). Also an `iters` bin that sweeps iteration caps and window sizes. |
| `wasm/` | wasm-bindgen wrapper exposing detection and pyramid-build-plus-tracking as separate calls. Built twice: `pkg-simd` (simd128, as the web demo ships) and `pkg-scalar`. |
| `harness.cjs` | Node harness. Times both crate builds and OpenCV.js, checks the wasm builds reproduce the native results, writes `out/wasm_results.json`. |
| `parity.py` | Native OpenCV parity and timing, writes `out/native_results.json`. |
| `run.sh` | Runs everything in order. |

## Parameters

Detection: quality 0.01, minimum distance 5 px, block size 3, no Harris. The
crate uses Scharr gradients; OpenCV.js only exports the Sobel overload, so the
native Python run reports both Sobel and Scharr.

Tracking: the crate's first 200 corners as the common point set, 21 px window,
4 pyramid levels (OpenCV `maxLevel` 3), 30 iterations, epsilon 0.001, no
initial guess, no forward-backward check. OpenCV compares the squared step
against epsilon and has an oscillation exit, so it stops at a step of about
0.03 px; the crate stops at a 0.01 px step or when consecutive steps cancel.

Each timing is the median of 100 calls after warm-up. The frame pair is the
bundled 720×405 example; its median inter-frame motion is about 19 px.

## Results, 2026-10-17 and 2026-10-18

AMD Ryzen 9 9900X, Windows 11. Rust 1.97 nightly, Node 24.15, Python 3.12,
OpenCV 4.11.0 for both OpenCV.js (`@techstark/opencv-js` 4.11.0-release.1,
the official docs build: scalar, single-threaded) and `opencv-python-headless`.

Three snapshots of the crate: **before** is the 0.4.0 merge of PR #2, **after**
adds the rewritten tracker loop, the row-wise detector and `DetectorContext`,
and **exact** is `main` after PR #5, which keeps the Scharr gradients
unquantized, fuses products, window sums and scores into one row pass and runs
non-maximum suppression with an early quality floor. The before and after
columns were measured on 2026-10-17, the exact columns on 2026-10-18.

### Speed, milliseconds per call

| Build | Detect before | Detect after | Detect exact | Track before | Track after |
|---|---:|---:|---:|---:|---:|
| crate, wasm simd128 | 5.13 | 1.61 | **0.76** | 24.4 | **4.19** |
| crate, wasm scalar | 6.11 | 3.15 | 3.39 | 26.2 | 5.41 |
| OpenCV.js 4.11, wasm scalar | 2.72 | 2.45 | 2.47 | 5.97 | 5.81 |
| crate, native | 5.06 | 1.03 | **0.77** | 21.8 | **3.34** |
| OpenCV native, 1 thread | 2.05 | 2.02 | 2.05 | 2.32 | 2.32 |
| OpenCV native, 24 threads | 2.06 | 2.03 | 2.05 | 0.98 | 1.04 |

Track is pyramids plus Lucas-Kanade for 200 points. p90 values are within 3%
of the medians in every row. The OpenCV columns are repeat measurements of the
same code and show the run-to-run noise. PR #5 did not touch the tracker; its
track timings came back within 2% of the after column and are not repeated.

Where the crate's tracking time went before, from the `iters` bin (native,
track only, pyramids excluded): 4.6 ms with a 1-iteration cap and 21.5 ms at
30, so almost all of it was the per-point sampling loop, which did a bounds
check, a floor and four weight multiplies for every sample. After the rewrite
the same sweep reads 0.8 ms and 3.3 ms. The pyramid build is 0.07 ms either
way.

Detection before was dominated by the three 3×3 box passes (1.9 ms, a variable
divisor per pixel) and non-maximum suppression over a 3.5 MB vector of
per-pixel tuples (1.6 ms plus 0.7 ms to build it). After: the box passes take
1.1 ms and NMS 0.2 ms with the baseline SSE2 build, and the whole pipeline
1.1 ms with the runtime-selected AVX2 copy. Replacing the f64 eigenvalue
square root with a cancellation-free f32 form (exact integer determinant and
discriminant) then took native detection to 1.0 ms and wasm simd128 from
2.1 to 1.6 ms; the f32 gain is larger in wasm because simd128 has only two
f64 lanes.

PR #5 (exact) then dropped the integer truncation of the gradients, so the
products are i32 and the 3×3 sums are accumulated exactly, and fused products,
window sums and scores into one pass over a three-row ring instead of three
tensor planes. Non-maximum suppression became a separable 3×3 maximum on the
score plane, and when the global maximum lies in the interior the relative
quality threshold is applied while collecting maxima. Native detection went
from 1.03 to 0.77 ms and wasm simd128 from 1.61 to 0.76 ms, so the simd128
build now matches native. The scalar wasm build got slower, 3.15 to 3.39 ms,
reproduced in two runs; the cause was not investigated.

OpenCV's tracking time barely moves with its termination criteria: 2.13 ms at
epsilon 0.01, 2.33 ms at 1e-6, 2.15 ms with a fixed 30 iterations. About 1.6 ms
of its 2.3 ms is building the two Gaussian pyramids.

### Detection parity, native OpenCV

The rewrite (after) left corner positions identical to before; scores moved by
at most one part in ten million with the f32 eigenvalue form. PR #5 (exact)
changes the corners: the crate now finds 354 instead of 319, and they line up
with OpenCV's Scharr corners almost exactly. In every snapshot the wasm builds
reproduce the native crate's corners to within 5e-5.

| Crate | OpenCV gradient | OpenCV corners | crate corners | crate corners within 1.5 px of an OpenCV corner | OpenCV corners within 1.5 px of a crate corner | top-100 overlap |
|---|---|---:|---:|---:|---:|---:|
| after | Sobel 3 | 349 | 319 | 94.4% | 86.2% | 94% |
| after | Scharr | 359 | 319 | 97.2% | 86.4% | 96% |
| exact | Sobel 3 | 349 | 354 | 94.1% | 95.4% | 95% |
| exact | Scharr | 359 | 354 | 100% | 98.6% | 98% |

The median nearest-neighbour distance is 0 px in every case: matched corners sit
on the same pixel. Before PR #5 the crate found fewer corners because it
quantized gradients and the structure tensor to integers, which dropped the
weakest responses; with exact accumulation it finds nearly the same set as
OpenCV's Scharr detector.

### Tracking parity

| | OpenCV | crate before | crate after | crate exact |
|---|---:|---:|---:|---:|
| points reported tracked, of 200 | 199 | 121 | 143 | 142 |
| median endpoint difference to OpenCV on points both report, px | | 0.0016 | 0.0042 | 0.0039 |
| of those, within 1 px of OpenCV | | 108 of 121 | 111 of 143 | 112 of 142 |

The exact column tracks a different point set, since the first 200 corners
changed with the detector; the tracker itself is the same as after.

The status columns are not comparable. OpenCV only fails a point that leaves
the image. The crate also reports `Diverged` for any point whose step never
met the convergence test within 30 iterations: 63 points before, 39 after the
relaxed criterion, 40 on the exact point set. The points that disagree by more than 1 px lie on the
image's long bright streaks, where motion along the streak is ambiguous;
OpenCV's own forward-backward check rejects 49 of its 200 answers on this pair.

## Running

```sh
./run.sh
```

Requires stable or nightly Rust with the `wasm32-unknown-unknown` target,
wasm-pack, Node 18+, and Python 3 with `opencv-python-headless` and `numpy`.
The script builds `native/`, runs it to populate `out/`, builds both wasm
variants, installs OpenCV.js, and runs the Node and Python harnesses.
