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

## Results, 2026-10-17

AMD Ryzen 9 9900X, Windows 11. Rust 1.97 nightly, Node 24.15, Python 3.12,
OpenCV 4.11.0 for both OpenCV.js (`@techstark/opencv-js` 4.11.0-release.1,
the official docs build: scalar, single-threaded) and `opencv-python-headless`.

Two snapshots of the crate: **before** is the 0.4.0 merge of PR #2, **after**
adds the rewritten tracker loop, the row-wise detector and `DetectorContext`.

### Speed, milliseconds per call

| Build | Detect before | Detect after | Track before | Track after |
|---|---:|---:|---:|---:|
| crate, wasm simd128 | 5.13 | **2.11** | 24.4 | **4.18** |
| crate, wasm scalar | 6.11 | 3.36 | 26.2 | 5.36 |
| OpenCV.js 4.11, wasm scalar | 2.72 | 2.44 | 5.97 | 5.80 |
| crate, native | 5.06 | **1.12** | 21.8 | **3.31** |
| OpenCV native, 1 thread | 2.05 | 2.02 | 2.32 | 2.32 |
| OpenCV native, 24 threads | 2.06 | 2.03 | 0.98 | 1.04 |

Track is pyramids plus Lucas-Kanade for 200 points. p90 values are within 3%
of the medians in every row. The OpenCV columns are repeat measurements of the
same code and show the run-to-run noise.

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
1.1 ms with the runtime-selected AVX2 copy. The remaining single largest stage
is the f64 eigenvalue square root, about 0.7 ms before AVX2.

OpenCV's tracking time barely moves with its termination criteria: 2.13 ms at
epsilon 0.01, 2.33 ms at 1e-6, 2.15 ms with a fixed 30 iterations. About 1.6 ms
of its 2.3 ms is building the two Gaussian pyramids.

### Detection parity, native OpenCV

Unchanged by the rewrite: the detector's arithmetic is identical, and the wasm
builds reproduce the native crate's corners to within 5e-5.

| OpenCV gradient | OpenCV corners | crate corners | crate corners within 1.5 px of an OpenCV corner | OpenCV corners within 1.5 px of a crate corner | top-100 overlap |
|---|---:|---:|---:|---:|---:|
| Sobel 3 | 349 | 319 | 94.4% | 86.2% | 94% |
| Scharr | 359 | 319 | 97.2% | 86.4% | 96% |

The median nearest-neighbour distance is 0 px in every case: matched corners sit
on the same pixel. The crate finds fewer corners because it quantizes gradients
and the structure tensor to integers, which drops the weakest responses.

### Tracking parity

| | OpenCV | crate before | crate after |
|---|---:|---:|---:|
| points reported tracked, of 200 | 199 | 121 | 143 |
| median endpoint difference to OpenCV on points both report, px | | 0.0016 | 0.0042 |
| of those, within 1 px of OpenCV | | 108 of 121 | 111 of 143 |

The status columns are not comparable. OpenCV only fails a point that leaves
the image. The crate also reports `Diverged` for any point whose step never
met the convergence test within 30 iterations: 63 points before, 39 after the
relaxed criterion. The points that disagree by more than 1 px lie on the
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
