# Optical-flow web demo

A tiny in-browser demo of the Lucas-Kanade tracker. Point your phone's camera at
a scene, **tap to drop a point** (or hit **Auto** to detect Shi-Tomasi corners),
and watch the points ride the optical flow as the scene (or the camera) moves.
Everything runs client-side in WebAssembly — no server, no upload.

**▶ Live: [lk-demo.jt3.ru](https://lk-demo.jt3.ru)**

```
web-demo/
├── index.html        # the whole UI (camera + canvas + tap/Auto handling)
├── src/lib.rs        # wasm-bindgen wrapper around TrackerContext
├── pkg/              # built wasm + JS glue (build output; not committed)
└── build.sh / .ps1   # build pkg/ with wasm-pack
```

## Run locally

First build the wasm package (see [Rebuild the wasm](#rebuild-the-wasm) below),
which produces `pkg/`. Then serve the folder.

Camera access (`getUserMedia`) only works in a **secure context**: HTTPS, or
`http://localhost`. Any static file server over localhost works:

```bash
cd web-demo
./build.sh                 # or ./build.ps1 on Windows
python -m http.server 8080
# then open http://localhost:8080
```

On desktop you can test with your webcam. To try it on your phone over your LAN
you need HTTPS (a plain `http://192.168.x.x` will be blocked) — deploy it, or
use a tunneling tool that provides HTTPS (e.g. `cloudflared tunnel`, `ngrok`).

## Deploy (so you can open it on your phone)

Build `pkg/` first, then publish the static files (`index.html` + `pkg/`) to any
HTTPS host:

- **Your own server / nginx** (as at [lk-demo.jt3.ru](https://lk-demo.jt3.ru)):
  copy `index.html` and `pkg/` to the web root. Make sure `.wasm` is served as
  `application/wasm` (nginx: `types { application/wasm wasm; }` or add to
  `mime.types`).
- **GitHub Pages / Netlify / Vercel / Cloudflare Pages**: run `wasm-pack` as the
  build step (or commit `pkg/`) and set the publish directory to `web-demo`.

Then open the deployed URL on your phone, allow the camera, and tap the video.

## Rebuild the wasm

Needed before the first run, and whenever you change `src/lib.rs` or the library:

```bash
cd web-demo
./build.sh          # or ./build.ps1 on Windows
```

This requires [`wasm-pack`](https://rustwasm.github.io/wasm-pack/). SIMD
(`+simd128`) is enabled automatically via the repo's `.cargo/config.toml`.

## Benchmark mode

`index.html?bench=1` skips the camera and times the per-frame tracking step in
the browser: two synthetic textured frames 3 px apart (360x640 by default) go
through `TrackerContext::advance` in turn, and each step tracks the strongest
corners of the first frame with `track` or `track_fb`, 4 levels, 30
iterations. **Start benchmark** (or `&auto=1`) spins for two seconds so the
CPU clock ramps up, then runs every case for 300 steps back to back and prints
medians over samples of 5 steps into `<pre id="bench">`; the same numbers and
a checksum of each case's results are in `window.__lkbench`.

Query options: `w`, `h` (frame size), `points` (default `44,150`), `window`
(`13,21`), `fb` (`-1` = `track` only, `0` = `track_fb` over all levels,
`k` = backward pass over the finest `k` levels; default `-1,0,1`), `iters`,
`batch`, `levels`, `maxiter`, `warmup` (ms).

To run it on an Android phone over USB, serve the folder on localhost,
forward the port with `adb reverse tcp:8080 tcp:8080`, and open
`http://localhost:8080/?bench=1` in Chrome on the phone; `localhost` is a
secure context, so plain HTTP works. Chrome's DevTools protocol
(`adb forward tcp:9222 localabstract:chrome_devtools_remote`) can read
`window.__lkbench` back.

## How it works

Each animation frame the page draws the camera image into a small processing
canvas (longer side ≈ 360 px, chosen for real-time speed), converts it to
grayscale, and hands the bytes to `Tracker.push_frame`. The Rust side keeps the
previous frame, runs pyramidal Lucas-Kanade for every point, drops points that
leave the frame or fail to converge, and returns the survivors to be drawn.

The **Auto** button calls `Tracker.auto_detect`, which runs grid-based
Shi-Tomasi (`good_features_to_track_grid`) on the latest frame to add strong
corners spread uniformly across the image, kept clear of points already tracked.

Tracker parameters (pyramid levels, window size, iterations) are set in
[`src/lib.rs`](src/lib.rs) and tuned for phones; adjust there and rebuild.
