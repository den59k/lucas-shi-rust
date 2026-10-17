// Times optical-flow-lk (wasm, simd128 and scalar builds) against OpenCV.js on
// the same frame pair and point set, and checks that the wasm builds reproduce
// the native crate's results. Run the `native` crate first so `out/` exists.
'use strict';
const fs = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');

const OUT = path.join(__dirname, 'out');
const WARMUP = 20;
const REPS = 100;
const MATCH_RADIUS = 1.5;

function readPgm(file) {
  const buf = fs.readFileSync(file);
  const m = buf.toString('latin1', 0, 64).match(/^P5\s+(\d+)\s+(\d+)\s+255\s/);
  if (!m) throw new Error(`not a P5 PGM: ${file}`);
  const w = +m[1], h = +m[2];
  return { w, h, data: new Uint8Array(buf.buffer, buf.byteOffset + m[0].length, w * h) };
}

function stats(samples) {
  const s = [...samples].sort((a, b) => a - b);
  const pick = (q) => s[Math.round((s.length - 1) * q)];
  return { median: pick(0.5), p90: pick(0.9), min: s[0] };
}

function bench(fn) {
  for (let i = 0; i < WARMUP; i++) fn();
  const t = [];
  for (let i = 0; i < REPS; i++) {
    const s = performance.now();
    fn();
    t.push(performance.now() - s);
  }
  return stats(t);
}

async function loadCv() {
  let cv = require('@techstark/opencv-js');
  if (typeof cv.then === 'function') cv = await cv;
  if (!cv.Mat) {
    await new Promise((resolve) => {
      cv.onRuntimeInitialized = resolve;
      setTimeout(resolve, 15000);
    });
  }
  if (!cv.Mat) throw new Error('OpenCV.js did not initialise');
  return cv;
}

// a, b: arrays of [x, y]. Fraction of a within MATCH_RADIUS of some b, and the
// median nearest-neighbour distance.
function nearestStats(a, b) {
  const mins = a.map(([x, y]) => {
    let best = Infinity;
    for (const [bx, by] of b) {
      const d = Math.hypot(x - bx, y - by);
      if (d < best) best = d;
    }
    return best;
  });
  const sorted = [...mins].sort((p, q) => p - q);
  return {
    within: mins.filter((d) => d <= MATCH_RADIUS).length / Math.max(1, mins.length),
    median: sorted[Math.floor(sorted.length / 2)] ?? NaN,
  };
}

const fmt = (x) => x.toFixed(3);

async function main() {
  const prev = readPgm(path.join(OUT, 'input1.pgm'));
  const next = readPgm(path.join(OUT, 'input2.pgm'));
  const rust = JSON.parse(fs.readFileSync(path.join(OUT, 'rust.json'), 'utf8'));
  const P = rust.params;
  const flatPts = Float32Array.from(rust.track_points.flat());
  const nPts = rust.track_points.length;
  const results = { node: process.version, image: `${prev.w}x${prev.h}`, points: nPts, params: P, reps: REPS, rows: {} };
  console.log(`node ${process.version}, frame ${prev.w}x${prev.h}, ${nPts} points, quality ${P.quality}, min_distance ${P.min_distance}, levels ${P.levels}, window ${P.window}, iterations ${P.iterations}`);

  // --- optical-flow-lk wasm builds -------------------------------------------
  for (const [label, pkg] of [['crate wasm simd128', './wasm/pkg-simd'], ['crate wasm scalar', './wasm/pkg-scalar']]) {
    const { Bench } = require(pkg);
    const b = new Bench(prev.w, prev.h, prev.data, next.data);
    b.set_points(flatPts);
    const simd = b.simd128();

    // Parity against the native crate: corners and tracks should be identical.
    const corners = b.detect_points(P.quality, P.min_distance);
    let cornerDiff = corners.length / 3 === rust.corners.length ? 0 : Infinity;
    rust.corners.forEach(([x, y, s], i) => {
      cornerDiff = Math.max(cornerDiff, Math.abs(corners[3 * i] - x), Math.abs(corners[3 * i + 1] - y), Math.abs(corners[3 * i + 2] - s));
    });
    const tracks = b.track_results(P.levels, P.window, P.iterations);
    let trackDiff = 0, statusMismatch = 0;
    rust.track.forEach(([x, y, st], i) => {
      if (tracks[3 * i + 2] !== st) statusMismatch++;
      if (st === 0) trackDiff = Math.max(trackDiff, Math.abs(tracks[3 * i] - x), Math.abs(tracks[3 * i + 1] - y));
    });

    const detect = bench(() => b.detect(P.quality, P.min_distance));
    const track = bench(() => b.track(P.levels, P.window, P.iterations));
    const tracked = b.track(P.levels, P.window, P.iterations);
    results.rows[label] = { simd128: simd, corners: corners.length / 3, tracked, detect, track, parityVsNative: { cornerMaxDiff: cornerDiff, trackMaxDiff: trackDiff, statusMismatch } };
    console.log(`${label}: simd128=${simd}, ${corners.length / 3} corners, ${tracked} tracked; detect ${fmt(detect.median)} ms, track ${fmt(track.median)} ms; vs native: corner max diff ${cornerDiff}, track max diff ${trackDiff}, status mismatches ${statusMismatch}`);
  }

  // --- OpenCV.js ----------------------------------------------------------------
  const cv = await loadCv();
  const cvVersion = (cv.getBuildInformation().match(/Version control:\s+(\S+)/) || [])[1] || 'unknown';
  const prevMat = new cv.Mat(prev.h, prev.w, cv.CV_8UC1); prevMat.data.set(prev.data);
  const nextMat = new cv.Mat(next.h, next.w, cv.CV_8UC1); nextMat.data.set(next.data);
  const mask = new cv.Mat();
  const corners = new cv.Mat();
  // Only the Sobel overload is exported by OpenCV.js; blockSize 3, no Harris.
  const detectFn = () => cv.goodFeaturesToTrack(prevMat, corners, 0, P.quality, P.min_distance, mask, 3, false, 0.04);
  detectFn();
  const ocvCorners = [];
  for (let i = 0; i < corners.rows; i++) ocvCorners.push([corners.data32F[2 * i], corners.data32F[2 * i + 1]]);
  const rustCorners = rust.corners.map(([x, y]) => [x, y]);
  const cornerParity = {
    opencvCount: ocvCorners.length,
    crateCount: rustCorners.length,
    crateWithinRadiusOfOpencv: nearestStats(rustCorners, ocvCorners),
    opencvWithinRadiusOfCrate: nearestStats(ocvCorners, rustCorners),
    top100Overlap: nearestStats(rustCorners.slice(0, 100), ocvCorners.slice(0, 100)).within,
  };

  const prevPts = cv.matFromArray(nPts, 1, cv.CV_32FC2, Array.from(flatPts));
  const nextPts = new cv.Mat(), status = new cv.Mat(), err = new cv.Mat();
  const win = new cv.Size(P.window, P.window);
  const crit = new cv.TermCriteria(cv.TermCriteria_COUNT | cv.TermCriteria_EPS, P.iterations, 0.001);
  const trackFn = () => cv.calcOpticalFlowPyrLK(prevMat, nextMat, prevPts, nextPts, status, err, win, P.levels - 1, crit);
  trackFn();
  let ocvTracked = 0, both = 0, agree = 0;
  const dists = [];
  for (let i = 0; i < nPts; i++) {
    const ok = status.data[i] === 1;
    const rustOk = rust.track[i][2] === 0;
    if (ok) ocvTracked++;
    if (ok === rustOk) agree++;
    if (ok && rustOk) {
      both++;
      dists.push(Math.hypot(nextPts.data32F[2 * i] - rust.track[i][0], nextPts.data32F[2 * i + 1] - rust.track[i][1]));
    }
  }
  dists.sort((a, b) => a - b);
  const pick = (q) => dists[Math.min(dists.length - 1, Math.floor(dists.length * q))] ?? NaN;
  const trackParity = {
    opencvTracked: ocvTracked,
    crateTracked: rust.track.filter((t) => t[2] === 0).length,
    statusAgreement: agree / nPts,
    bothTracked: both,
    endpointDistPx: { median: pick(0.5), p95: pick(0.95), max: dists[dists.length - 1] ?? NaN },
  };

  const detect = bench(detectFn);
  const track = bench(trackFn);
  const ocvLabel = `OpenCV.js ${cvVersion} (wasm)`;
  results.rows[ocvLabel] = { corners: ocvCorners.length, tracked: ocvTracked, detect, track };
  results.parity = { corners: cornerParity, tracking: trackParity };
  console.log(`${ocvLabel}: ${ocvCorners.length} corners, ${ocvTracked} tracked; detect ${fmt(detect.median)} ms, track ${fmt(track.median)} ms`);
  console.log('corner parity:', JSON.stringify(cornerParity));
  console.log('tracking parity:', JSON.stringify(trackParity));

  results.rows['crate native'] = { corners: rust.corners.length, tracked: trackParity.crateTracked, detect: rust.native_ms.detect, track: rust.native_ms.track };
  fs.writeFileSync(path.join(OUT, 'wasm_results.json'), JSON.stringify(results, null, 2));

  console.log('\n| Build | Detect median ms | Detect p90 | Track median ms | Track p90 |');
  console.log('|---|---:|---:|---:|---:|');
  for (const [label, r] of Object.entries(results.rows)) {
    console.log(`| ${label} | ${fmt(r.detect.median)} | ${fmt(r.detect.p90)} | ${fmt(r.track.median)} | ${fmt(r.track.p90)} |`);
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
