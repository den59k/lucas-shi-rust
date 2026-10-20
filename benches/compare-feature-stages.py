#!/usr/bin/env python3
"""Compare complete detector calls and fused stages against a historical revision.

Usage: python3 benches/compare-feature-stages.py [before-commit] [new-run-directory]
Resume after a pre-measurement failure: --resume existing-run-directory
Requires cached dependencies, stable wasm32-wasip1 and Node WASI preview1.
The default baseline is the precise, pre-performance detector.
"""

import hashlib
import io
import json
import os
import re
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[1]
RESUME = len(sys.argv) > 1 and sys.argv[1] == "--resume"
if RESUME:
    ROOT = Path(sys.argv[2]).resolve()
    BEFORE = json.loads((ROOT / "metadata.json").read_text())["before_commit"]
else:
    BEFORE = sys.argv[1] if len(sys.argv) > 1 else "e03b82c"
    ROOT = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(tempfile.mkdtemp(prefix="lk-scoring-"))
ROOT.mkdir(parents=True, exist_ok=True)
CASES = ["example_720x405", "texture_320x240", "texture_640x480", "low_contrast_320x240", "flat_640x480"]
SUITES = {"feature_stages": "feature_stages", "features": "features"}
print(f"Run directory: {ROOT}", flush=True)


def command(args, cwd=REPO, env=None):
    return subprocess.check_output(args, cwd=cwd, env=env, stderr=subprocess.STDOUT).decode()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(commit, destination):
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(["git", "archive", commit], cwd=REPO))) as archive:
        archive.extractall(destination)  # Trusted local git archive.


before, after = ROOT / "before-source", ROOT / "after-source"
if not RESUME:
    snapshot(BEFORE, before)
    snapshot("HEAD", after)
    # Freeze the working tree so a concurrent edit cannot change a measured binary.
    for name in ["src", "benches", "examples", ".cargo"]:
        if (REPO / name).exists():
            shutil.copytree(REPO / name, after / name, dirs_exist_ok=True)
    for name in ["Cargo.toml", "Cargo.lock"]:
        shutil.copy2(REPO / name, after / name)
        shutil.copy2(after / name, before / name)
    shutil.copytree(after / "benches", before / "benches", dirs_exist_ok=True)
    source = (before / "src/features.rs").read_text()
    if "fn compute_gradient_products(" in source:
        adapter = (after / "benches/support/feature_stage_access_planes.rs").read_text()
        if "fn compute_min_eigenvalues(" in source:
            adapter = adapter.replace("min_eigenvalue_scores", "compute_min_eigenvalues")
        if "box_sum_3x3_in_place" not in source:
            adapter = adapter.replace("box_sum_3x3_in_place", "box_filter_3x3_in_place")
        (before / "benches/support/feature_stage_access.rs").write_text(adapter)
        baseline_adapter = "full-plane adapter; native tuple buffers, no timed format conversion"
    else:
        # Earlier fused revisions have a non-generic tensor buffer and a
        # quality-level filter.
        adapter = (after / "benches/support/feature_stage_access.rs").read_text()
        if "struct TensorRows<" not in source:
            adapter = adapter.replace("TensorRows::<i32>", "TensorRows")
        if "fn filter_by_threshold(" not in source:
            adapter = adapter.replace("filter_by_threshold(p, Threshold::relative(q))",
                                      "filter_by_quality(p, q)")
        (before / "benches/support/feature_stage_access.rs").write_text(adapter)
        baseline_adapter = "current fused adapter"

    metadata = {
        "before_commit": command(["git", "rev-parse", BEFORE]).strip(),
        "after_commit": command(["git", "rev-parse", "HEAD"]).strip(),
        "after_diff": command(["git", "diff", "--", "src", "Cargo.toml", "Cargo.lock", "benches"]),
        "rustc": command(["rustc", "+stable", "-vV"]).strip(),
        "node": command(["node", "--version"]).strip(), "host": platform.platform(),
        "cpu_model": "not recorded", "background_load": "not controlled",
        "criterion": "0.8.2", "root": str(ROOT), "hashes": {},
        "parameters": {
            "features": {"samples": 50, "warmup_seconds": 1, "measurement_seconds": 3, "bootstrap_resamples": 100000},
            "feature_stages": {"samples": 30, "warmup_seconds": 0.3, "measurement_seconds": 1, "bootstrap_resamples": 10000},
        },
        "order": "all builds and smoke checks first; native then WASI; suites features then stages; before then after per input",
        "baseline_overlay": "same current manifest, lockfile and harness; historical src unchanged",
        "baseline_adapter": baseline_adapter,
        "stage_boundaries": "native score representations; NMS setup clones and returned buffer drops excluded; allocating iter stages include output drops",
    }
    for phase, source in [("before", before), ("after", after)]:
        paths = [source / "Cargo.toml", source / "Cargo.lock", source / "examples/input1.png"]
        for directory in ["src", ".cargo", "benches/support"]:
            paths.extend(p for p in (source / directory).rglob("*") if p.is_file())
        paths.extend(source / "benches" / name for name in ["feature_stages.rs", "features.rs", "wasi-runner.cjs", "compare-feature-stages.py"])
        metadata["hashes"][phase] = {str(p.relative_to(source)): sha(p) for p in paths}
    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    binaries = {}
    for phase, source in [("before", before), ("after", after)]:
        for runtime in ["native", "wasi"]:
            args = ["cargo", "+stable", "bench", "--locked", "--offline", "--bench", "feature_stages", "--bench", "features", "--no-run", "--message-format=json"]
            env = os.environ.copy()
            env["CARGO_TARGET_DIR"] = str(ROOT / f"{phase}-target")
            if runtime == "wasi":
                args.extend(["--target", "wasm32-wasip1"])
            print(f"Building {runtime}/{phase}", flush=True)
            completed = subprocess.run(args, cwd=source, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            (ROOT / f"build-{runtime}-{phase}.log").write_text(completed.stdout)
            if completed.returncode:
                raise RuntimeError(completed.stdout[-12000:])
            for suite in SUITES:
                artifacts = []
                for line in completed.stdout.splitlines():
                    if line.startswith("{"):
                        item = json.loads(line)
                        if item.get("reason") == "compiler-artifact" and item["target"]["name"] == suite and item.get("executable"):
                            artifacts.append(Path(item["executable"]))
                assert len(artifacts) == 1, artifacts
                dest = ROOT / f"{suite}-{runtime}-{phase}{'.wasm' if runtime == 'wasi' else ''}"
                shutil.copy2(artifacts[0], dest)
                binaries[suite, runtime, phase] = dest
                metadata["hashes"][phase][f"{suite}_{runtime}_binary"] = sha(dest)


    # Test the same frozen source in fresh target directories; avoid stale artifacts
    # from other checkouts sharing the workspace's ordinary target directory.
    metadata["tests"] = {}
    for runtime in ["native", "wasi"]:
        print(f"Testing frozen after source on {runtime}", flush=True)
        args = ["cargo", "+stable", "test", "--locked", "--offline", "--all-features", "--no-run", "--message-format=json"]
        if runtime == "wasi":
            args.extend(["--target", "wasm32-wasip1"])
        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = str(ROOT / "after-target")
        completed = subprocess.run(args, cwd=after, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (ROOT / f"test-build-{runtime}.log").write_text(completed.stdout)
        if completed.returncode:
            raise RuntimeError(completed.stdout[-12000:])
        tests = []
        for line in completed.stdout.splitlines():
            if line.startswith("{"):
                item = json.loads(line)
                if item.get("reason") == "compiler-artifact" and item.get("executable") and item["profile"].get("test"):
                    tests.append(Path(item["executable"]))
        assert tests
        metadata["tests"][runtime] = []
        for binary in tests:
            cmd = (["node", str(after / "benches/wasi-runner.cjs"), str(binary)]
                   if runtime == "wasi" else [str(binary)]) + ["--test-threads=1"]
            env["CRITERION_HOME"] = str(ROOT / "test-reports")
            output = command(cmd, cwd=after, env=env)
            (ROOT / f"test-{runtime}-{binary.stem}.log").write_text(output)
            summaries = [line for line in output.splitlines() if line.startswith("test result:")]
            assert summaries and all(line.startswith("test result: ok.") for line in summaries)
            metadata["tests"][runtime].append({"binary": binary.name, "sha256": sha(binary), "results": summaries})
else:
    metadata = json.loads((ROOT / "metadata.json").read_text())
    for name, digest in metadata["hashes"]["after"].items():
        if name == "Cargo.toml":
            strip_authors = lambda p: re.sub(rb"(?m)^authors = .*\r?\n", b"", p.read_bytes())
            assert strip_authors(REPO / name) == strip_authors(after / name), "Build configuration changed since snapshot"
        elif name.startswith("src/") or name in ["Cargo.lock", "benches/feature_stages.rs", "benches/features.rs"] or name.startswith("benches/support/"):
            assert sha(REPO / name) == digest, f"Source changed since snapshot: {name}"
    assert not list((ROOT / "reports").glob("**/new/estimates.json")), "Resume is only for failures before timing"
    binaries = {}
    for phase, source in [("before", before), ("after", after)]:
        for runtime in ["native", "wasi"]:
            for suite in SUITES:
                binary = ROOT / f"{suite}-{runtime}-{phase}{'.wasm' if runtime == 'wasi' else ''}"
                assert binary.exists()
                binaries[suite, runtime, phase] = binary
                metadata["hashes"][phase][f"{suite}_{runtime}_binary"] = sha(binary)
        for name in ["wasi-runner.cjs", "compare-feature-stages.py"]:
            shutil.copy2(REPO / "benches" / name, source / "benches" / name)
            metadata["hashes"][phase][f"benches/{name}"] = sha(source / "benches" / name)
    metadata["after_diff"] = command(["git", "diff", "--", "src", "Cargo.toml", "Cargo.lock", "benches"])
    metadata["runtime_workaround"] = "WASI fd_write uses a JS forwarding boundary after direct imports crashed pre-measurement baseline smoke; detection kernels and clocks unchanged; all smoke checks restarted"
    metadata["tests"] = {}
    for runtime in ["native", "wasi"]:
        metadata["tests"][runtime] = []
        for line in (ROOT / f"test-build-{runtime}.log").read_text().splitlines():
            if not line.startswith("{"):
                continue
            item = json.loads(line)
            if item.get("reason") != "compiler-artifact" or not item.get("executable") or not item["profile"].get("test"):
                continue
            binary = Path(item["executable"])
            text = (ROOT / f"test-{runtime}-{binary.stem}.log").read_text()
            summaries = [s for s in text.splitlines() if s.startswith("test result:")]
            assert summaries and all(s.startswith("test result: ok.") for s in summaries)
            metadata["tests"][runtime].append({"binary": binary.name, "sha256": sha(binary), "results": summaries})
    (ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def run(suite, runtime, phase, args, label):
    executable = binaries[suite, runtime, phase]
    cmd = (["node", str(after / "benches/wasi-runner.cjs"), str(executable)]
           if runtime == "wasi" else [str(executable)]) + args
    env = os.environ.copy()
    env["CRITERION_HOME"] = str(ROOT / "reports" / runtime / phase)
    path = ROOT / f"{suite}-{runtime}-{phase}-{label}.log"
    with path.open("w") as log:
        subprocess.run(cmd, cwd=after, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return path.read_text() if label == "smoke" else None


# Discover actual benchmark IDs rather than hard-coding a result count.
expected = {}
for runtime in ["native", "wasi"]:
    for phase in ["before", "after"]:
        for suite in SUITES:
            print(f"Smoke check {suite}/{runtime}/{phase}", flush=True)
            output = run(suite, runtime, phase, ["--test", "--noplot"], "smoke")
            expected[suite, runtime, phase] = {line[len("Testing "):].strip() for line in output.splitlines() if line.startswith("Testing ")}
            assert expected[suite, runtime, phase], "No benchmarks discovered"
for suite in SUITES:
    reference = expected[suite, "native", "before"]
    assert all(ids == reference for (s, _, _), ids in expected.items() if s == suite)
metadata["benchmark_ids"] = {suite: sorted(expected[suite, "native", "before"]) for suite in SUITES}
(ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

for runtime in ["native", "wasi"]:
    for suite in ["features", "feature_stages"]:
        for case in CASES + (["controlled"] if suite == "feature_stages" else []):
            for phase in ["before", "after"]:
                print(f"Measuring {suite}/{runtime}/{case}/{phase}", flush=True)
                run(suite, runtime, phase, [f"{suite}/.*{case}", "--bench", "--noplot"], case)

records = []
for runtime in ["native", "wasi"]:
    for phase in ["before", "after"]:
        for suite in SUITES:
            report = ROOT / "reports" / runtime / phase / suite
            found = set()
            for estimate in sorted(report.glob("**/new/estimates.json")):
                parts = estimate.relative_to(report).parts
                if suite == "features":
                    stage, case = parts[:2]
                    identifier = f"{suite}/{stage}/{case}"
                else:
                    case, stage = (("controlled", parts[0][len("controlled_"):])
                                   if parts[0].startswith("controlled_") else parts[:2])
                    identifier = f"{suite}/{case}/{stage}"
                found.add(identifier)
                estimates = json.loads(estimate.read_text())
                mean = estimates["mean"]
                ci = mean["confidence_interval"]
                records.append({"suite": suite, "runtime": runtime, "phase": phase, "case": case, "stage": stage,
                                "mean_ns": mean["point_estimate"], "mean_ci95_ns": [ci["lower_bound"], ci["upper_bound"]],
                                "median_ns": estimates["median"]["point_estimate"]})
            assert found == expected[suite, runtime, phase], (suite, runtime, phase, found ^ expected[suite, runtime, phase])
        for line in (ROOT / f"feature_stages-{runtime}-{phase}-smoke.log").read_text().splitlines():
            for tag in ["COUNTS", "FLOOR", "CONTROL", "OUTPUT"]:
                if line.startswith(tag + " "):
                    records.append({"runtime": runtime, "phase": phase, "kind": tag.lower(), **json.loads(line[len(tag) + 1:])})
controls = [{k: v for k, v in r.items() if k not in ["runtime", "phase"]}
            for r in records if r.get("kind") == "control"]
assert len(controls) == 4 and all(c == controls[0] for c in controls)
for phase in ["before", "after"]:
    def same_target_records(runtime, kind):
        return [{k: v for k, v in r.items() if k != "runtime"} for r in records
                if r.get("kind") == kind and r["runtime"] == runtime and r["phase"] == phase]
    for kind in ["counts", "output"]:
        assert len(same_target_records("native", kind)) == len(CASES)
        assert same_target_records("native", kind) == same_target_records("wasi", kind), (phase, kind)
metadata["validation"] = "all discovered benchmark IDs measured; four control fixtures identical; native/WASI counts and complete outputs identical per revision"
metadata["timing_records"] = sum("mean_ns" in r for r in records)
# Detect edits since the snapshot; source hashes identify the measured version.
metadata["working_tree_source_matches_snapshot"] = all(
    sha(REPO / name) == digest for name, digest in metadata["hashes"]["after"].items()
    if name.startswith("src/") or name == "Cargo.lock")
metadata["working_tree_manifest_matches_snapshot"] = sha(REPO / "Cargo.toml") == sha(after / "Cargo.toml")
strip_authors = lambda p: re.sub(rb"(?m)^authors = .*\r?\n", b"", p.read_bytes())
assert strip_authors(REPO / "Cargo.toml") == strip_authors(after / "Cargo.toml"), "Build configuration changed"
metadata["working_tree_manifest_performance_matches_snapshot"] = True
metadata["current_commit_at_completion"] = command(["git", "rev-parse", "HEAD"]).strip()
(ROOT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
(ROOT / "results.json").write_text(json.dumps({"metadata": metadata, "records": records}, indent=2) + "\n")
print(f"Finished: {ROOT / 'results.json'} ({metadata['timing_records']} timings)", flush=True)
