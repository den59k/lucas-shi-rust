// Run the existing Criterion harness on Node's WASI preview1 runtime. The
// detector uses the repository's wasm32-wasip1 +simd128 target configuration.
const fs = require("node:fs");
const path = require("node:path");
const { WASI } = require("node:wasi");

async function main() {
    const [modulePath, ...args] = process.argv.slice(2);
    if (!modulePath) {
        throw new Error("usage: node benches/wasi-runner.cjs <features.wasm> [Criterion arguments]");
    }
    const reports = path.resolve(process.env.CRITERION_HOME || "target/criterion-wasi");
    fs.mkdirSync(reports, { recursive: true });
    const wasi = new WASI({
        version: "preview1",
        args: [modulePath, ...args],
        env: { CRITERION_HOME: "/reports" },
        preopens: { "/reports": reports },
        returnOnExit: true,
    });
    const { instance } = await WebAssembly.instantiate(
        fs.readFileSync(modulePath),
        wasi.getImportObject(),
    );
    process.exitCode = wasi.start(instance);
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
