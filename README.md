# GPU Fluid Lab — SI units

The app uses browser-native ES modules and WebGPU WGSL shaders. It has no package dependencies or build step; open it through GitHub Pages or serve the folder from localhost.

## Physical model

- The 2D simulation domain is **0.60 m × 0.60 m**, interpreted as a cross-section extruded through **0.10 m** of depth.
- Positions, velocities, time, acceleration, and kernel support are in m, m/s, s, m/s², and m.
- Density is volumetric kg/m³. A particle stores mass per unit slice depth (kg/m), computed as `rest density × spacing²`; its represented mass is that value multiplied by the 0.10 m depth.
- Density uses a normalized 2D poly6 SPH kernel. Position Based Fluids (PBF) density constraints maintain near incompressibility at interactive time steps. The pressure view shows an estimated pressure from the water bulk modulus and density error; it is diagnostic, not the force solver.
- Dynamic viscosity is Pa·s, surface tension N/m, dye diffusivity m²/s, and restitution/velocity retention are dimensionless.

This is a physically dimensioned 2D educational model, not a high-resolution 3D CFD solver. The rendered particle radius and lighting controls are visual choices. The 50,000-particle limit is a capacity setting; achieved frame rate depends on the GPU and solver iteration count. Spatial grid search is intended for high counts; all-pairs search is a slow reference mode.

## Controls and checks

Presets include balanced flow, water-like properties, a thick fluid, and low gravity. Diagnostic views include density, estimated pressure, velocity, particle ID, grid cells, and neighbor count. The fixed physics clock is independent of render FPS.

Run the CPU-side checks with Node 22 or newer (no package installation required):

```sh
for test in tests/*.test.mjs; do node "$test" || exit; done
```

WebGPU shader compilation and real-time performance still need to be checked in a WebGPU-capable browser.

Fill replaces the current liquid with a fitting lattice. New batches are initialized on the GPU and rejected where existing particles leave insufficient space. Changing density preserves particle positions, dye and per-particle volume. Changing spacing deliberately restarts a centered sample of the same volume, limited by the budget and tank capacity. All-pairs reference search is limited to 1,000 particles.

## Performance and browser validation

The solver batches ordered dispatches into one compute pass, combines density/lambda and density/dye accumulation, and reuses final grid/density state until particle count, support, mass or grid layout changes. At the default settings it performs 32 neighbor sweeps per steady physics tick, versus 52 previously. Small scenes use linked cells; scenes with at least 2,048 particles use GPU count/scan/scatter cell ranges. This crossover is provisional and should be benchmarked on the target adapter.

Fluid render resolution is independent of the presentation canvas. The 50% setting reduces accumulation pixels and splat raster area to one quarter; fine droplets and edges can lose detail. Full resolution remains the default. Normals use field texel size, and splat weights account for spacing.

Serve the checkout for development and optional GPU checks:

```sh
python3 -m http.server 8002 --bind 127.0.0.1
```

With Playwright and Chromium available in the Node environment, run:

```sh
node tools/webgpu-check.cjs
node tools/benchmark.cjs
```

Set `CHROMIUM_BIN` or `FLUID_URL` when needed. For a software-only cloud machine, prefix these commands with `WEBGPU_SOFTWARE=1`. Software checks validate execution, not hardware speed. The harness uses an explicit Vulkan SwiftShader backend, which supports both compute and presentation in this environment.

The benchmark defaults to 1k/8k/20k/50k particles on hardware, 60 warmup ticks, 180 measured ticks, and three repetitions. Software mode defaults to 1k. Override `FLUID_COUNTS`, `FLUID_WARMUP`, `FLUID_SAMPLES`, `FLUID_REPEATS` and `FLUID_GRID_LAYOUT=linked|contiguous|auto`. It emits JSON with GPU stage and CPU encoding median/p95, completion latency, and basic state checks. The unfused control uses the same corrected equations and selected grid as the optimized path; it is not the old SI implementation. It submits one tick at a time and does not measure real-time FPS.

The opt-in `?benchmark=1` page exposes `window.fluidLab` for controlled snapshots, stepping and metrics. Ordinary runs retain GPU particle state; only emission counts (4 bytes), sampled timestamps (48 bytes), and diagnostics (40 bytes) return to the host. Normal rendering permits two outstanding frames, and Sim Hz reports completed work rather than enqueued work.

See [performance notes](docs/performance.md) for validation evidence and remaining numerical limitations.
