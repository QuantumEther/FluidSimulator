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

**Smooth flow** is the startup preset, matching the supplied tuning screenshot. Its 0.30× speed is intentional slow motion. Selecting it preserves the current boundary and color choices, which the screenshot does not fully specify; startup uses wall restitution 0.02, tangential retention 0.99, and random colors.

| Control | Smooth flow |
| --- | --- |
| Maximum particles / simulation speed | 34,000 / 0.30× |
| Time subdivisions / density iterations | 8 / 4 |
| Gravity / rest density | 9.81 m/s² / 1,070 kg/m³ |
| Particle spacing | 2.2 mm |
| Dynamic viscosity / surface tension | 0.327 Pa·s / 0.002 N/m |
| Render radius / render resolution | 5.0 mm / 100% |
| Surface threshold / normal strength | 0.89 / 150 |
| Specular / Fresnel / subsurface | 1.20 / 0.40 / 0.90 |
| Dye diffusivity | 1.0e-10 m²/s |
| Neighbor search / diagnostic view | Spatial grid / Fluid appearance |

Balanced flow, Water (experimental), Thick fluid, and Low gravity remain available. Water retains its reference parameters; its surface-force model still needs calibration before that preset can be relied upon. Smooth flow is an empirical visual preset, not a calibrated water model. Diagnostic views include density, estimated pressure, velocity, particle ID, grid cells, and neighbor count. The fixed physics clock is independent of render FPS.

Run the CPU-side checks with Node 22 or newer (no package installation required):

```sh
for test in tests/*.test.mjs; do node "$test" || exit; done
```

WebGPU shader compilation and real-time performance still need to be checked in a WebGPU-capable browser.

Fill replaces the current liquid with a fitting lattice. New batches are initialized on the GPU and rejected where existing particles leave insufficient space. Changing density preserves particle positions, dye and per-particle volume. Changing spacing deliberately restarts a centered sample of the same volume, limited by the budget and tank capacity. All-pairs reference search is limited to 1,000 particles.

## Performance and browser validation

The solver batches ordered dispatches into one compute pass, combines density/lambda and density/dye accumulation, and reuses final grid/density state until particle count, support, mass or grid layout changes. At the Water reference settings it performs 32 neighbor sweeps per steady physics tick, versus 52 previously. Smooth flow uses 80 sweeps per tick and targets 18 physics ticks per wall-clock second at 0.30× speed. Small scenes use linked cells; scenes with at least 2,048 particles use GPU count/scan/scatter cell ranges. This crossover is provisional and should be benchmarked on the target adapter.

Fluid render resolution is independent of the presentation canvas. The 50% setting reduces accumulation pixels and splat raster area to one quarter; fine droplets and edges can lose detail. Full resolution remains the default. Normals use field texel size, and splat weights account for spacing.

Serve the checkout for development and optional GPU checks:

```sh
python3 -m http.server 8002 --bind 127.0.0.1
```

With Playwright and Chromium available in the Node environment, run:

```sh
node tools/webgpu-check.cjs
node tools/preset-check.cjs
node tools/benchmark.cjs
```

Set `CHROMIUM_BIN` or `FLUID_URL` when needed. For a software-only cloud machine, prefix these commands with `WEBGPU_SOFTWARE=1`. Software checks validate execution, not hardware speed. The harness uses an explicit Vulkan SwiftShader backend, which supports both compute and presentation in this environment.

The preset check exercises startup, selection after edits, native slider precision, preservation of hidden boundary/color choices, and a 34,000-particle Fill followed by one physics tick. This smoke check does not establish long-term stability.

The benchmark defaults to 1k/8k/20k/50k particles on hardware, 60 warmup ticks, 180 measured ticks, and three repetitions. Software mode defaults to 1k. Override `FLUID_COUNTS`, `FLUID_WARMUP`, `FLUID_SAMPLES`, `FLUID_REPEATS` and `FLUID_GRID_LAYOUT=linked|contiguous|auto`. It emits JSON with GPU stage and CPU encoding median/p95, completion latency, and basic state checks. The unfused control uses the same corrected equations and selected grid as the optimized path; it is not the old SI implementation. It submits one tick at a time and does not measure real-time FPS.

The opt-in `?benchmark=1` page exposes `window.fluidLab` for controlled snapshots, stepping and metrics. Ordinary runs retain GPU particle state; only emission counts (4 bytes), sampled timestamps (48 bytes), and diagnostics (40 bytes) return to the host. Normal rendering permits two outstanding frames, and Sim Hz reports completed work rather than enqueued work.

See [performance notes](docs/performance.md) for validation evidence and remaining numerical limitations.
