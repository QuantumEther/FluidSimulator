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

Run the small CPU-side checks with:

```sh
node --experimental-default-type=module tests/fixed-step.test.mjs
node --experimental-default-type=module tests/random.test.mjs
node --experimental-default-type=module tests/si-units.test.mjs
```

WebGPU shader compilation and real-time performance still need to be checked in a WebGPU-capable browser.
