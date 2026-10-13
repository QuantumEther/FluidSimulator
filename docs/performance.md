# Performance implementation and verification

Implemented on October 13, 2026 against `0a1a556` (the tree audited at `4286048`). No repository package dependencies or build step were added. Playwright is optional tooling for the browser checks.

## What changed

- GPU command-based initialization replaces CPU particle/color/history staging. Commands upload 48 bytes; a four-byte accepted count updates the host. Emission checks current GPU positions, preserves spacing within each batch, and does not clamp oversized patches onto walls.
- Material edits dispatch mass-only GPU updates. Density edits preserve positions, velocities, dye and interpolation history, including edits while emission is pending. Spacing edits explicitly restart a volume-preserving sample within capacity limits. Fill explicitly replaces the scene.
- Ordered compute dispatches share a compute pass. Density and lambda use one traversal. Final density and dye share another traversal. Unchanged final grid/density state supplies the next force stage. Changes to particle count, spacing, mass, search mode or grid layout invalidate it. Integration and each constraint application still require rebuilt neighborhoods.
- Both linked cells and GPU count/scan/scatter cell ranges are supported. Cell ranges retain every particle, without fixed per-cell limits. The automatic crossover at 2,048 particles is provisional; use the benchmark overrides to measure alternatives on the actual adapter.
- Kernel constants are precomputed once per parameter upload. Exactly zero diffusivity skips dye evaluation and preserves the authoritative ping-pong buffer.
- Missing free-surface support no longer creates attractive density constraints. Compression constraints use a larger scale-aware regularizer. Wall correction contributes only accepted displacement to velocity. Surface-force classification uses a dimensionless gradient threshold instead of accepting almost any bulk noise.
- Settings exceeding a reference-lattice viscosity timestep criterion use eight GPU Jacobi iterations of a backward-Euler velocity operator. This prevents the reproduced high-viscosity alternating-mode amplification in the tested scene. It is an approximate iterative solve, not a fully converged/calibrated material model.
- Optional 75%/50% accumulation resolution preserves presentation resolution. Normal sampling uses actual field texels. Splat weights scale with particle area, and the grid diagnostic now displays the true support-sized cells.
- Timestamp samples separate compute, splat and composite stages. Sampled GPU reductions report invalid state, maximum speed/compression/occupancy, kinetic energy, dye integrals and cap fractions. Diagnostic readback is 40 bytes every 60 submitted frames; timestamp readback is 48 bytes every 15 frames. GPU completion callbacks bound normal frame submissions to two outstanding frames and supply completed simulation progress.
- Pipeline validation now fails during initialization with a readable message. Device loss stops submissions and asks for a reload.

## Work counts

Water reference: four subdivisions, three projection iterations, nonzero dye, one physics tick. Counts exclude optional emission, mass edits, diagnostic samples and rendering. The later Smooth flow startup preset uses eight subdivisions and four iterations, or 80 steady-state neighbor sweeps; its 0.30× speed deliberately reduces simulated time per wall-clock second.

| Work | Original SI | Optimized linked grid | Optimized contiguous grid |
| --- | ---: | ---: | ---: |
| Neighbor sweeps | 52 | 32 | 32 |
| Grid builds | 20 | 16 | 16 |
| Compute dispatches, steady state | 109 | 81 | 129 |
| Compute passes per submitted frame | 109 for one tick | 1 | 1 |

The contiguous builder needs five dispatches instead of two, so launch count alone is misleading. Its purpose is contiguous neighbor IDs and removal of dependent list-pointer loads. A cold/invalidated state adds one grid build and density evaluation: three dispatches for linked cells or six for contiguous cells. High-viscosity implicit iterations add work; thinning the material is not a performance optimization.

## Validation performed

All five CPU suites passed: fixed-clock, random, SI kernel/mass, fitting emission/capacity, and scheduling/cache/dye behavior.

Chromium 154 / Vulkan SwiftShader GPU checks passed for shader and pipeline validation, grid/all-pairs parity, linked/contiguous parity, fused/cached versus unfused scheduling parity, dye parity, zero-diffusion ping-pong, mass-only state preservation, sparse-pair behavior, high-viscosity mode damping, overlap rejection, dense cells containing all 100 coincident neighbors, a fitting 9,604-particle fill at 6 mm, all-pairs capacity guard, diagnostic reduction, and half-resolution field allocation. Normal interactive startup/clear is also checked. These checks use the actual WGSL and GPU resources; they do not establish all physical conservation or visual acceptance conditions.

A short 1,000-particle software benchmark used identical initial lattices and corrected equations, three warmup ticks and five measured ticks, with order reversed across two repetitions. With linked cells, unfused median compute times were 65.73 and 55.67 ms; optimized times were 37.37 and 36.46 ms. This is roughly 35–43% lower compute time in that small software experiment. Full completed-tick medians were 154.7/149.5 ms versus 132.7/122.5 ms, because rendering and queue/browser costs remain. The original linked-list SI shader was not timed in this comparison. Short samples and software rendering make these observations unsuitable for desktop/mobile FPS predictions.

An earlier forced-contiguous experiment at the same small count spent about 261–281 ms in optimized compute; it motivated retaining the simpler grid for small scenes. Hardware can have a very different crossover. The benchmark tool records adapter/browser identity and enables explicit grid comparisons.

## Remaining work and limits

The screenshot-based Smooth flow preset passed native UI/slider checks and a 34,000-particle GPU Fill plus one physics tick, with finite state and maximum speed 0.245 m/s on Vulkan SwiftShader. CPU suites and the existing GPU regressions also passed after changing the startup preset. This is a smoke check, not a long-run stability result.

A separate controlled probe used a zero-velocity, jittered 11×11 particle patch, zero gravity and 30 completed physics ticks (0.5 simulated seconds). Maximum speeds were 0.858 m/s for Water, 0.060 m/s for Water with surface tension disabled, and 0.120 m/s for Smooth flow. All states remained finite. This implicates surface-force activity in Water's excess motion in that small test; it does not reproduce the reported full-tank explosion or establish a calibrated force law. Water's parameters and the solver equations remain unchanged by the preset addition.

This is a faster development baseline, not completion of the audit's scientific calibration program. Solid-wall density support, a discretely consistent surface-tension law, viscosity convergence/constitutive calibration, conservative calibrated dye diffusion, calibrated pressure diagnostics, and broad physical acceptance scenes remain. The surface classifier suppresses weak bulk noise but does not fix the omitted Laplacian self term or the derivative stencil's consistency. The reference-lattice viscosity switch is not a stability bound for arbitrary arrangements, and eight Jacobi iterations can under-converge for strong viscosity. Maximum speed/compression and cap rates expose failures; they do not prove stability by themselves.

Default full-resolution rendering remains selected. Lower-resolution silhouette, tiny-jet and temporal image comparisons are still needed before choosing a lower default. Normal scaling and area-weighted splats deliberately change previous resolution-dependent appearance. External Tailwind and Google Fonts were blocked in the current cloud runtime, so styled UI appearance was not validated.

No hardware GPU was available. Measure 8k/20k/50k loads on the target adapter, then tune the grid crossover and profile memory traffic. Spatial record reordering/SoA, adaptive iteration budgets and neighbor skins remain conditional next steps; they should follow measured bottlenecks and quality gates. Neither a 60 FPS claim nor a universal 2× speedup is established here.
