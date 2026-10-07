# GPU Fluid Lab

The app is split into page markup, styles, JavaScript modules, and WGSL shaders. It uses browser-native ES modules and has no package dependencies or build step. The simulator includes screenshot-matched defaults, fluid presets, a selectable spatial grid alongside the all-pairs reference search, and density, pressure, velocity, particle ID, grid cell, and neighbor-count views. The particle budget can be raised to 50,000; use the spatial grid at high counts because the all-pairs reference mode becomes very slow.

Open the GitHub Pages URL for the repository, or serve this folder over localhost and open `index.html` at that local address. WebGPU requires a secure browser context; localhost and GitHub Pages are suitable. There are no extra dependencies or build commands.
