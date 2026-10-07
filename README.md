# GPU Fluid Lab

The app is split into page markup, styles, JavaScript modules, and WGSL shaders. It uses browser-native ES modules and has no package dependencies or build step. The current development pass separates read and write data for color diffusion and velocity integration, adds a selectable spatial grid alongside the all-pairs reference search, and provides density, pressure, velocity, particle ID, grid cell, and neighbor-count views.

Open the GitHub Pages URL for the repository, or serve this folder over localhost and open `index.html` at that local address. WebGPU requires a secure browser context; localhost and GitHub Pages are suitable. There are no extra dependencies or build commands.
