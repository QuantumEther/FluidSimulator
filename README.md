# GPU Fluid Lab

The app is split into page markup, styles, JavaScript modules, and WGSL shaders. It uses browser-native ES modules and has no package dependencies or build step.

Serve this folder over localhost, then open `index.html` at that local address. WebGPU requires a secure browser context; localhost is suitable. For example, a Python installation can serve the directory with `python3 -m http.server 8000`, then visit `http://localhost:8000`.
