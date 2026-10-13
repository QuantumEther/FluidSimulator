export const DEFAULT_PRESET_ID = 'smoothFlow';

// Smooth flow records the visible slider values in the supplied screenshot.
// Omitted boundary/color choices retain the user's current settings.
export const PRESETS = {
  smoothFlow: {
    budget:34000, timeScale:0.30, substeps:8, solverIterations:4,
    gravity:9.81, restDensity:1070, spacing:0.0022, viscosity:0.327, surfaceTension:0.002,
    splatRadius:0.005, renderScale:1, densityThreshold:0.89, normalStrength:150,
    specular:1.20, fresnel:0.40, subsurface:0.90, diffusivity:1e-10,
    neighborMode:1, debugView:0,
  },
  screenshot: {
    budget:8000, timeScale:1, substeps:4, solverIterations:2, gravity:9.81, viscosity:0.25,
    surfaceTension:0.072, restDensity:1000, spacing:0.002, restitution:0.15,
    wallRetention:0.94, splatRadius:0.006, densityThreshold:0.72, normalStrength:300, specular:2.8,
    fresnel:0.5, subsurface:0.35, diffusivity:1e-8, neighborMode:1, debugView:0,
  },
  water: {
    budget:8000, timeScale:1, substeps:4, solverIterations:3, gravity:9.81, viscosity:0.001,
    surfaceTension:0.072, restDensity:1000, spacing:0.002, restitution:0.02,
    wallRetention:0.99, splatRadius:0.006, densityThreshold:0.55,
    normalStrength:150, specular:1.4, fresnel:0.6, subsurface:0.65, diffusivity:1e-9,
    neighborMode:1, debugView:0,
  },
  honey: {
    budget:8000, timeScale:1, substeps:4, solverIterations:3, gravity:9.81, viscosity:2.0,
    surfaceTension:0.05, restDensity:1400, spacing:0.002, restitution:0.01,
    wallRetention:0.995, splatRadius:0.007, densityThreshold:0.55,
    normalStrength:150, specular:1.2, fresnel:0.4, subsurface:0.9, diffusivity:1e-10,
    neighborMode:1, debugView:0,
  },
  lowGravity: {
    budget:8000, timeScale:1, substeps:4, solverIterations:2, gravity:1.62, viscosity:0.001,
    surfaceTension:0.072, restDensity:1000, spacing:0.002, restitution:0.02,
    wallRetention:0.99, splatRadius:0.006, densityThreshold:0.55,
    normalStrength:150, specular:1.4, fresnel:0.6, subsurface:0.65, diffusivity:1e-9,
    neighborMode:1, debugView:0,
  },
};
