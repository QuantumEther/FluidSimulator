import { createRandom } from './random.mjs';

export const MAX_PART = 50000;
export const MAX_GRID_CELLS = 32768;
export const STRIDE = 16;
export const ACCUM = 'rgba16float';
export const WORLD_WIDTH = 0.60;
export const WORLD_HEIGHT = 0.60;
export const SLICE_DEPTH = 0.10;
const queryString = typeof location === 'undefined' ? '' : location.search;
export const SIM_SEED = Number(new URLSearchParams(queryString).get('seed')) >>> 0 || 20261007;
export const random01 = createRandom(SIM_SEED);
export const randomVibrant = () => {
  const h = random01(), s = 0.72 + random01() * 0.25, v = 0.82 + random01() * 0.16;
  const i = Math.floor(h * 6), f = h * 6 - i, p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
  return [[v,t,p],[q,v,p],[p,v,t],[p,q,v],[t,p,v],[v,p,q]][i % 6];
};

export const S = {
  budget: 8000, timeScale: 1, substeps: 4, solverIterations: 3,
  gravity: 9.81, restDensity: 1000, spacing: 0.002,
  viscosity: 0.001, surfaceTension: 0.072, diffusivity: 1e-9,
  sliceDepth: SLICE_DEPTH, restitution: 0.02, wallRetention: 0.99,
  splatRadius: 0.006, densityThreshold: 0.55, normalStrength: 150,
  specular: 1.4, fresnel: 0.6, subsurface: 0.65,
  neighborMode: 1, debugView: 0, randomColor: true, customColor: [0.13, 0.83, 0.92],
};

const fmt = (n, digits=2) => Number(n).toFixed(digits);
export const SLIDERS = [
  {g:'time', k:'budget', l:'Maximum particles', min:1000, max:50000, step:1000, fmt:n=>n.toLocaleString()},
  {g:'time', k:'timeScale', l:'Simulation speed', min:0.1, max:2, step:0.05, fmt:n=>`${fmt(n,2)}×`},
  {g:'time', k:'substeps', l:'Time subdivisions / frame', min:1, max:8, step:1, fmt:n=>`${n}`},
  {g:'time', k:'solverIterations', l:'Density solver iterations', min:1, max:5, step:1, fmt:n=>`${n}`},
  {g:'physics', k:'gravity', l:'Gravity', min:0, max:20, step:0.1, fmt:n=>`${fmt(n,2)} m/s²`},
  {g:'physics', k:'restDensity', l:'Rest density', min:500, max:1500, step:10, fmt:n=>`${Math.round(n)} kg/m³`},
  {g:'physics', k:'spacing', l:'Particle spacing', min:0.002, max:0.006, step:0.0001, fmt:n=>`${fmt(n*1000,1)} mm`},
  {g:'physics', k:'viscosity', l:'Dynamic viscosity', min:0.0001, max:10, step:0.0001, scale:'log', fmt:n=>`${Number(n)<0.01?Number(n).toExponential(1):fmt(n,3)} Pa·s`},
  {g:'physics', k:'surfaceTension', l:'Surface tension', min:0, max:0.12, step:0.001, fmt:n=>`${fmt(n,3)} N/m`},
  {g:'color', k:'diffusivity', l:'Dye diffusivity', min:0, max:1e-7, step:1e-9, fmt:n=>`${Number(n).toExponential(1)} m²/s`},
  {g:'boundary', k:'restitution', l:'Wall restitution', min:0, max:1, step:0.01, fmt:n=>fmt(n,2)},
  {g:'boundary', k:'wallRetention', l:'Tangential velocity retention', min:0.5, max:1, step:0.01, fmt:n=>fmt(n,2)},
  {g:'render', k:'splatRadius', l:'Particle render radius', min:0.003, max:0.012, step:0.0002, fmt:n=>`${fmt(n*1000,1)} mm`},
  {g:'render', k:'densityThreshold', l:'Fluid surface threshold', min:0.2, max:1.5, step:0.01, fmt:n=>fmt(n,2)},
  {g:'render', k:'normalStrength', l:'Surface normal strength', min:50, max:500, step:5, fmt:n=>`${Math.round(n)}`},
  {g:'render', k:'specular', l:'Specular light', min:0, max:4, step:0.05, fmt:n=>fmt(n,2)},
  {g:'render', k:'fresnel', l:'Fresnel light', min:0, max:1, step:0.02, fmt:n=>fmt(n,2)},
  {g:'render', k:'subsurface', l:'Subsurface light', min:0, max:1, step:0.02, fmt:n=>fmt(n,2)},
];
