import { createRandom } from './random.mjs';

const MAX_PART = 50000;
const MAX_GRID_CELLS = 4096;
const STRIDE = 12;
const ACCUM = 'rgba16float';
const BASE_H = 24.0;

const requestedSeed = Number.parseInt(new URLSearchParams(location.search).get('seed') || '', 10);
const SIM_SEED = Number.isFinite(requestedSeed) && requestedSeed >= 0
  ? requestedSeed >>> 0
  : (() => {
      const value = new Uint32Array(1);
      if (globalThis.crypto?.getRandomValues) crypto.getRandomValues(value);
      else value[0] = Date.now() >>> 0;
      return value[0] || 1;
    })();
const random01 = createRandom(SIM_SEED);

const S = {
  budget: 8000, timeScale: 1.0, substeps: 4,
  gravity: 900, viscosity: 1.16, surfaceTension: 0.94,
  stiffness: 100, restDensity: 2.0, hScale: 0.5, damping: 1.0,
  bounce: 0.2, wallFriction: 0.85,
  blobRadius: 8, densityThreshold: 0.75, normalStrength: 300,
  specular: 2.8, fresnel: 0.5, subsurface: 0.35,
  colorMix: 0.05,
  neighborMode: 1,
  debugView: 0,
  randomColor: true, customColor: [0.13, 0.83, 0.93],
};

const SLIDERS = [
  { g:'time',    k:'budget',          l:'Particle Budget',      min: 500, max: 50000, step: 500, fmt: v => v.toLocaleString() },
  { g:'time',    k:'timeScale',       l:'Time Scale',           min: 0.1, max: 2,    step: 0.05, fmt: v => v.toFixed(2) + '×' },
  { g:'time',    k:'substeps',        l:'Substeps',             min: 1,   max: 4,    step: 1,    fmt: v => v },
  { g:'physics', k:'gravity',         l:'Gravity',              min: 0,   max: 2500, step: 25,   fmt: v => v.toFixed(0) },
  { g:'physics', k:'viscosity',       l:'Viscosity (XSPH)',     min: 0,   max: 3,    step: 0.02, fmt: v => v.toFixed(2) },
  { g:'physics', k:'surfaceTension',  l:'Surface Tension',      min: 0,   max: 2,    step: 0.02, fmt: v => v.toFixed(2) },
  { g:'physics', k:'stiffness',       l:'Pressure Stiffness',   min: 100, max: 5000, step: 50,   fmt: v => v.toFixed(0) },
  { g:'physics', k:'restDensity',     l:'Rest Density',         min: 1,   max: 15,   step: 0.5,  fmt: v => v.toFixed(1) },
  { g:'physics', k:'hScale',          l:'Kernel Radius',        min: 0.5, max: 1.8,  step: 0.05, fmt: v => v.toFixed(2) + '×' },
  { g:'physics', k:'damping',         l:'Velocity Damping',     min: 0.95,max: 1.0,  step: 0.001,fmt: v => v.toFixed(3) },
  { g:'boundary',k:'bounce',          l:'Wall Bounce',          min: 0,   max: 0.7,  step: 0.05, fmt: v => v.toFixed(2) },
  { g:'boundary',k:'wallFriction',    l:'Wall Friction',        min: 0.3, max: 1.0,  step: 0.05, fmt: v => v.toFixed(2) },
  { g:'render',  k:'blobRadius',      l:'Blob Radius',          min: 8,   max: 50,   step: 1,    fmt: v => v },
  { g:'render',  k:'densityThreshold',l:'Density Threshold',    min: 0.05,max: 1.5,  step: 0.05, fmt: v => v.toFixed(2) },
  { g:'render',  k:'normalStrength',  l:'Normal Strength',      min: 10,  max: 300,  step: 5,    fmt: v => v },
  { g:'render',  k:'specular',        l:'Specular Boost',       min: 0,   max: 3,    step: 0.1,  fmt: v => v.toFixed(1) },
  { g:'render',  k:'fresnel',         l:'Fresnel Rim',          min: 0,   max: 1,    step: 0.05, fmt: v => v.toFixed(2) },
  { g:'render',  k:'subsurface',      l:'Subsurface Scatter',   min: 0,   max: 1.5,  step: 0.05, fmt: v => v.toFixed(2) },
  { g:'color',   k:'colorMix',        l:'Colour Mix Rate',      min: 0,   max: 1,    step: 0.05, fmt: v => v.toFixed(2) },
];

function randomVibrant() {
  const h = random01(), s = 0.88, l = 0.55;
  const q = l < 0.5 ? l*(1+s) : l+s-l*s, p = 2*l-q;
  const f = t => {
    if (t<0) t+=1; if (t>1) t-=1;
    if (t<1/6) return p+(q-p)*6*t;
    if (t<1/2) return q;
    if (t<2/3) return p+(q-p)*(2/3-t)*6;
    return p;
  };
  return [f(h+1/3), f(h), f(h-1/3)];
}

export { MAX_PART, MAX_GRID_CELLS, STRIDE, ACCUM, BASE_H, S, SLIDERS, SIM_SEED, random01, randomVibrant };
