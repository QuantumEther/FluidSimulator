import { MAX_PART, MAX_GRID_CELLS, STRIDE, ACCUM, WORLD_WIDTH, WORLD_HEIGHT } from './config.js?v=si-performance-v5';

const SHADER_SET_VERSION = 'si-performance-v5';

export async function createGpuRuntime({ canvas, wrap, $ }) {
/* ---------- WebGPU ---------- */
if (!navigator.gpu) {
  $('backendText').textContent = 'NO WEBGPU'; $('backendText').style.color = '#f43f5e';
  throw new Error('WebGPU not supported');
}
// Some browsers expose only a default or low-power adapter. Try the preferred
// adapter first, then let the browser choose before treating WebGPU as absent.
let adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
if (!adapter) adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
  $('backendText').textContent = 'GPU UNAVAILABLE';
  $('backendText').style.color = '#f43f5e';
  throw new Error('This browser did not provide a WebGPU adapter.');
}
let device;
if (adapter.features.has('timestamp-query')) {
  try {
    device = await adapter.requestDevice({ requiredFeatures: ['timestamp-query'] });
  } catch (error) {
    console.warn('GPU timestamp queries unavailable; using CPU-side timing only.', error);
  }
}
if (!device) device = await adapter.requestDevice();
let lostReason = null;
device.lost.then(info => { lostReason = info.message || 'The GPU device was lost. Reload to restart.'; });
device.addEventListener('uncapturederror', (e) =>
  console.error('WebGPU:', e.error?.message || e.error));

let gpuTimer = null;
if (device.features.has('timestamp-query')) {
  try {
    gpuTimer = {
      querySet: device.createQuerySet({ type: 'timestamp', count: 6, label: 'frame stage timestamps' }),
      resolveBuffer: device.createBuffer({
        size: 48,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        label: 'timestamp resolve buffer',
      }),
      slots: Array.from({ length: 4 }, (_, index) => ({
        buffer: device.createBuffer({
          size: 48,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
          label: `timestamp readback ${index}`,
        }),
        busy: false,
      })),
      nextSlot: 0,
    };
  } catch (error) {
    console.warn('GPU timestamp queries unavailable; using CPU-side timing only.', error);
    gpuTimer = null;
  }
}

const ctx = canvas.getContext('webgpu');
const PRESENT = navigator.gpu.getPreferredCanvasFormat();
ctx.configure({ device, format: PRESENT, alphaMode: 'opaque' });

/* ============================================================================
   WGSL
   ============================================================================ */
async function fetchShader(path) {
  const url = new URL(path, import.meta.url);
  url.searchParams.set('build', SHADER_SET_VERSION);
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) throw new Error(`Could not load shader ${url.pathname} (HTTP ${response.status}).`);
  return { source: await response.text(), url: url.href };
}
const shaderFiles = await Promise.all([
  fetchShader('../shaders/simulation.wgsl'),
  fetchShader('../shaders/splat.wgsl'),
  fetchShader('../shaders/composite.wgsl'),
]);
const [PHYS_SHADER, SPLAT_SHADER, COMP_SHADER] = shaderFiles.map(file => file.source);

for (const [index, [name, source]] of [['simulation', PHYS_SHADER], ['splat', SPLAT_SHADER], ['composite', COMP_SHADER]].entries()) {
  if (!source.includes(`// Fluid shader set: ${SHADER_SET_VERSION}`)) {
    const servedMarker = source.match(/\/\/ Fluid shader set:\s*[^\s]+/)?.[0] || '(no version marker)';
    throw new Error(`${name} shader build mismatch: expected ${SHADER_SET_VERSION}, received ${servedMarker} from ${shaderFiles[index].url}. Reload the SI conversion preview after its latest files finish deploying.`);
  }
}

/* ============================================================================
   Modules + pipelines
   ============================================================================ */
const physMod  = device.createShaderModule({ code: PHYS_SHADER,  label: 'phys'  });
const splatMod = device.createShaderModule({ code: SPLAT_SHADER, label: 'splat' });
const compMod  = device.createShaderModule({ code: COMP_SHADER,  label: 'comp'  });

for (const [n, m] of [['phys',physMod],['splat',splatMod],['comp',compMod]]) {
  const info = await m.getCompilationInfo?.();
  const errors = info?.messages.filter(msg => msg.type === 'error') ?? [];
  if (errors.length) {
    const details = errors.map(msg => `${msg.lineNum}:${msg.linePos} ${msg.message}`).join('\n');
    throw new Error(`${n} WGSL compilation failed:\n${details}`);
  }
}

/* ---------- buffers ---------- */
device.pushErrorScope('validation');
const particleBuf = device.createBuffer({
  size:  MAX_PART * STRIDE * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
const cellCountsBuf = device.createBuffer({ size: (2 * MAX_GRID_CELLS + 3) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const cellOffsetsBuf = device.createBuffer({ size: (MAX_GRID_CELLS + 256 + 782 * 8 + 10) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const sortedIdsBuf = device.createBuffer({ size: MAX_PART * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const emissionBuf = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const statsReadback = device.createBuffer({ size: 40, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
const emissionReadback = device.createBuffer({ size: 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
const accelerationBuf = device.createBuffer({ size: MAX_PART * 8, usage: GPUBufferUsage.STORAGE });
const previousPositionsBuf = device.createBuffer({
  size: MAX_PART * 8,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
const colorBuffers = [0, 1].map(() => device.createBuffer({
  size: MAX_PART * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
}));
const paramsBuf = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const renderBuf = device.createBuffer({ size:  80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

/* ---------- bind group layouts ---------- */
const physBGL = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  ],
});
const physPL = device.createPipelineLayout({ bindGroupLayouts: [physBGL] });
const makePhysBG = (src, dst) => device.createBindGroup({
  layout: physBGL,
  entries: [
    { binding: 0, resource: { buffer: particleBuf } },
    { binding: 1, resource: { buffer: paramsBuf } },
    { binding: 2, resource: { buffer: cellCountsBuf } },
    { binding: 3, resource: { buffer: cellOffsetsBuf } },
    { binding: 4, resource: { buffer: accelerationBuf } },
    { binding: 5, resource: { buffer: colorBuffers[src] } },
    { binding: 6, resource: { buffer: colorBuffers[dst] } },
    { binding: 7, resource: { buffer: previousPositionsBuf } },
    { binding: 8, resource: { buffer: sortedIdsBuf } },
    { binding: 9, resource: { buffer: emissionBuf } },
  ],
});
const physBGs = [makePhysBG(0, 1), makePhysBG(1, 0)];
const pipeClearGrid = device.createComputePipeline({label: 'clearGrid', layout: physPL, compute: { module: physMod, entryPoint: 'clearGrid' } });
const pipeClearLinkedGrid = device.createComputePipeline({label:'clearLinkedGrid',layout:physPL,compute:{module:physMod,entryPoint:'clearLinkedGrid'}});
const pipeBuildLinkedGrid = device.createComputePipeline({label:'buildLinkedGrid',layout:physPL,compute:{module:physMod,entryPoint:'buildLinkedGrid'}});
const pipeCountGrid = device.createComputePipeline({label: 'countGrid', layout: physPL, compute: { module: physMod, entryPoint: 'countGrid' } });
const pipeScanCells = device.createComputePipeline({label: 'scanCells', layout: physPL, compute: { module: physMod, entryPoint: 'scanCells' } });
const pipeScanBlocks = device.createComputePipeline({label: 'scanBlocks', layout: physPL, compute: { module: physMod, entryPoint: 'scanBlocks' } });
const pipeScatterGrid = device.createComputePipeline({label: 'scatterGrid', layout: physPL, compute: { module: physMod, entryPoint: 'scatterGrid' } });
const pipeDensity = device.createComputePipeline({label: 'computeDensity', layout: physPL, compute: { module: physMod, entryPoint: 'computeDensity' } });
const pipeDensityLambda = device.createComputePipeline({label: 'computeDensityLambda', layout: physPL, compute: { module: physMod, entryPoint: 'computeDensityLambda' } });
const pipeDensityColors = device.createComputePipeline({label: 'computeDensityColors', layout: physPL, compute: { module: physMod, entryPoint: 'computeDensityColors' } });
const pipeLambdasReference = device.createComputePipeline({label:'lambdasReference',layout:physPL,compute:{module:physMod,entryPoint:'lambdasReference'}});
const pipeColorsReference = device.createComputePipeline({label:'colorsReference',layout:physPL,compute:{module:physMod,entryPoint:'colorsReference'}});
const pipeCorrections = device.createComputePipeline({label: 'computeCorrections', layout: physPL, compute: { module: physMod, entryPoint: 'computeCorrections' } });
const pipeApplyCorrections = device.createComputePipeline({label: 'applyCorrections', layout: physPL, compute: { module: physMod, entryPoint: 'applyCorrections' } });
const pipeForces = device.createComputePipeline({label: 'computeForces', layout: physPL, compute: { module: physMod, entryPoint: 'computeForces' } });
const pipeIntegrate = device.createComputePipeline({label: 'integrate', layout: physPL, compute: { module: physMod, entryPoint: 'integrate' } });
const pipeSavePrevious = device.createComputePipeline({label: 'savePreviousPositions', layout: physPL, compute: { module: physMod, entryPoint: 'savePreviousPositions' } });
const pipeUpdateMasses = device.createComputePipeline({label: 'updateMasses', layout: physPL, compute: { module: physMod, entryPoint: 'updateMasses' } });
const pipeClearEmission = device.createComputePipeline({label: 'clearEmission', layout: physPL, compute: { module: physMod, entryPoint: 'clearEmission' } });
const pipeEmit = device.createComputePipeline({label: 'emitParticles', layout: physPL, compute: { module: physMod, entryPoint: 'emitParticles' } });
const pipeCopyEmittedColors = device.createComputePipeline({label: 'copyEmittedColors', layout: physPL, compute: { module: physMod, entryPoint: 'copyEmittedColors' } });
const pipeClearStats = device.createComputePipeline({label:'clearStats',layout:physPL,compute:{module:physMod,entryPoint:'clearStats'}});
const pipeComputeStats = device.createComputePipeline({label:'computeStats',layout:physPL,compute:{module:physMod,entryPoint:'computeStats'}});
const pipeReduceStats = device.createComputePipeline({label:'reduceStats',layout:physPL,compute:{module:physMod,entryPoint:'reduceStats'}});
const pipeBeginViscosity = device.createComputePipeline({label: 'beginViscosity', layout: physPL, compute: { module: physMod, entryPoint: 'beginViscosity' } });
const pipeViscosityA = device.createComputePipeline({label: 'viscosityA', layout: physPL, compute: { module: physMod, entryPoint: 'viscosityA' } });
const pipeViscosityB = device.createComputePipeline({label: 'viscosityB', layout: physPL, compute: { module: physMod, entryPoint: 'viscosityB' } });
const pipeApplyViscosity = device.createComputePipeline({label: 'applyViscosity', layout: physPL, compute: { module: physMod, entryPoint: 'applyViscosity' } });

const splatBGL = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
  ],
});
const splatPL = device.createPipelineLayout({ bindGroupLayouts: [splatBGL] });
const splatBGs = colorBuffers.map(colors => device.createBindGroup({
  layout: splatBGL,
  entries: [
    { binding: 0, resource: { buffer: particleBuf } },
    { binding: 1, resource: { buffer: paramsBuf } },
    { binding: 2, resource: { buffer: colors } },
    { binding: 3, resource: { buffer: previousPositionsBuf } },
    { binding: 4, resource: { buffer: renderBuf } },
  ],
}));
const splatPipe = device.createRenderPipeline({
  layout: splatPL,
  vertex:   { module: splatMod, entryPoint: 'vs' },
  fragment: {
    module: splatMod, entryPoint: 'fs',
    targets: [{
      format: ACCUM,
      blend: {
        color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
        alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
      },
    }],
  },
  primitive: { topology: 'triangle-list' },
});

const compBGL = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
  ],
});
const compPL = device.createPipelineLayout({ bindGroupLayouts: [compBGL] });
const compPipe = device.createRenderPipeline({
  layout: compPL,
  vertex:   { module: compMod, entryPoint: 'vs' },
  fragment: { module: compMod, entryPoint: 'fs', targets: [{ format: PRESENT }] },
  primitive: { topology: 'triangle-list' },
});
const linearSamp = device.createSampler({
  magFilter: 'linear', minFilter: 'linear',
  addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
});

/* ---------- accum target ---------- */
let accumTex = null, accumView = null, compBG = null;
let canvasW = WORLD_WIDTH, canvasH = WORLD_HEIGHT, dpr = 1, fieldWidth = 1, fieldHeight = 1, renderScale = 1;

function makeAccum(w, h) {
  if (accumTex) accumTex.destroy();
  accumTex = device.createTexture({
    size: [w, h], format: ACCUM,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  accumView = accumTex.createView();
  fieldWidth = w; fieldHeight = h;
  compBG = device.createBindGroup({
    layout: compBGL,
    entries: [
      { binding: 0, resource: accumView },
      { binding: 1, resource: linearSamp },
      { binding: 2, resource: { buffer: renderBuf } },
    ],
  });
}

function resize() {
  const small = window.innerWidth < 640;
  const pad = small ? 24 : 48;
  const availW = window.innerWidth  - pad;
  const availH = window.innerHeight - (small ? 240 : 170);
  const side   = Math.max(240, Math.min(availW, availH, 720));
  wrap.style.width  = side + 'px';
  wrap.style.height = side + 'px';
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  const pw = Math.round(side * dpr), ph = Math.round(side * dpr);
  canvas.width = pw; canvas.height = ph;
  canvas.style.width = side + 'px'; canvas.style.height = side + 'px';
  canvasW = WORLD_WIDTH; canvasH = WORLD_HEIGHT;
  makeAccum(Math.max(1, Math.round(pw * renderScale)), Math.max(1, Math.round(ph * renderScale)));
}
window.addEventListener('resize', resize);
resize();
const validationError = await device.popErrorScope();
if (validationError) throw new Error(`GPU pipeline validation failed: ${validationError.message}`);

return {
  get lostReason() { return lostReason; },
  device, ctx, particleBuf, paramsBuf, renderBuf, colorBuffers, physBGs,
  previousPositionsBuf, gpuTimer,
  pipeClearGrid, pipeCountGrid, pipeScanCells, pipeScanBlocks, pipeScatterGrid, pipeDensity, pipeDensityLambda, pipeDensityColors, pipeCorrections, pipeApplyCorrections, pipeForces, pipeIntegrate, pipeSavePrevious, pipeUpdateMasses, pipeClearEmission, pipeEmit, pipeCopyEmittedColors, pipeBeginViscosity, pipeViscosityA, pipeViscosityB, pipeApplyViscosity,
  cellCountsBuf, cellOffsetsBuf, sortedIdsBuf, emissionBuf, emissionReadback, statsReadback,
  pipeClearStats, pipeComputeStats, pipeReduceStats,
  pipeLambdasReference, pipeColorsReference,
  pipeClearLinkedGrid, pipeBuildLinkedGrid,
  splatPipe, splatBGs, compPipe,
  setRenderScale(value) {
    const next = Math.max(0.5, Math.min(1, value));
    if (next !== renderScale) { renderScale = next; resize(); }
  },
  get fieldWidth() { return fieldWidth; },
  get fieldHeight() { return fieldHeight; },
  get compBG() { return compBG; },
  get accumView() { return accumView; },
  get canvasW() { return canvasW; },
  get canvasH() { return canvasH; },
};
}
