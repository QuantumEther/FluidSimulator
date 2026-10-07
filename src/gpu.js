import { MAX_PART, MAX_GRID_CELLS, STRIDE, ACCUM, WORLD_WIDTH, WORLD_HEIGHT } from './config.js';

const SHADER_SET_VERSION = 'si-units-v1';

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
device.addEventListener('uncapturederror', (e) =>
  console.error('WebGPU:', e.error?.message || e.error));

let gpuTimer = null;
if (device.features.has('timestamp-query')) {
  try {
    gpuTimer = {
      querySet: device.createQuerySet({ type: 'timestamp', count: 2, label: 'frame timestamps' }),
      resolveBuffer: device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        label: 'timestamp resolve buffer',
      }),
      slots: Array.from({ length: 4 }, (_, index) => ({
        buffer: device.createBuffer({
          size: 16,
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
const [PHYS_SHADER, SPLAT_SHADER, COMP_SHADER] = await Promise.all([
  fetch(new URL('../shaders/simulation.wgsl', import.meta.url)).then(r => r.text()),
  fetch(new URL('../shaders/splat.wgsl', import.meta.url)).then(r => r.text()),
  fetch(new URL('../shaders/composite.wgsl', import.meta.url)).then(r => r.text()),
]);

for (const [name, source] of [['simulation', PHYS_SHADER], ['splat', SPLAT_SHADER], ['composite', COMP_SHADER]]) {
  if (!source.includes(`// Fluid shader set: ${SHADER_SET_VERSION}`)) {
    throw new Error(`The ${name} shader does not match this app build. Reload the SI foundation branch preview so its JavaScript and shader files are served together.`);
  }
}

/* ============================================================================
   Modules + pipelines
   ============================================================================ */
const physMod  = device.createShaderModule({ code: PHYS_SHADER,  label: 'phys'  });
const splatMod = device.createShaderModule({ code: SPLAT_SHADER, label: 'splat' });
const compMod  = device.createShaderModule({ code: COMP_SHADER,  label: 'comp'  });

for (const [n, m] of [['phys',physMod],['splat',splatMod],['comp',compMod]]) {
  m.getCompilationInfo?.().then(info => {
    for (const msg of info.messages) if (msg.type === 'error')
      console.error(`[${n}] ${msg.lineNum}:${msg.linePos} ${msg.message}`);
  });
}

/* ---------- buffers ---------- */
const particleBuf = device.createBuffer({
  size:  MAX_PART * STRIDE * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const cellHeadsBuf = device.createBuffer({ size: MAX_GRID_CELLS * 4, usage: GPUBufferUsage.STORAGE });
const particleNextBuf = device.createBuffer({ size: MAX_PART * 4, usage: GPUBufferUsage.STORAGE });
const accelerationBuf = device.createBuffer({ size: MAX_PART * 8, usage: GPUBufferUsage.STORAGE });
const previousPositionsBuf = device.createBuffer({
  size: MAX_PART * 8,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
const colorBuffers = [0, 1].map(() => device.createBuffer({
  size: MAX_PART * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
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
  ],
});
const physPL = device.createPipelineLayout({ bindGroupLayouts: [physBGL] });
const makePhysBG = (src, dst) => device.createBindGroup({
  layout: physBGL,
  entries: [
    { binding: 0, resource: { buffer: particleBuf } },
    { binding: 1, resource: { buffer: paramsBuf } },
    { binding: 2, resource: { buffer: cellHeadsBuf } },
    { binding: 3, resource: { buffer: particleNextBuf } },
    { binding: 4, resource: { buffer: accelerationBuf } },
    { binding: 5, resource: { buffer: colorBuffers[src] } },
    { binding: 6, resource: { buffer: colorBuffers[dst] } },
    { binding: 7, resource: { buffer: previousPositionsBuf } },
  ],
});
const physBGs = [makePhysBG(0, 1), makePhysBG(1, 0)];
const pipeClearGrid = device.createComputePipeline({ layout: physPL, compute: { module: physMod, entryPoint: 'clearGrid' } });
const pipeBuildGrid = device.createComputePipeline({ layout: physPL, compute: { module: physMod, entryPoint: 'buildGrid' } });
const pipeDensity   = device.createComputePipeline({ layout: physPL, compute: { module: physMod, entryPoint: 'computeDensity' } });
const pipeColors    = device.createComputePipeline({ layout: physPL, compute: { module: physMod, entryPoint: 'diffuseColors' } });
const pipeLambdas   = device.createComputePipeline({ layout: physPL, compute: { module: physMod, entryPoint: 'computeLambdas' } });
const pipeCorrections = device.createComputePipeline({ layout: physPL, compute: { module: physMod, entryPoint: 'computeCorrections' } });
const pipeApplyCorrections = device.createComputePipeline({ layout: physPL, compute: { module: physMod, entryPoint: 'applyCorrections' } });
const pipeForces    = device.createComputePipeline({ layout: physPL, compute: { module: physMod, entryPoint: 'computeForces'  } });
const pipeIntegrate = device.createComputePipeline({ layout: physPL, compute: { module: physMod, entryPoint: 'integrate'     } });
const pipeSavePrevious = device.createComputePipeline({ layout: physPL, compute: { module: physMod, entryPoint: 'savePreviousPositions' } });

const splatBGL = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
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
let canvasW = WORLD_WIDTH, canvasH = WORLD_HEIGHT, dpr = 1;

function makeAccum(w, h) {
  if (accumTex) accumTex.destroy();
  accumTex = device.createTexture({
    size: [w, h], format: ACCUM,
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  });
  accumView = accumTex.createView();
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
  makeAccum(pw, ph);
}
window.addEventListener('resize', resize);
resize();

return {
  device, ctx, particleBuf, paramsBuf, renderBuf, colorBuffers, physBGs,
  previousPositionsBuf, gpuTimer,
  pipeClearGrid, pipeBuildGrid, pipeDensity, pipeColors, pipeLambdas, pipeCorrections, pipeApplyCorrections,
  pipeForces, pipeIntegrate, pipeSavePrevious,
  splatPipe, splatBGs, compPipe,
  get compBG() { return compBG; },
  get accumView() { return accumView; },
  get canvasW() { return canvasW; },
  get canvasH() { return canvasH; },
};
}
