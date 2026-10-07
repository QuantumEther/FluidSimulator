import { MAX_PART, STRIDE, ACCUM } from './config.js';

export async function createGpuRuntime({ canvas, wrap, $ }) {
/* ---------- WebGPU ---------- */
if (!navigator.gpu) {
  $('backendText').textContent = 'NO WEBGPU'; $('backendText').style.color = '#f43f5e';
  throw new Error('WebGPU not supported');
}
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
if (!adapter) { $('backendText').textContent = 'NO ADAPTER'; throw new Error('no adapter'); }
const device  = await adapter.requestDevice();
device.addEventListener('uncapturederror', (e) =>
  console.error('WebGPU:', e.error?.message || e.error));

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
const paramsBuf = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const renderBuf = device.createBuffer({ size:  80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

/* ---------- bind group layouts ---------- */
const physBGL = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  ],
});
const physPL = device.createPipelineLayout({ bindGroupLayouts: [physBGL] });
const physBG = device.createBindGroup({
  layout: physBGL,
  entries: [
    { binding: 0, resource: { buffer: particleBuf } },
    { binding: 1, resource: { buffer: paramsBuf   } },
  ],
});
const pipeDensity   = device.createComputePipeline({ layout: physPL, compute: { module: physMod, entryPoint: 'computeDensity' } });
const pipeForces    = device.createComputePipeline({ layout: physPL, compute: { module: physMod, entryPoint: 'computeForces'  } });
const pipeIntegrate = device.createComputePipeline({ layout: physPL, compute: { module: physMod, entryPoint: 'integrate'     } });

const splatBGL = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
    { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform' } },
  ],
});
const splatPL = device.createPipelineLayout({ bindGroupLayouts: [splatBGL] });
const splatBG = device.createBindGroup({
  layout: splatBGL,
  entries: [
    { binding: 0, resource: { buffer: particleBuf } },
    { binding: 1, resource: { buffer: paramsBuf   } },
  ],
});
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
let canvasW = 600, canvasH = 600, dpr = 1;

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
  canvasW = side; canvasH = side;
  makeAccum(pw, ph);
}
window.addEventListener('resize', resize);
resize();

return {
  device, ctx, particleBuf, paramsBuf, renderBuf, physBG,
  pipeDensity, pipeForces, pipeIntegrate,
  splatPipe, splatBG, compPipe,
  get compBG() { return compBG; },
  get accumView() { return accumView; },
  get canvasW() { return canvasW; },
  get canvasH() { return canvasH; },
};
}
