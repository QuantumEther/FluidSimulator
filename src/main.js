
/* ============================================================================
   WebGPU Fluid Lab — realism upgrade
   ============================================================================ */
import { MAX_PART, STRIDE, S, SIM_SEED, random01, randomVibrant, WORLD_WIDTH, WORLD_HEIGHT } from './config.js?v=si-units-v4';
import { createGpuRuntime } from './gpu.js?v=si-units-v4';
import { setupInput, setupControls } from './ui.js?v=si-units-v4';
import { advanceFixedClock, FIXED_DT } from './fixed-step.mjs';
import { particleMassKg, particleMassPerDepth } from './si-units.mjs';

const canvas = document.getElementById('simCanvas');
const wrap   = document.getElementById('canvasWrap');
const $      = (id) => document.getElementById(id);

let gpu;
try {
  gpu = await createGpuRuntime({ canvas, wrap, $ });
} catch (error) {
  const notice = $('startupError');
  if (notice) {
    const message = !navigator.gpu
      ? 'WebGPU is unavailable in this browser. Try a current browser with hardware acceleration enabled.'
      : error.message.includes('adapter')
        ? 'The browser could not access a GPU adapter. Enable hardware acceleration, then reload the page.'
        : `The simulator could not start: ${error.message}`;
    $('startupErrorText').textContent = message;
    notice.hidden = false;
  }
  console.error('Fluid Lab startup failed:', error);
  throw error;
}

/* ============================================================================
   Particle staging
   ============================================================================ */
const stage = new Float32Array(MAX_PART * STRIDE);
const colorStage = new Float32Array(MAX_PART * 4);
const previousPositionStage = new Float32Array(MAX_PART * 2);
let live = 0;

function writeP(i, x, y, vx, vy, r, g, b) {
  const o = i * STRIDE;
  stage[o+0]=x; stage[o+1]=y; stage[o+2]=vx; stage[o+3]=vy;
  stage[o+4]=r; stage[o+5]=g; stage[o+6]=b; stage[o+7]=1;
  stage[o+8]=S.restDensity; stage[o+9]=0;
  stage[o+10]=particleMassPerDepth(S.restDensity,S.spacing); stage[o+11]=1;
  stage[o+12]=0; stage[o+13]=0; stage[o+14]=0; stage[o+15]=0;
  const c = i * 4;
  colorStage[c]=r; colorStage[c+1]=g; colorStage[c+2]=b; colorStage[c+3]=1;
  const p = i * 2;
  previousPositionStage[p]=x; previousPositionStage[p+1]=y;
}
function upload(from = 0) {
  if (live <= from) return;
  gpu.device.queue.writeBuffer(gpu.particleBuf, from * STRIDE * 4, stage, from * STRIDE, (live - from) * STRIDE);
  gpu.device.queue.writeBuffer(gpu.colorBuffers[0], from * 16, colorStage, from * 4, (live - from) * 4);
  gpu.device.queue.writeBuffer(gpu.colorBuffers[1], from * 16, colorStage, from * 4, (live - from) * 4);
  gpu.device.queue.writeBuffer(gpu.previousPositionsBuf, from * 8, previousPositionStage, from * 2, (live - from) * 2);
}

/* ---------- spawning ---------- */
function spawn(px, py, n, col) {
  const cap = Math.min(S.budget, MAX_PART);
  if (live >= cap) return;
  n = Math.min(n, cap - live);
  const firstNew = live;
  const spacing = S.spacing;
  const cols = Math.max(1, Math.ceil(Math.sqrt(n)));
  const rows = Math.ceil(n / cols);
  const pad = spacing * 1.25;
  for (let k = 0; k < n; k++) {
    const colIndex = k % cols, rowIndex = Math.floor(k / cols);
    const x = Math.max(pad, Math.min(gpu.canvasW-pad, px + (colIndex-(cols-1)*0.5)*spacing + (random01()-0.5)*spacing*0.18));
    const y = Math.max(pad, Math.min(gpu.canvasH-pad, py + (rowIndex-(rows-1)*0.5)*spacing + (random01()-0.5)*spacing*0.18));
    writeP(live + k, x, y,
      (random01()-0.5) * 0.04,
      random01() * 0.04,
      col[0], col[1], col[2]);
  }
  live += n;
  upload(firstNew);
  $('countVal').textContent = live.toLocaleString();
}
function refreshParticleMasses() {
  const massPerDepth = particleMassPerDepth(S.restDensity,S.spacing);
  for (let i=0;i<live;i++) stage[i*STRIDE+10]=massPerDepth;
  upload(0);
  const mass = particleMassKg(S.restDensity,S.spacing,S.sliceDepth);
  $('particleMassVal').textContent = mass < 0.001 ? `${(mass*1e6).toFixed(0)} mg` : `${(mass*1000).toFixed(2)} g`;
}
function trimToBudget() {
  const cap = Math.min(S.budget, MAX_PART);
  if (live > cap) { live = cap; $('countVal').textContent = live.toLocaleString(); }
}

const getCanvasSize = () => ({ width: gpu.canvasW, height: gpu.canvasH });
const input = setupInput({ canvas, getCanvasSize, spawn, S });
setupControls({
  S, $, trimToBudget, getLive: () => live, setLive: value => { live = value; },
  upload, spawn, pick: () => S.randomColor ? randomVibrant() : S.customColor, getCanvasSize,
  refreshParticleMasses,
});
refreshParticleMasses();

/* ============================================================================
   Uniform writers
   ============================================================================ */
const pF = new Float32Array(32);
const pU = new Uint32Array(pF.buffer);
const rF = new Float32Array(20);

function writeParams(dt) {
  const h = S.spacing * 2;
  pF[0]  = 0;
  pF[1]  = S.gravity;
  pF[2]  = h;
  pF[3]  = h * h;
  pF[4]  = S.restDensity;
  pF[5]  = 2.2e9;
  pF[6]  = S.viscosity;
  pF[7]  = S.surfaceTension;
  pF[8]  = S.diffusivity;
  pF[9]  = S.restitution;
  pF[10] = S.wallRetention;
  pF[11] = dt;
  pF[12] = performance.now() * 0.001;
  pF[13] = WORLD_WIDTH;
  pF[14] = WORLD_HEIGHT;
  pU[15] = live;
  pU[16] = Math.ceil(WORLD_WIDTH / h);
  pU[17] = Math.ceil(WORLD_HEIGHT / h);
  pU[18] = S.neighborMode;
  pU[19] = S.debugView;
  pF[20] = S.splatRadius;
  gpu.device.queue.writeBuffer(gpu.paramsBuf, 0, pF);
}

function writeRender(interpolationAlpha) {
  rF[0] = canvas.width;
  rF[1] = canvas.height;
  rF[2] = 1 / canvas.width;
  rF[3] = 1 / canvas.height;
  rF[4] = S.normalStrength;
  rF[5] = S.densityThreshold;
  rF[6] = S.specular;
  rF[7] = S.fresnel;
  rF[8] = S.subsurface;
  rF[9] = performance.now() * 0.001;
  rF[10] = S.debugView;
  rF[11] = S.spacing * canvas.width / WORLD_WIDTH;
  rF[12] = interpolationAlpha;
  gpu.device.queue.writeBuffer(gpu.renderBuf, 0, rF);
}

/* ============================================================================
   Frame loop
   ============================================================================ */
let lastT = performance.now();
let fpsAcc = 0, fpsFrames = 0, msSmooth = 0, sampledGpuMs = null;
let simHzTicks = 0;
let colorIndex = 0;
const simClock = { accumulator: 0, droppedSeconds: 0 };

function reserveGpuTimingSlot() {
  const timer = gpu.gpuTimer;
  if (!timer) return null;
  for (let offset = 0; offset < timer.slots.length; offset++) {
    const index = (timer.nextSlot + offset) % timer.slots.length;
    const slot = timer.slots[index];
    if (slot.busy) continue;
    slot.busy = true;
    timer.nextSlot = (index + 1) % timer.slots.length;
    return slot;
  }
  return null;
}

function readGpuTiming(slot) {
  if (!slot) return;
  slot.buffer.mapAsync(GPUMapMode.READ).then(() => {
    const view = new DataView(slot.buffer.getMappedRange());
    const start = view.getBigUint64(0, true);
    const end = view.getBigUint64(8, true);
    if (end >= start) sampledGpuMs = Number(end - start) / 1e6;
  }).catch(error => {
    console.warn('Could not read GPU timestamp sample.', error);
  }).finally(() => {
    if (slot.buffer.mapState === 'mapped') slot.buffer.unmap();
    slot.busy = false;
  });
}

function frame(now) {
  const rawDt = Math.max(0, (now - lastT) / 1000);
  lastT = now;

  fpsAcc += rawDt; fpsFrames++;
  if (fpsAcc >= 0.5) {
    $('fpsVal').textContent = Math.round(fpsFrames / fpsAcc);
    $('msVal').textContent  = msSmooth.toFixed(1);
    $('gpuMsVal').textContent = sampledGpuMs === null ? 'n/a' : sampledGpuMs.toFixed(2);
    $('simHzVal').textContent = (simHzTicks / fpsAcc).toFixed(0);
    $('droppedVal').textContent = simClock.droppedSeconds.toFixed(2);
    fpsAcc = 0; fpsFrames = 0;
    simHzTicks = 0;
  }

  $('seedVal').textContent = String(SIM_SEED);

  const t0 = performance.now();

  const schedule = advanceFixedClock(simClock, rawDt, S.timeScale);
  simHzTicks += schedule.ticks;
  const lastPtr = input.getLastPointer();

  const substeps = Math.max(1, Math.round(S.substeps));
  const subDt = FIXED_DT / substeps;
  writeParams(subDt);

  const enc = gpu.device.createCommandEncoder();
  const timingSlot = reserveGpuTimingSlot();
  let timingStarted = false;

  if (live > 0) {
    const wg = Math.ceil(live / 64);
    for (let tick = 0; tick < schedule.ticks; tick++) {
      const timingWrites = timingSlot && !timingStarted
        ? { timestampWrites: { querySet: gpu.gpuTimer.querySet, beginningOfPassWriteIndex: 0 } }
        : {};
      if (timingSlot && !timingStarted) timingStarted = true;
      let cp = enc.beginComputePass({ label: 'save interpolation positions', ...timingWrites });
      cp.setPipeline(gpu.pipeSavePrevious);
      cp.setBindGroup(0, gpu.physBGs[colorIndex]);
      cp.dispatchWorkgroups(wg);
      cp.end();

      const buildGridAndDensity = (label) => {
        if (S.neighborMode === 1) {
          let pass=enc.beginComputePass({label:`${label}: clear grid`});
          pass.setPipeline(gpu.pipeClearGrid);pass.setBindGroup(0,gpu.physBGs[colorIndex]);pass.dispatchWorkgroups(Math.ceil(pU[16]*pU[17]/64));pass.end();
          pass=enc.beginComputePass({label:`${label}: build grid`});
          pass.setPipeline(gpu.pipeBuildGrid);pass.setBindGroup(0,gpu.physBGs[colorIndex]);pass.dispatchWorkgroups(wg);pass.end();
        }
        const pass=enc.beginComputePass({label:`${label}: density`});
        pass.setPipeline(gpu.pipeDensity);pass.setBindGroup(0,gpu.physBGs[colorIndex]);pass.dispatchWorkgroups(wg);pass.end();
      };
      for (let step = 0; step < substeps; step++) {
        buildGridAndDensity('pre-step');
        cp=enc.beginComputePass({label:'physical forces'});
        cp.setPipeline(gpu.pipeForces);cp.setBindGroup(0,gpu.physBGs[colorIndex]);cp.dispatchWorkgroups(wg);cp.end();
        cp=enc.beginComputePass({label:'predict particle motion'});
        cp.setPipeline(gpu.pipeIntegrate);cp.setBindGroup(0,gpu.physBGs[colorIndex]);cp.dispatchWorkgroups(wg);cp.end();
        for (let iteration=0;iteration<S.solverIterations;iteration++) {
          buildGridAndDensity(`constraint ${iteration+1}`);
          cp=enc.beginComputePass({label:'PBF density multipliers'});
          cp.setPipeline(gpu.pipeLambdas);cp.setBindGroup(0,gpu.physBGs[colorIndex]);cp.dispatchWorkgroups(wg);cp.end();
          cp=enc.beginComputePass({label:'PBF position corrections'});
          cp.setPipeline(gpu.pipeCorrections);cp.setBindGroup(0,gpu.physBGs[colorIndex]);cp.dispatchWorkgroups(wg);cp.end();
          cp=enc.beginComputePass({label:'apply PBF corrections'});
          cp.setPipeline(gpu.pipeApplyCorrections);cp.setBindGroup(0,gpu.physBGs[colorIndex]);cp.dispatchWorkgroups(wg);cp.end();
        }
        buildGridAndDensity('post-constraint');
        cp=enc.beginComputePass({label:'dye diffusion'});
        cp.setPipeline(gpu.pipeColors);cp.setBindGroup(0,gpu.physBGs[colorIndex]);cp.dispatchWorkgroups(wg);cp.end();
        colorIndex = 1 - colorIndex;
      }
    }
  }

  {
    const timingWrites = timingSlot && !timingStarted
      ? { timestampWrites: { querySet: gpu.gpuTimer.querySet, beginningOfPassWriteIndex: 0 } }
      : {};
    if (timingSlot && !timingStarted) timingStarted = true;
    const rp = enc.beginRenderPass({
      ...timingWrites,
      colorAttachments: [{
        view: gpu.accumView,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: 'clear', storeOp: 'store',
      }],
    });
    if (live > 0) {
      rp.setPipeline(gpu.splatPipe);
      rp.setBindGroup(0, gpu.splatBGs[colorIndex]);
      rp.draw(6, live, 0, 0);
    }
    rp.end();
  }

  writeRender(schedule.alpha);

  {
    const timestampWrites = timingSlot
      ? {
          querySet: gpu.gpuTimer.querySet,
          ...(timingStarted ? {} : { beginningOfPassWriteIndex: 0 }),
          endOfPassWriteIndex: 1,
        }
      : undefined;
    const rp = enc.beginRenderPass({
      ...(timestampWrites ? { timestampWrites } : {}),
      colorAttachments: [{
        view: gpu.ctx.getCurrentTexture().createView(),
        clearValue: { r: 0.02, g: 0.03, b: 0.06, a: 1 },
        loadOp: 'clear', storeOp: 'store',
      }],
    });
    rp.setPipeline(gpu.compPipe);
    rp.setBindGroup(0, gpu.compBG);
    rp.draw(3, 1, 0, 0);
    rp.end();
  }

  if (timingSlot) {
    enc.resolveQuerySet(gpu.gpuTimer.querySet, 0, 2, gpu.gpuTimer.resolveBuffer, 0);
    enc.copyBufferToBuffer(gpu.gpuTimer.resolveBuffer, 0, timingSlot.buffer, 0, 16);
  }

  gpu.device.queue.submit([enc.finish()]);
  msSmooth = msSmooth * 0.9 + (performance.now() - t0) * 0.1;
  readGpuTiming(timingSlot);

  requestAnimationFrame(frame);
}

/* ---------- seed & go ---------- */
spawn(gpu.canvasW * 0.50,  0.10, 120, randomVibrant());
spawn(gpu.canvasW * 0.30, 0.16, 100, randomVibrant());
spawn(gpu.canvasW * 0.70, 0.16, 100, randomVibrant());

requestAnimationFrame((t) => { lastT = t; frame(t); });
