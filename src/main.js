
/* ============================================================================
   WebGPU Fluid Lab — realism upgrade
   ============================================================================ */
import { MAX_PART, STRIDE, BASE_H, S, SIM_SEED, random01, randomVibrant } from './config.js';
import { createGpuRuntime } from './gpu.js';
import { setupInput, setupControls } from './ui.js';
import { advanceFixedClock, FIXED_DT } from './fixed-step.mjs';

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
  stage[o+8]=0; stage[o+9]=0; stage[o+10]=0; stage[o+11]=0;
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
  const R = Math.max(20, gpu.canvasW / 30);
  for (let k = 0; k < n; k++) {
    const a = random01() * Math.PI * 2;
    const r = Math.sqrt(random01()) * R;
    const x = Math.max(15, Math.min(gpu.canvasW - 15, px + Math.cos(a) * r));
    const y = Math.max(15, Math.min(gpu.canvasH - 15, py + Math.sin(a) * r));
    writeP(live + k, x, y,
      (random01()-0.5) * 25,
      random01() * 20 + 10,
      col[0], col[1], col[2]);
  }
  live += n;
  upload(firstNew);
  $('countVal').textContent = live.toLocaleString();
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
});

/* ============================================================================
   Uniform writers
   ============================================================================ */
const pF = new Float32Array(32);
const pU = new Uint32Array(pF.buffer);
const rF = new Float32Array(20);

function writeParams(dt) {
  const h = BASE_H * S.hScale;
  pF[0]  = 0;
  pF[1]  = S.gravity;
  pF[2]  = h;
  pF[3]  = h * h;
  pF[4]  = S.restDensity;
  pF[5]  = S.stiffness;
  pF[6]  = S.viscosity;
  pF[7]  = S.surfaceTension;
  pF[8]  = S.damping;
  pF[9]  = S.colorMix;
  pF[10] = S.bounce;
  pF[11] = S.wallFriction;
  pF[12] = dt;
  pF[13] = performance.now() * 0.001;
  pF[14] = gpu.canvasW;
  pF[15] = gpu.canvasH;
  pU[16] = live;
  pU[17] = Math.ceil(gpu.canvasW / h);
  pU[18] = Math.ceil(gpu.canvasH / h);
  pU[19] = S.neighborMode;
  pU[20] = S.debugView;
  pF[21] = S.blobRadius;
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
  rF[11] = BASE_H * S.hScale * canvas.width / gpu.canvasW;
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
  if (schedule.ticks > 0 && input.isDown() && live < Math.min(S.budget, MAX_PART) && lastPtr[0] >= 0)
    spawn(lastPtr[0], lastPtr[1], 3 * schedule.ticks, input.getActiveColor());

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

      for (let step = 0; step < substeps; step++) {
        cp = enc.beginComputePass({ label: 'grid, density, and color' });
        if (S.neighborMode === 1) {
          cp.setPipeline(gpu.pipeClearGrid); cp.setBindGroup(0, gpu.physBGs[colorIndex]); cp.dispatchWorkgroups(Math.ceil(pU[17] * pU[18] / 64));
          cp.setPipeline(gpu.pipeBuildGrid); cp.setBindGroup(0, gpu.physBGs[colorIndex]); cp.dispatchWorkgroups(wg);
        }
        cp.setPipeline(gpu.pipeDensity); cp.setBindGroup(0, gpu.physBGs[colorIndex]); cp.dispatchWorkgroups(wg);
        cp.setPipeline(gpu.pipeColors); cp.setBindGroup(0, gpu.physBGs[colorIndex]); cp.dispatchWorkgroups(wg);
        cp.end();
        colorIndex = 1 - colorIndex;
        cp = enc.beginComputePass({ label: 'forces' });
        cp.setPipeline(gpu.pipeForces); cp.setBindGroup(0, gpu.physBGs[colorIndex]); cp.dispatchWorkgroups(wg); cp.end();
        cp = enc.beginComputePass({ label: 'integrate positions' });
        cp.setPipeline(gpu.pipeIntegrate); cp.setBindGroup(0, gpu.physBGs[colorIndex]); cp.dispatchWorkgroups(wg); cp.end();
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
spawn(gpu.canvasW * 0.50,  80, 120, randomVibrant());
spawn(gpu.canvasW * 0.30, 140, 100, randomVibrant());
spawn(gpu.canvasW * 0.70, 140, 100, randomVibrant());

requestAnimationFrame((t) => { lastT = t; frame(t); });
