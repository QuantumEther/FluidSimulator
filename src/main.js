
/* ============================================================================
   WebGPU Fluid Lab — realism upgrade
   ============================================================================ */
import { MAX_PART, STRIDE, S, SIM_SEED, random01, randomVibrant, WORLD_WIDTH, WORLD_HEIGHT } from './config.js?v=si-smooth-flow-v7';
import { createGpuRuntime } from './gpu.js?v=si-smooth-flow-v7';
import { setupInput, setupControls } from './ui.js?v=si-smooth-flow-v7';
import { advanceFixedClock, FIXED_DT } from './fixed-step.mjs';
import { particleMassKg, particleMassPerDepth } from './si-units.mjs';
import { particleCapacity, emissionPatch, REFERENCE_LIMIT } from './emission.mjs';
import { encodeGrid, encodeSimulation, gridLayout } from './solver.mjs';
import { usesImplicitViscosity } from './viscosity.mjs';

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

/* Particle state stays on the GPU. The host queues small emission commands. */
let live = 0, colorIndex = 0, valid = false, massDirty = false;
let emissionPending = false, emissionEpoch = 0, emissionQueue = [];
let appliedSpacing = S.spacing;
let appliedSignature = '';
const metrics = { frames: 0, submittedTicks: 0, completedTicks: 0, dispatches: 0, neighborSweeps: 0,
  gridBuilds: 0, computePasses: 0, cpuMs: 0, stages: {}, completionMs: null, samples: [] };
let benchmarkPaused = false;

function capacity() { return particleCapacity(WORLD_WIDTH, WORLD_HEIGHT, S.spacing, Math.min(S.budget, MAX_PART)); }
function showCount() { $('countVal').textContent = live.toLocaleString(); }
function resetParticles() {
  live = 0; valid = false; emissionEpoch++; emissionQueue = []; showCount();
}
function spawn(x, y, count, color) {
  if (emissionQueue.length >= 32) return;
  emissionQueue.push({ x, y, count, color, seed: Math.floor(random01() * 0x100000000) });
}
function fill() {
  // Fill is an explicit replacement scene, rather than compressing new liquid into old liquid.
  resetParticles();
  spawn(WORLD_WIDTH / 2, WORLD_HEIGHT / 2, capacity(), S.randomColor ? randomVibrant() : S.customColor);
}
function refreshParticleMasses() {
  if (S.spacing !== appliedSpacing) {
    const area = live * appliedSpacing * appliedSpacing;
    appliedSpacing = S.spacing;
    resetParticles();
    if (area > 0) spawn(WORLD_WIDTH / 2, WORLD_HEIGHT / 2, Math.min(capacity(), Math.round(area / (S.spacing * S.spacing))), S.customColor);
  }
  massDirty = true; valid = false;
  const mass = particleMassKg(S.restDensity, S.spacing, S.sliceDepth);
  $('particleMassVal').textContent = mass < 0.001 ? `${(mass*1e6).toFixed(0)} mg` : `${(mass*1000).toFixed(2)} g`;
}
function trimToBudget() {
  if (live > capacity()) { live = capacity(); valid = false; showCount(); }
  if (live > REFERENCE_LIMIT && S.neighborMode === 0) {
    S.neighborMode = 1; $('neighborMode').value = '1';
    $('modeStatus').textContent = `Search: Spatial grid · View: ${$('debugView').selectedOptions[0].textContent}`;
  }
}
const getCanvasSize = () => ({ width: WORLD_WIDTH, height: WORLD_HEIGHT });
setupInput({ canvas, getCanvasSize, spawn, S });
setupControls({ S, $, trimToBudget, getLive: () => live, setLive: resetParticles,
  spawn, fill, pick: () => S.randomColor ? randomVibrant() : S.customColor,
  getCanvasSize, refreshParticleMasses, setRenderScale: value => gpu.setRenderScale(value) });
refreshParticleMasses();

/* ============================================================================
   Uniform writers
   ============================================================================ */
const pF = new Float32Array(32);
const pU = new Uint32Array(pF.buffer);
const rF = new Float32Array(20);

function writeParams(dt, sampleStats = false) {
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
  const invH8 = 1 / (h ** 8);
  pF[21] = 4 / Math.PI * invH8;
  pF[22] = -24 / Math.PI * invH8;
  pF[23] = 40 / Math.PI * invH8;
  pF[24] = 48 / Math.PI * invH8;
  pF[25] = pF[21] * (0.91 * h * h) ** 3;
  pF[26] = particleMassPerDepth(S.restDensity, S.spacing);
  pF[27] = (S.spacing / 0.002) ** 2;
  pF[28] = sampleStats ? 1 : 0;
  pF[29] = usesImplicitViscosity(S) ? 1 : 0;
  pF[30] = gridLayout(S, live) === 'contiguous' ? 1 : 0;
  pF[31] = S.sliceDepth;
  gpu.device.queue.writeBuffer(gpu.paramsBuf, 0, pF);
}

function writeRender(interpolationAlpha) {
  rF[0] = canvas.width;
  rF[1] = canvas.height;
  rF[2] = 1 / gpu.fieldWidth;
  rF[3] = 1 / gpu.fieldHeight;
  rF[4] = S.normalStrength;
  rF[5] = S.densityThreshold;
  rF[6] = S.specular;
  rF[7] = S.fresnel;
  rF[8] = S.subsurface;
  rF[9] = performance.now() * 0.001;
  rF[10] = S.debugView;
  rF[11] = 2 * S.spacing * canvas.width / WORLD_WIDTH;
  rF[12] = interpolationAlpha;
  gpu.device.queue.writeBuffer(gpu.renderBuf, 0, rF);
}

/* ============================================================================
   Frame loop
   ============================================================================ */
let lastT = performance.now();
let fpsAcc = 0, fpsFrames = 0, msSmooth = 0, sampledGpuMs = null;
let completedAtLastUi = 0, inFlight = 0, diagnosticPending = false;
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
    const stages = {};
    for (const [index, name] of ['compute', 'splat', 'composite'].entries()) {
      const start = view.getBigUint64(index * 16, true), end = view.getBigUint64(index * 16 + 8, true);
      stages[name] = end >= start ? Number(end - start) / 1e6 : 0;
    }
    metrics.stages = stages;
    sampledGpuMs = stages.compute + stages.splat + stages.composite;
    metrics.samples.push({ ...stages, cpuMs: slot.cpuMs, ticks: slot.ticks });
    if (metrics.samples.length > 600) metrics.samples.shift();
  }).catch(error => {
    console.warn('Could not read GPU timestamp sample.', error);
  }).finally(() => {
    if (slot.buffer.mapState === 'mapped') slot.buffer.unmap();
    slot.busy = false;
  });
}

function frame(now, forcedTicks = null) {
  if (gpu.lostReason) {
    $('startupErrorText').textContent = 'GPU connection lost. Reload the page to restart.';
    $('startupError').hidden = false;
    return;
  }
  if (benchmarkPaused && forcedTicks === null && !emissionQueue.length && !massDirty) {
    requestAnimationFrame(frame);
    return;
  }
  // Bound queue latency instead of continuing to enqueue an overloaded simulation.
  if (forcedTicks === null && inFlight >= 2) { requestAnimationFrame(frame); return; }
  const rawDt = Math.max(0, (now - lastT) / 1000);
  lastT = now;

  fpsAcc += rawDt; fpsFrames++;
  if (fpsAcc >= 0.5) {
    $('fpsVal').textContent = Math.round(fpsFrames / fpsAcc);
    $('msVal').textContent  = msSmooth.toFixed(1);
    $('gpuMsVal').textContent = sampledGpuMs === null ? 'n/a' : sampledGpuMs.toFixed(2);
    $('simHzVal').textContent = ((metrics.completedTicks - completedAtLastUi) / fpsAcc).toFixed(0);
    $('droppedVal').textContent = simClock.droppedSeconds.toFixed(2);
    fpsAcc = 0; fpsFrames = 0;
    completedAtLastUi = metrics.completedTicks;
  }

  $('seedVal').textContent = String(SIM_SEED);

  const t0 = performance.now();

  const schedule = forcedTicks === null ? advanceFixedClock(simClock, rawDt, benchmarkPaused ? 0 : S.timeScale) : { ticks: forcedTicks, alpha: 1 };
  trimToBudget();
  gpu.setRenderScale(S.renderScale);

  const substeps = Math.max(1, Math.round(S.substeps));
  const subDt = FIXED_DT / substeps;
  const sampleStats = live > 0 && !diagnosticPending && S.sampleDiagnostics !== false && metrics.frames % 60 === 0;
  writeParams(subDt, sampleStats);

  const signature = `${live}:${S.spacing}:${S.restDensity}:${S.neighborMode}:${gridLayout(S, live)}`;
  if (signature !== appliedSignature) valid = false;
  appliedSignature = signature;
  const enc = gpu.device.createCommandEncoder();
  const timingSlot = metrics.frames % 15 === 0 ? reserveGpuTimingSlot() : null;
  if (timingSlot) timingSlot.ticks = schedule.ticks;
  const timestampWrites = index => timingSlot ? { timestampWrites: { querySet: gpu.gpuTimer.querySet,
    beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 } } : {};
  const stats = { dispatches: 0, neighborSweeps: 0, gridBuilds: 0 };
  const cp = enc.beginComputePass({ label: 'ordered fluid simulation', ...timestampWrites(0) });
  const dispatch = (pipeline, groups, bind = gpu.physBGs[colorIndex]) => {
    cp.setPipeline(pipeline); cp.setBindGroup(0, bind); cp.dispatchWorkgroups(groups); stats.dispatches++;
  };
  if (sampleStats) dispatch(gpu.pipeClearStats, 1);
  if (massDirty && live) dispatch(gpu.pipeUpdateMasses, Math.ceil(live / 64));
  massDirty = false;
  let emitted = false, pendingEpoch = emissionEpoch;
  if (!emissionPending && emissionQueue.length) {
    const command = emissionQueue.shift();
    const patch = emissionPatch({ width: WORLD_WIDTH, height: WORLD_HEIGHT, spacing: S.spacing,
      x: command.x, y: command.y, count: Math.min(command.count, capacity() - live) });
    if (patch) {
      if (S.neighborMode === 1 && !valid) {
        encodeGrid(cp, gpu, live, pU[16] * pU[17], gpu.physBGs[colorIndex], stats, gridLayout(S, live));
      }
      const emissionData = new Float32Array(12), emissionU = new Uint32Array(emissionData.buffer);
      emissionData[0] = patch.x; emissionData[1] = patch.y; emissionData[2] = S.spacing;
      emissionU[3] = patch.count; emissionData.set([...command.color, 1], 4);
      emissionU[8] = patch.columns; emissionU[9] = command.seed;
      gpu.device.queue.writeBuffer(gpu.emissionBuf, 0, emissionData);
      dispatch(gpu.pipeClearEmission, 1, gpu.physBGs[0]);
      dispatch(gpu.pipeEmit, Math.ceil(patch.count / 64), gpu.physBGs[0]);
      dispatch(gpu.pipeCopyEmittedColors, Math.ceil(patch.count / 64), gpu.physBGs[1]);
      emitted = true; emissionPending = true;
    }
  }
  ({ colorIndex, valid } = encodeSimulation({ pass: cp, gpu, settings: S, count: live,
    cells: pU[16] * pU[17], ticks: schedule.ticks, colorIndex, valid, stats }));
  const collectedStats = sampleStats && valid;
  if (collectedStats) {
    dispatch(gpu.pipeComputeStats, Math.ceil(Math.max(live, pU[16] * pU[17]) / 64));
    dispatch(gpu.pipeReduceStats, 1);
  }
  cp.end();
  if (collectedStats) enc.copyBufferToBuffer(gpu.cellOffsetsBuf, 39280 * 4, gpu.statsReadback, 0, 40);
  if (emitted) enc.copyBufferToBuffer(gpu.cellCountsBuf, 65536 * 4, gpu.emissionReadback, 0, 4);
  const rp = enc.beginRenderPass({ label: 'particle field', ...timestampWrites(2),
    colorAttachments: [{ view: gpu.accumView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: 'clear', storeOp: 'store' }] });
  if (live) { rp.setPipeline(gpu.splatPipe); rp.setBindGroup(0, gpu.splatBGs[colorIndex]); rp.draw(6, live); }
  rp.end();

  writeRender(schedule.alpha);

  {
    const rp = enc.beginRenderPass({
      label: 'composite', ...timestampWrites(4),
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
    enc.resolveQuerySet(gpu.gpuTimer.querySet, 0, 6, gpu.gpuTimer.resolveBuffer, 0);
    enc.copyBufferToBuffer(gpu.gpuTimer.resolveBuffer, 0, timingSlot.buffer, 0, 48);
  }

  gpu.device.queue.submit([enc.finish()]);
  metrics.frames++; metrics.submittedTicks += schedule.ticks;
  Object.assign(metrics, stats, { computePasses: 1, cpuMs: performance.now() - t0 });
  if (timingSlot) timingSlot.cpuMs = metrics.cpuMs;
  msSmooth = msSmooth * 0.9 + metrics.cpuMs * 0.1;
  if (collectedStats) {
    diagnosticPending = true;
    const forceSamples = live * substeps * schedule.ticks;
    const correctionSamples = forceSamples * S.solverIterations;
    gpu.statsReadback.mapAsync(GPUMapMode.READ).then(() => {
      const data = new Float32Array(gpu.statsReadback.getMappedRange());
      metrics.diagnostics = { nonfiniteCount: data[0], maxSpeed: data[1], maxCompression: data[2], kineticEnergyJ: data[3],
        dyeMass: Array.from(data.slice(4, 7)), maxCellOccupancy: data[7],
        accelerationCapFraction: forceSamples ? data[8] / forceSamples : 0,
        correctionCapFraction: correctionSamples ? data[9] / correctionSamples : 0 };
    }).catch(error => console.warn('GPU diagnostic sample failed:', error)).finally(() => {
      if (gpu.statsReadback.mapState === 'mapped') gpu.statsReadback.unmap(); diagnosticPending = false;
    });
  }
  if (emitted) {
    gpu.emissionReadback.mapAsync(GPUMapMode.READ).then(() => {
      const accepted = new Uint32Array(gpu.emissionReadback.getMappedRange())[0];
      if (pendingEpoch === emissionEpoch) {
        live += accepted; trimToBudget(); valid = false;
        // A material edit can arrive while the accepted-count readback is pending.
        massDirty = true; showCount();
      }
    }).catch(error => console.error('Particle emission readback failed:', error)).finally(() => {
      if (gpu.emissionReadback.mapState === 'mapped') gpu.emissionReadback.unmap();
      emissionPending = false;
    });
  }
  inFlight++;
  const submitted = metrics.submittedTicks, completionStart = performance.now();
  const sampleCompletion = metrics.frames % 30 === 0 || forcedTicks !== null;
  gpu.device.queue.onSubmittedWorkDone().then(() => {
    metrics.completedTicks = Math.max(metrics.completedTicks, submitted);
    if (sampleCompletion) metrics.completionMs = performance.now() - completionStart;
  }).catch(error => console.warn('GPU completion failed:', error)).finally(() => { inFlight--; });
  readGpuTiming(timingSlot);

  if (forcedTicks === null) requestAnimationFrame(frame);
  return { ...stats, computePasses: 1, cpuMs: metrics.cpuMs };
}

/* ---------- seed & go ---------- */
spawn(gpu.canvasW * 0.50,  0.10, 120, randomVibrant());
spawn(gpu.canvasW * 0.30, 0.16, 100, randomVibrant());
spawn(gpu.canvasW * 0.70, 0.16, 100, randomVibrant());

requestAnimationFrame((t) => { lastT = t; frame(t); });

// Opt-in harness API; ordinary runs do not expose device resources or read particle state back.
if (new URLSearchParams(location.search).has('benchmark')) {
  benchmarkPaused = true;
  window.fluidLab = {
    settings: S, metrics, gpu,
    get count() { return live; }, get emissionPending() { return emissionPending || emissionQueue.length > 0; },
    pause(value = true) { benchmarkPaused = value; },
    async step(ticks = 1) { const result = frame(performance.now(), ticks); await gpu.device.queue.onSubmittedWorkDone(); return result; },
    async setScene(positions, velocities = [], colors = []) {
      if (emissionPending) await gpu.device.queue.onSubmittedWorkDone();
      while (emissionPending) await new Promise(resolve => setTimeout(resolve, 0));
      resetParticles();
      appliedSpacing = S.spacing;
      live = Math.min(positions.length, capacity());
      const data = new Float32Array(live * STRIDE), integers = new Uint32Array(data.buffer);
      const dye = new Float32Array(live * 4), previous = new Float32Array(live * 2);
      for (let i = 0; i < live; i++) {
        data.set([...positions[i], ...(velocities[i] || [0, 0])], i * STRIDE);
        data[i * STRIDE + 8] = S.restDensity; data[i * STRIDE + 10] = particleMassPerDepth(S.restDensity, S.spacing);
        integers[i * STRIDE + 11] = 1;
        dye.set([...(colors[i] || [0.2, 0.6, 0.9]), 1], i * 4); previous.set(positions[i], i * 2);
      }
      if (live) {
        gpu.device.queue.writeBuffer(gpu.particleBuf, 0, data);
        for (const buffer of gpu.colorBuffers) gpu.device.queue.writeBuffer(buffer, 0, dye);
        gpu.device.queue.writeBuffer(gpu.previousPositionsBuf, 0, previous);
      }
      colorIndex = 0; showCount();
    },
    async readState() {
      const sizes = [live * STRIDE * 4, live * 16, live * 8];
      const sources = [gpu.particleBuf, gpu.colorBuffers[colorIndex], gpu.previousPositionsBuf];
      const result = [];
      for (let index = 0; index < sources.length; index++) {
        if (!sizes[index]) { result.push([]); continue; }
        const buffer = gpu.device.createBuffer({size:sizes[index],usage:GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
        const encoder = gpu.device.createCommandEncoder(); encoder.copyBufferToBuffer(sources[index],0,buffer,0,sizes[index]);
        gpu.device.queue.submit([encoder.finish()]); await buffer.mapAsync(GPUMapMode.READ);
        result.push(Array.from(new Float32Array(buffer.getMappedRange()))); buffer.unmap(); buffer.destroy();
      }
      return { particles: result[0], colors: result[1], previous: result[2] };
    },
    spawn, fill, refreshParticleMasses,
  };
}
