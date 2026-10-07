
/* ============================================================================
   WebGPU Fluid Lab — realism upgrade
   ============================================================================ */
import { MAX_PART, STRIDE, BASE_H, S, randomVibrant } from './config.js';
import { createGpuRuntime } from './gpu.js';
import { setupInput, setupControls } from './ui.js';

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
let live = 0;

function writeP(i, x, y, vx, vy, r, g, b) {
  const o = i * STRIDE;
  stage[o+0]=x; stage[o+1]=y; stage[o+2]=vx; stage[o+3]=vy;
  stage[o+4]=r; stage[o+5]=g; stage[o+6]=b; stage[o+7]=1;
  stage[o+8]=0; stage[o+9]=0; stage[o+10]=0; stage[o+11]=0;
  const c = i * 4;
  colorStage[c]=r; colorStage[c+1]=g; colorStage[c+2]=b; colorStage[c+3]=1;
}
function upload(from = 0) {
  if (live <= from) return;
  gpu.device.queue.writeBuffer(gpu.particleBuf, from * STRIDE * 4, stage, from * STRIDE, (live - from) * STRIDE);
  gpu.device.queue.writeBuffer(gpu.colorBuffers[0], from * 16, colorStage, from * 4, (live - from) * 4);
  gpu.device.queue.writeBuffer(gpu.colorBuffers[1], from * 16, colorStage, from * 4, (live - from) * 4);
}

/* ---------- spawning ---------- */
function spawn(px, py, n, col) {
  const cap = Math.min(S.budget, MAX_PART);
  if (live >= cap) return;
  n = Math.min(n, cap - live);
  const firstNew = live;
  const R = Math.max(20, gpu.canvasW / 30);
  for (let k = 0; k < n; k++) {
    const a = Math.random() * Math.PI * 2;
    const r = Math.sqrt(Math.random()) * R;
    const x = Math.max(15, Math.min(gpu.canvasW - 15, px + Math.cos(a) * r));
    const y = Math.max(15, Math.min(gpu.canvasH - 15, py + Math.sin(a) * r));
    writeP(live + k, x, y,
      (Math.random()-0.5) * 25,
      Math.random() * 20 + 10,
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

function writeRender() {
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
  gpu.device.queue.writeBuffer(gpu.renderBuf, 0, rF);
}

/* ============================================================================
   Frame loop
   ============================================================================ */
let lastT = performance.now();
let fpsAcc = 0, fpsFrames = 0, msSmooth = 0;
let colorIndex = 0;

function frame(now) {
  const rawDt = Math.min((now - lastT) / 1000, 0.05);
  lastT = now;

  fpsAcc += rawDt; fpsFrames++;
  if (fpsAcc >= 0.5) {
    $('fpsVal').textContent = Math.round(fpsFrames / fpsAcc);
    $('msVal').textContent  = msSmooth.toFixed(1);
    fpsAcc = 0; fpsFrames = 0;
  }

  const t0 = performance.now();

  const lastPtr = input.getLastPointer();
  if (input.isDown() && live < Math.min(S.budget, MAX_PART) && lastPtr[0] >= 0)
    spawn(lastPtr[0], lastPtr[1], 3, input.getActiveColor());

  const dt = Math.min(rawDt * S.timeScale, 1/20);
  const steps = Math.max(1, Math.round(S.substeps));
  const subDt = dt / steps;

  const enc = gpu.device.createCommandEncoder();

  if (live > 0) {
    const wg = Math.ceil(live / 64);
    for (let s = 0; s < steps; s++) {
      writeParams(subDt);
      let cp = enc.beginComputePass();
      if (S.neighborMode === 1) {
        cp.setPipeline(gpu.pipeClearGrid); cp.setBindGroup(0, gpu.physBGs[colorIndex]); cp.dispatchWorkgroups(Math.ceil(pU[17] * pU[18] / 64));
        cp.setPipeline(gpu.pipeBuildGrid); cp.setBindGroup(0, gpu.physBGs[colorIndex]); cp.dispatchWorkgroups(wg);
      }
      cp.setPipeline(gpu.pipeDensity); cp.setBindGroup(0, gpu.physBGs[colorIndex]); cp.dispatchWorkgroups(wg);
      cp.setPipeline(gpu.pipeColors); cp.setBindGroup(0, gpu.physBGs[colorIndex]); cp.dispatchWorkgroups(wg);
      cp.end();
      colorIndex = 1 - colorIndex;
      cp = enc.beginComputePass();
      cp.setPipeline(gpu.pipeForces);    cp.setBindGroup(0, gpu.physBGs[colorIndex]); cp.dispatchWorkgroups(wg); cp.end();
      cp = enc.beginComputePass();
      cp.setPipeline(gpu.pipeIntegrate); cp.setBindGroup(0, gpu.physBGs[colorIndex]); cp.dispatchWorkgroups(wg); cp.end();
    }
  }

  {
    const rp = enc.beginRenderPass({
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

  writeRender();

  {
    const rp = enc.beginRenderPass({
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

  gpu.device.queue.submit([enc.finish()]);
  msSmooth = msSmooth * 0.9 + (performance.now() - t0) * 0.1;

  requestAnimationFrame(frame);
}

/* ---------- seed & go ---------- */
spawn(gpu.canvasW * 0.50,  80, 120, randomVibrant());
spawn(gpu.canvasW * 0.30, 140, 100, randomVibrant());
spawn(gpu.canvasW * 0.70, 140, 100, randomVibrant());

requestAnimationFrame((t) => { lastT = t; frame(t); });
