import { MAX_PART, SLIDERS, random01, randomVibrant } from './config.js?v=si-units-v4';

export function setupInput({ canvas, getCanvasSize, spawn, S }) {
let down = false, lastSpawn = [0,0], lastPtr = [-1,-1];
let activeColor = randomVibrant();

function pos(e, getCanvasSize) {
  const r = canvas.getBoundingClientRect();
  const { width: canvasW, height: canvasH } = getCanvasSize();
  let cx, cy;
  if (e.touches && e.touches.length) { cx = e.touches[0].clientX; cy = e.touches[0].clientY; }
  else if (e.changedTouches && e.changedTouches.length) { cx = e.changedTouches[0].clientX; cy = e.changedTouches[0].clientY; }
  else { cx = e.clientX; cy = e.clientY; }
  return [
    Math.max(0.006, Math.min(canvasW - 0.006, (cx - r.left) * (canvasW / (r.width  || 1)))),
    Math.max(0.006, Math.min(canvasH - 0.006, (cy - r.top ) * (canvasH / (r.height || 1)))),
  ];
}
const pick = () => S.randomColor ? randomVibrant() : S.customColor;

function onDown(e) {
  e.preventDefault();
  down = true;
  const [x, y] = pos(e, getCanvasSize);
  lastSpawn = [x, y]; lastPtr = [x, y];
  activeColor = pick();
  spawn(x, y, 50, activeColor);
}
function onMove(e) {
  const [x, y] = pos(e, getCanvasSize);
  lastPtr = [x, y];
  if (!down) return;
  const dx = x - lastSpawn[0], dy = y - lastSpawn[1];
  if (dx*dx + dy*dy > 0.0001) { spawn(x, y, 8, activeColor); lastSpawn = [x, y]; }
}
const onUp = () => { down = false; };

canvas.addEventListener('mousedown', onDown);
window.addEventListener('mousemove', onMove);
window.addEventListener('mouseup',   onUp);
canvas.addEventListener('touchstart', onDown, { passive: false });
window.addEventListener('touchmove',  onMove, { passive: false });
window.addEventListener('touchend',   onUp);
window.addEventListener('touchcancel',onUp);
canvas.addEventListener('contextmenu', e => e.preventDefault());

  return { isDown: () => down, getLastPointer: () => lastPtr, getActiveColor: () => activeColor };
}


export function setupControls({ S, $, trimToBudget, getLive, setLive, upload, spawn, pick, getCanvasSize, refreshParticleMasses }) {
function hexRgb01(hex) {
  const n = parseInt(hex.replace('#',''), 16);
  return [((n>>16)&255)/255, ((n>>8)&255)/255, (n&255)/255];
}

/* ============================================================================
   Dynamic UI construction
   ============================================================================ */
function buildUI() {
  const groups = { time: $('group-time'), physics: $('group-physics'),
                   boundary: $('group-boundary'), render: $('group-render'),
                   color: $('group-color') };
  for (const sl of SLIDERS) {
    const host = groups[sl.g];
    if (!host) continue;
    const row = document.createElement('div');
    row.innerHTML = `
      <div class="flex justify-between text-[11px] mb-1">
        <span class="text-slate-300">${sl.l}</span>
        <span class="text-indigo-400 font-mono" id="val-${sl.k}">${sl.fmt(S[sl.k])}</span>
      </div>
      <input type="range" id="sl-${sl.k}" min="${sl.scale === 'log' ? 0 : sl.min}" max="${sl.scale === 'log' ? 1000 : sl.max}" step="${sl.scale === 'log' ? 1 : sl.step}" value="${sliderValue(sl,S[sl.k])}">
    `;
    host.appendChild(row);
    const inp = row.querySelector('input');
    inp.addEventListener('input', (e) => {
      S[sl.k] = sliderActual(sl,parseFloat(e.target.value));
      $('val-' + sl.k).textContent = sl.fmt(S[sl.k]);
      $('presetSelect').value = 'custom';
      if (sl.k === 'budget') trimToBudget();
      if (sl.k === 'spacing' || sl.k === 'restDensity') refreshParticleMasses();
    });
  }
}
function sliderValue(sl,value) {
  if(sl.scale!=='log') return value;
  return (Math.log10(value)-Math.log10(sl.min))/(Math.log10(sl.max)-Math.log10(sl.min))*1000;
}
function sliderActual(sl,value) {
  if(sl.scale!=='log') return value;
  return 10**(Math.log10(sl.min)+value/1000*(Math.log10(sl.max)-Math.log10(sl.min)));
}
buildUI();

const neighborMode = $('neighborMode');
neighborMode.value = String(S.neighborMode);
const debugView = $('debugView');
debugView.value = String(S.debugView);
const presetSelect = $('presetSelect');
function updateModeStatus() {
  $('modeStatus').textContent = `Search: ${neighborMode.selectedOptions[0].textContent} · View: ${debugView.selectedOptions[0].textContent}`;
}
neighborMode.addEventListener('change', () => {
  S.neighborMode = Number(neighborMode.value);
  presetSelect.value = 'custom';
  updateModeStatus();
});
debugView.addEventListener('change', () => {
  S.debugView = Number(debugView.value);
  presetSelect.value = 'custom';
  updateModeStatus();
});

const presets = {
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
presetSelect.value = 'water';
updateModeStatus();
presetSelect.addEventListener('change', () => {
  const preset = presets[presetSelect.value];
  if (!preset) return;
  Object.assign(S, preset);
  refreshParticleMasses();
  for (const sl of SLIDERS) {
    const input = $('sl-' + sl.k);
    if (!input) continue;
    input.value = sliderValue(sl,S[sl.k]);
    $('val-' + sl.k).textContent = sl.fmt(S[sl.k]);
  }
  neighborMode.value = String(S.neighborMode);
  debugView.value = String(S.debugView);
  updateModeStatus();
  trimToBudget();
});

/* Buttons & colour */
$('randBtn').onclick = () => { S.randomColor = true;
  $('randBtn').className = 'px-3 py-2 rounded-xl text-xs font-medium border border-indigo-500/50 bg-indigo-500/25 text-indigo-200'; };
$('colorPick').oninput = e => { S.randomColor = false;
  S.customColor = hexRgb01(e.target.value);
  $('randBtn').className = 'px-3 py-2 rounded-xl text-xs font-medium border border-white/10 bg-slate-800 text-slate-400'; };
$('resetBtn').onclick = () => { setLive(0); upload(); $('countVal').textContent = '0'; };
$('fillBtn').onclick = () => {
  const live = getLive();
  const target = Math.min(S.budget, MAX_PART);
  const need = target - live;
  if (need <= 0) return;
  spawn(getCanvasSize().width*0.5,getCanvasSize().height*0.48,need,pick());
};
$('spawnBtn').onclick = () => {
  const x = 0.1 + random01() * Math.max(0.01, getCanvasSize().width - 0.2);
  const y = 0.08 + random01() * Math.max(0.01, getCanvasSize().height * 0.3);
  spawn(x, y, 80, pick());
};


}
