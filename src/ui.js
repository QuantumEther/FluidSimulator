import { MAX_PART, SLIDERS, random01, randomVibrant } from './config.js';

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
    Math.max(8, Math.min(canvasW - 8, (cx - r.left) * (canvasW / (r.width  || 1)))),
    Math.max(8, Math.min(canvasH - 8, (cy - r.top ) * (canvasH / (r.height || 1)))),
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
  if (dx*dx + dy*dy > 144) { spawn(x, y, 8, activeColor); lastSpawn = [x, y]; }
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


export function setupControls({ S, $, trimToBudget, getLive, setLive, upload, spawn, pick, getCanvasSize }) {
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
      <input type="range" id="sl-${sl.k}" min="${sl.min}" max="${sl.max}" step="${sl.step}" value="${S[sl.k]}">
    `;
    host.appendChild(row);
    const inp = row.querySelector('input');
    inp.addEventListener('input', (e) => {
      S[sl.k] = parseFloat(e.target.value);
      $('val-' + sl.k).textContent = sl.fmt(S[sl.k]);
      $('presetSelect').value = 'custom';
      if (sl.k === 'budget') trimToBudget();
    });
  }
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
    budget:8000, timeScale:1, substeps:4, gravity:900, viscosity:1.16,
    surfaceTension:0.94, stiffness:100, restDensity:2, hScale:0.5, damping:1,
    bounce:0.2, wallFriction:0.85, blobRadius:8, densityThreshold:0.75, normalStrength:300, specular:2.8,
    fresnel:0.5, subsurface:0.35, colorMix:0.05, neighborMode:1, debugView:0,
  },
  water: {
    budget:8000, timeScale:1, substeps:3, gravity:900, viscosity:0.55,
    surfaceTension:0.35, stiffness:800, restDensity:4, hScale:0.9, damping:0.998,
    bounce:0.2, wallFriction:0.92, blobRadius:18, densityThreshold:0.45,
    normalStrength:100, specular:1.2, fresnel:0.5, subsurface:0.65, colorMix:0.25,
    neighborMode:1, debugView:0,
  },
  honey: {
    budget:8000, timeScale:1, substeps:3, gravity:900, viscosity:2.4,
    surfaceTension:0.45, stiffness:300, restDensity:3, hScale:0.75, damping:0.999,
    bounce:0.1, wallFriction:0.97, blobRadius:20, densityThreshold:0.55,
    normalStrength:120, specular:1.2, fresnel:0.4, subsurface:1, colorMix:0.15,
    neighborMode:1, debugView:0,
  },
  lowGravity: {
    budget:8000, timeScale:1, substeps:3, gravity:150, viscosity:0.35,
    surfaceTension:0.2, stiffness:500, restDensity:2.5, hScale:0.7, damping:0.998,
    bounce:0.6, wallFriction:0.95, blobRadius:15, densityThreshold:0.5,
    normalStrength:120, specular:1.2, fresnel:0.5, subsurface:0.7, colorMix:0.2,
    neighborMode:1, debugView:0,
  },
};
presetSelect.value = 'screenshot';
updateModeStatus();
presetSelect.addEventListener('change', () => {
  const preset = presets[presetSelect.value];
  if (!preset) return;
  Object.assign(S, preset);
  for (const sl of SLIDERS) {
    const input = $('sl-' + sl.k);
    if (!input) continue;
    input.value = S[sl.k];
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
  let remaining = need;
  const clusters = 4;
  for (let c = 0; c < clusters && remaining > 0; c++) {
    const x = 80 + random01() * Math.max(1, getCanvasSize().width - 160);
    const y = 60 + random01() * Math.max(1, getCanvasSize().height * 0.3);
    const chunk = Math.min(remaining, Math.ceil(need / clusters));
    spawn(x, y, chunk, pick());
    remaining -= chunk;
  }
};
$('spawnBtn').onclick = () => {
  const x = 80 + random01() * Math.max(1, getCanvasSize().width - 160);
  const y = 60 + random01() * Math.max(1, getCanvasSize().height * 0.3);
  spawn(x, y, 80, pick());
};


}
