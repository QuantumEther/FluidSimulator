import { MAX_PART, SLIDERS, randomVibrant } from './config.js';

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
      if (sl.k === 'budget') trimToBudget();
    });
  }
}
buildUI();

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
    const x = 80 + Math.random() * Math.max(1, getCanvasSize().width - 160);
    const y = 60 + Math.random() * Math.max(1, getCanvasSize().height * 0.3);
    const chunk = Math.min(remaining, Math.ceil(need / clusters));
    spawn(x, y, chunk, pick());
    remaining -= chunk;
  }
};
$('spawnBtn').onclick = () => {
  const x = 80 + Math.random() * Math.max(1, getCanvasSize().width - 160);
  const y = 60 + Math.random() * Math.max(1, getCanvasSize().height * 0.3);
  spawn(x, y, 80, pick());
};


}
