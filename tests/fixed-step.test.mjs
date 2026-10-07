import assert from 'node:assert/strict';
import { advanceFixedClock, FIXED_DT } from '../src/fixed-step.mjs';

function runAtRenderRate(fps, seconds, timeScale = 1) {
  const clock = { accumulator: 0, droppedSeconds: 0 };
  let ticks = 0;
  const frames = Math.round(fps * seconds);
  for (let i = 0; i < frames; i++) {
    ticks += advanceFixedClock(clock, 1 / fps, timeScale).ticks;
  }
  return { ticks, clock };
}

for (const fps of [30, 60, 144]) {
  const result = runAtRenderRate(fps, 10);
  assert.equal(result.ticks, 600, `${fps} FPS should produce 60 physics ticks per second`);
  assert.ok(result.clock.accumulator >= 0 && result.clock.accumulator < FIXED_DT);
  assert.equal(result.clock.droppedSeconds, 0);
}

assert.equal(runAtRenderRate(30, 1, 2).ticks, 120);

const overloadedClock = { accumulator: 0, droppedSeconds: 0 };
const overloaded = advanceFixedClock(overloadedClock, 0.25, 2);
assert.equal(overloaded.ticks, 8, 'catch-up work should be capped per rendered frame');
assert.ok(overloaded.droppedSeconds > 0, 'discarded simulation time should be observable');
assert.ok(overloaded.alpha >= 0 && overloaded.alpha < 1);

console.log('Fixed-step clock tests passed at 30, 60, and 144 FPS.');
