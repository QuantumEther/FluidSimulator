export const FIXED_DT = 1 / 60;
export const MAX_FRAME_DELTA = 0.25;
export const MAX_FIXED_TICKS_PER_FRAME = 8;

export function advanceFixedClock(clock, frameDelta, timeScale = 1) {
  const safeDelta = Math.max(0, Number.isFinite(frameDelta) ? frameDelta : 0);
  const safeScale = Math.max(0, Number.isFinite(timeScale) ? timeScale : 0);
  const acceptedDelta = Math.min(safeDelta, MAX_FRAME_DELTA);
  let droppedSeconds = (safeDelta - acceptedDelta) * safeScale;

  clock.accumulator += acceptedDelta * safeScale;
  const due = Math.floor((clock.accumulator + 1e-9) / FIXED_DT);
  const ticks = Math.min(due, MAX_FIXED_TICKS_PER_FRAME);
  clock.accumulator -= ticks * FIXED_DT;

  const droppedTicks = Math.max(0, due - ticks);
  if (droppedTicks) {
    const droppedTickTime = droppedTicks * FIXED_DT;
    clock.accumulator -= droppedTickTime;
    droppedSeconds += droppedTickTime;
  }

  // Avoid a tiny negative remainder caused by floating-point rounding.
  clock.accumulator = Math.max(0, clock.accumulator);
  clock.droppedSeconds = (clock.droppedSeconds || 0) + droppedSeconds;

  return {
    ticks,
    alpha: Math.min(clock.accumulator / FIXED_DT, 1),
    droppedSeconds,
  };
}
