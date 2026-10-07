export function createRandom(seed) {
  // xorshift32: compact deterministic stream for repeatable spawn layouts.
  let state = (Number(seed) >>> 0) || 1;
  return function random01() {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x100000000;
  };
}
