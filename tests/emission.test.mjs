import assert from 'node:assert/strict';
import { particleCapacity, emissionPatch, WALL_MARGIN } from '../src/emission.mjs';

assert.equal(particleCapacity(0.6, 0.6, 0.006, 50000), 9604);
assert.equal(particleCapacity(0.6, 0.6, 0.002, 50000), 50000);
for (const spacing of [0.002, 0.004, 0.006]) {
  for (const count of [8, 80, 8000, 50000]) {
    for (const [x, y] of [[0, 0], [0.6, 0.6], [0.3, 0.3]]) {
      const patch = emissionPatch({width:0.6,height:0.6,spacing,x,y,count});
      assert.ok(patch.count <= particleCapacity(0.6,0.6,spacing,50000));
      assert.ok(patch.x >= WALL_MARGIN && patch.y >= WALL_MARGIN);
      assert.ok(patch.x + (patch.columns - 1) * spacing <= 0.6 - WALL_MARGIN);
      assert.ok(patch.y + (Math.ceil(patch.count / patch.columns) - 1) * spacing <= 0.6 - WALL_MARGIN);
    }
  }
}
assert.equal(emissionPatch({width:0.6,height:0.6,spacing:0.002,x:0,y:0,count:0}), null);
console.log('Emission capacity and fitting lattice tests passed.');
