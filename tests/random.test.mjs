import assert from 'node:assert/strict';
import { createRandom } from '../src/random.mjs';

const firstRun = createRandom(987654321);
const replay = createRandom(987654321);
const differentRun = createRandom(123456789);
const firstValues = Array.from({ length: 20 }, () => firstRun());
const replayValues = Array.from({ length: 20 }, () => replay());
const differentValues = Array.from({ length: 20 }, () => differentRun());

assert.deepEqual(replayValues, firstValues, 'matching seeds should replay the random sequence');
assert.notDeepEqual(differentValues, firstValues, 'different seeds should produce a different sequence');
assert.ok(firstValues.every(value => value >= 0 && value < 1));

console.log('Seeded random sequence tests passed.');
