import assert from 'node:assert/strict';
import { encodeSimulation } from '../src/solver.mjs';
import { usesImplicitViscosity } from '../src/viscosity.mjs';

function run(settings, { valid = true, count = 320, ticks = 1, colorIndex = 0 } = {}) {
  const calls = [];
  const pass = {setPipeline(p){calls.push(p);},setBindGroup(){},dispatchWorkgroups(){}};
  const gpu = new Proxy({physBGs:[{},{}]}, {get:(o,k)=>o[k] ?? k});
  const stats = {dispatches:0,neighborSweeps:0,gridBuilds:0};
  const result = encodeSimulation({pass,gpu,settings,count,cells:22500,ticks,colorIndex,valid,stats});
  return {calls,stats,result};
}
const settings = {neighborMode:1,substeps:4,solverIterations:3,diffusivity:1e-9,viscosity:0.001,restDensity:1000,spacing:0.002};
const warm = run(settings);
assert.equal(warm.stats.neighborSweeps, 32);
assert.equal(warm.stats.gridBuilds,16);
assert.equal(warm.stats.dispatches,81); // Small scenes avoid scan overhead.
assert.equal(run(settings,{valid:false}).stats.dispatches,84);
assert.equal(run({...settings,gridLayout:'contiguous'}).stats.dispatches,129);
assert.equal(run(settings,{ticks:2}).stats.neighborSweeps,64);
const reference = run({...settings,referenceSchedule:true});
assert.equal(reference.stats.neighborSweeps,52);
assert.equal(reference.stats.gridBuilds,20);
assert.equal(reference.stats.dispatches,109);
const zeroDye = run({...settings,substeps:1,diffusivity:0},{colorIndex:1});
assert.equal(zeroDye.result.colorIndex,1,'zero diffusion keeps the authoritative buffer');
assert.ok(!zeroDye.calls.includes('pipeDensityColors'));
assert.equal(run({...settings,substeps:1}).result.colorIndex,1);
assert.equal(run(settings,{count:0}).stats.dispatches,0);
assert.equal(run(settings,{ticks:0,valid:false}).result.valid,false);
const viscous = run({...settings,viscosity:2});
assert.equal(viscous.stats.neighborSweeps,64);
assert.ok(viscous.calls.indexOf('pipeApplyViscosity') < viscous.calls.indexOf('pipeForces'));
assert.ok(!usesImplicitViscosity({...settings,viscosity:0.25}));
assert.ok(usesImplicitViscosity({...settings,viscosity:0.25,substeps:1}));
assert.ok(!usesImplicitViscosity({...settings,viscosity:2,restDensity:1400,spacing:0.006}));
assert.ok(warm.calls.indexOf('pipeDensityLambda') < warm.calls.indexOf('pipeCorrections'));
assert.ok(warm.calls.indexOf('pipeCorrections') < warm.calls.indexOf('pipeApplyCorrections'));
console.log('Solver scheduling, invalidation, and dye buffer tests passed.');
