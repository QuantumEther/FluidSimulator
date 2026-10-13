import assert from 'node:assert/strict';
import { particleMassKg, particleMassPerDepth, poly6_2D } from '../src/si-units.mjs';

const rho = 1000, dx = 0.002, depth = 0.1;
assert.equal(particleMassPerDepth(rho, dx), 0.004, 'particle column mass is kg per metre of depth');
assert.equal(particleMassKg(rho, dx, depth), 0.0004, 'represented particle mass is 0.4 g');
assert.equal(poly6_2D(0.004, 0.004), 0, 'kernel has finite support');
assert.ok(poly6_2D(0, 0.004) > 0);

// Integrate the radially symmetric 2D kernel over its support: 2π∫rW(r)dr = 1.
const h = 0.004, steps = 20000, dr = h / steps;
let areaIntegral = 0;
for (let k=0;k<steps;k++) {
  const r=(k+0.5)*dr;
  areaIntegral += 2*Math.PI*r*poly6_2D(r,h)*dr;
}
assert.ok(Math.abs(areaIntegral-1)<1e-7, `2D kernel integral should be 1, got ${areaIntegral}`);

// A square lattice at the configured spacing should reconstruct approximately
// the chosen bulk density when h is twice the particle spacing.
const spacing=0.002, support=2*spacing, massPerDepth=particleMassPerDepth(rho,spacing);
let latticeDensity=0;
for(let y=-1;y<=1;y++) for(let x=-1;x<=1;x++) {
  latticeDensity+=massPerDepth*poly6_2D(Math.hypot(x*spacing,y*spacing),support);
}
assert.ok(latticeDensity>900&&latticeDensity<1100,`lattice density should approximate 1000 kg/m³, got ${latticeDensity}`);

console.log('SI mass and normalized 2D kernel tests passed.');
