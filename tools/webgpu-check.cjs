// Optional real WebGPU checks. Requires Playwright and a served repository.
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const url = process.env.FLUID_URL || 'http://127.0.0.1:8002';
const software = process.env.WEBGPU_SOFTWARE === '1';
const args = ['--no-sandbox', ...(software ? ['--enable-unsafe-webgpu', '--use-angle=swiftshader', '--enable-features=Vulkan', '--use-vulkan=swiftshader'] : [])];

(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_BIN || '/usr/bin/chromium', args });
  const errors = [];
  try {
    const page = await browser.newPage();
    page.on('pageerror', e => errors.push(e.message));
    page.on('console', m => { if (m.type() === 'error' && !m.text().startsWith('Failed to load resource')) errors.push(m.text()); });
    await page.goto(`${url}/?benchmark=1&seed=987654321`);
    await page.waitForFunction(() => window.fluidLab?.count === 320);
    const results = await page.evaluate(async () => {
      const lab = window.fluidLab;
      const settings = { gravity:0, surfaceTension:0, viscosity:0.001, spacing:0.002,
        substeps:4, solverIterations:3, diffusivity:1e-7, restDensity:1000, neighborMode:1,gridLayout:'contiguous' };
      Object.assign(lab.settings, settings);
      const positions = [], colors = [], velocities = [];
      for (let y=0;y<8;y++) for(let x=0;x<8;x++) {
        positions.push([0.25+x*0.002,0.25+y*0.002]);
        colors.push([x/8,y/8,0.5]); velocities.push([(x%2 ? -1:1)*0.01,0]);
      }
      await lab.setScene(positions, velocities, colors);
      const cold = await lab.step(1), warm = await lab.step(1);
      const grid = await lab.readState();
      lab.settings.referenceSchedule=true;
      await lab.setScene(positions,velocities,colors);
      await lab.step(1); const unfusedStats=await lab.step(1);
      const unfused = await lab.readState();
      lab.settings.referenceSchedule=false;
      lab.settings.gridLayout='linked';
      await lab.setScene(positions,velocities,colors);await lab.step(1);await lab.step(1);
      const linked=await lab.readState();
      lab.settings.neighborMode=0;
      await lab.setScene(positions, velocities, colors);
      await lab.step(1); await lab.step(1);
      const reference = await lab.readState();
      const before = reference;
      lab.settings.restDensity=1200;lab.refreshParticleMasses();await lab.step(0);
      const after = await lab.readState();
      lab.settings.diffusivity=0;lab.settings.neighborMode=1;
      await lab.setScene(positions, velocities, colors);await lab.step(1);
      const noDye = await lab.readState();
      // Surface-free, projection-free viscosity isolation; alternating velocities exercise damping.
      Object.assign(lab.settings,{viscosity:10,solverIterations:0,substeps:1,diffusivity:0});
      const wave = velocities.map((_,i)=>[i%2 ? -1:1,0]);
      await lab.setScene(positions,wave,colors);await lab.step(1);
      const viscous = await lab.readState();
      lab.metrics.frames=0;await lab.step(1);
      await new Promise(resolve=>setTimeout(resolve,20));
      const diagnostics=lab.metrics.diagnostics;
      Object.assign(lab.settings, settings, {diffusivity:0});
      await lab.setScene([[0.3,0.3],[0.302,0.3]]);await lab.step(1);
      const pair = await lab.readState();
      const distance = Math.hypot(pair.particles[0]-pair.particles[16],pair.particles[1]-pair.particles[17]);
      await lab.setScene([[0.3,0.3]]);
      lab.spawn(0.3,0.3,1,[1,0,0]);
      while(lab.emissionPending) await new Promise(resolve=>setTimeout(resolve,20));
      const overlapCount=lab.count;
      await lab.setScene(Array.from({length:100},()=>[0.3,0.3]));await lab.step(1);
      const dense=await lab.readState();
      const denseCounts=Array.from(new Uint32Array(new Float32Array(dense.particles).buffer)).filter((_,i)=>i%16===11);
      lab.settings.spacing=0.006;lab.settings.budget=50000;lab.refreshParticleMasses();lab.fill();
      while(lab.emissionPending) await new Promise(resolve=>setTimeout(resolve,20));
      const fillCount=lab.count, fillState=await lab.readState();
      lab.settings.neighborMode=0;await lab.step(0);
      const guardedMode=lab.settings.neighborMode;
      lab.settings.renderScale=0.5;await lab.step(0);
      const field = {width:lab.gpu.fieldWidth,height:lab.gpu.fieldHeight,canvasWidth:document.querySelector('canvas').width};
      const adapter=await navigator.gpu.requestAdapter();
      return {adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture},cold,warm,grid,linked,reference,unfused,unfusedStats,before,after,noDye,colors,
        viscous,diagnostics,distance,overlapCount,denseCounts,fillCount,fillState,guardedMode,field};
    });
    const near = (a,b,tolerance,message) => assert.ok(Math.abs(a-b)<=tolerance, `${message}: ${a} versus ${b}`);
    for(let i=0;i<results.grid.particles.length;i++) {
      const field=i%16;
      const tolerance=field===9 ? 5000 : field===8 ? 0.02 : field<2 ? 1e-6 : field<4 ? 1e-4 : 1e-6;
      near(results.grid.particles[i],results.reference.particles[i],tolerance,`grid/reference field ${field}`);
      near(results.grid.particles[i],results.unfused.particles[i],tolerance,`fused/unfused field ${field}`);
      near(results.grid.particles[i],results.linked.particles[i],tolerance,`contiguous/linked field ${field}`);
    }
    for(let i=0;i<results.grid.colors.length;i++) near(results.grid.colors[i],results.reference.colors[i],1e-5,'dye parity');
    for(let i=0;i<results.before.particles.length;i++) {
      if(i%16!==10) assert.equal(results.before.particles[i],results.after.particles[i],'mass update preserves other fields');
    }
    assert.deepEqual(results.before.colors,results.after.colors);
    assert.deepEqual(results.before.previous,results.after.previous);
    for(let i=0;i<results.noDye.colors.length;i++) near(results.noDye.colors[i], [...results.colors[Math.floor(i/4)],1][i%4],1e-7,'zero dye');
    assert.ok(results.viscous.particles.every(Number.isFinite));
    const speeds=results.viscous.particles.filter((_,i)=>i%16===2);
    assert.ok(Math.max(...speeds.map(Math.abs))<1,'high viscosity does not amplify the alternating mode');
    assert.equal(results.diagnostics.nonfiniteCount,0);
    assert.ok(Number.isFinite(results.diagnostics.kineticEnergyJ));
    assert.ok(results.diagnostics.maxCellOccupancy>=1);
    assert.ok(results.distance>=0.001999,'sparse pair is not pulled into a clump');
    assert.equal(results.overlapCount,1);
    assert.ok(results.denseCounts.every(count=>count===100),'dense cells retain every neighbor');
    assert.equal(results.fillCount,9604);
    for(let i=0;i<results.fillState.particles.length;i+=16) {
      assert.ok(results.fillState.particles[i]>=0.006 && results.fillState.particles[i]<=0.594);
      assert.ok(results.fillState.particles[i+1]>=0.006 && results.fillState.particles[i+1]<=0.594);
    }
    assert.equal(results.guardedMode,1);
    assert.equal(results.field.width,Math.round(results.field.canvasWidth*0.5));
    assert.equal(results.warm.neighborSweeps,32);
    assert.equal(results.warm.gridBuilds,16);
    assert.equal(results.warm.computePasses,1);
    assert.equal(results.unfusedStats.neighborSweeps,52);
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({status:'passed',adapter:results.adapter,cold:results.cold,warm:results.warm,
      checks:['grid/reference and dye parity','linked/contiguous parity','fused/cached versus unfused schedule parity','GPU mass-only state preservation','zero-diffusion ping-pong','bounded high-viscosity mode',
        'GPU diagnostic reduction','sparse pair','overlap rejection','100 coincident neighbors without truncation','feasible 9604-particle fill','reference mode guard','render scale'],
      viscousMaxSpeed:Math.max(...speeds.map(Math.abs)),fillCount:results.fillCount},null,2));
  } finally { await browser.close(); }
})().catch(error => {console.error(error);process.exitCode=1;});
