const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const software = process.env.WEBGPU_SOFTWARE === '1';
(async () => {
  const browser = await chromium.launch({executablePath:process.env.CHROMIUM_BIN || '/usr/bin/chromium',
    args:['--no-sandbox',...(software ? ['--enable-unsafe-webgpu','--use-angle=swiftshader','--enable-features=Vulkan','--use-vulkan=swiftshader'] : [])]});
  const errors=[];
  try {
    const page=await browser.newPage();
    page.on('pageerror',e=>errors.push(e.message));
    page.on('console',m=>{if(m.type()==='error'&&!m.text().startsWith('Failed to load resource'))errors.push(m.text());});
    await page.goto(`${process.env.FLUID_URL || 'http://127.0.0.1:8002'}/?benchmark=1`);
    await page.waitForFunction(()=>window.fluidLab?.count===320);
    const initial=await page.evaluate(()=>({preset:document.querySelector('#presetSelect').value,
      gravity:Number(document.querySelector('#sl-gravity').value),dye:Number(document.querySelector('#sl-diffusivity').value)}));
    assert.equal(initial.preset,'smoothFlow');assert.equal(initial.gravity,9.81);assert.equal(initial.dye,1e-10);
    // Reapply after other presets/edits, including collapsed controls the screenshot does not specify.
    await page.selectOption('#presetSelect','honey');
    await page.waitForFunction(()=>!window.fluidLab.emissionPending);
    await page.evaluate(()=>{const s=window.fluidLab.settings;s.timeScale=1;s.renderScale=0.5;s.restitution=0.21;s.wallRetention=0.83;s.randomColor=false;});
    await page.selectOption('#presetSelect','smoothFlow');
    await page.waitForFunction(()=>!window.fluidLab.emissionPending);
    const selected=await page.evaluate(async()=>{
      const {PRESETS,SLIDERS}=await import('/src/config.js?v=si-smooth-flow-v6');
      const f=window.fluidLab, s=f.settings;
      return {expected:PRESETS.smoothFlow,actual:Object.fromEntries(Object.keys(PRESETS.smoothFlow).map(k=>[k,s[k]])),
        boundary:[s.restitution,s.wallRetention],randomColor:s.randomColor,
        ranges:SLIDERS.map(sl=>({key:sl.k,valid:document.querySelector('#sl-'+sl.k).validity.valid,
          displayed:document.querySelector('#val-'+sl.k).textContent,expected:sl.fmt(s[sl.k]),
          linear:sl.scale!=='log',native:Number(document.querySelector('#sl-'+sl.k).value),value:s[sl.k]}))};
    });
    assert.deepEqual(selected.actual,selected.expected);
    assert.deepEqual(selected.boundary,[0.21,0.83]);assert.equal(selected.randomColor,false);
    for(const range of selected.ranges){
      assert.ok(range.valid,range.key);assert.equal(range.displayed,range.expected,range.key);
      if(range.linear)assert.ok(Math.abs(range.native-range.value)<1e-9,`native slider rounds ${range.key}`);
    }
    await page.click('#fillBtn');
    await page.waitForFunction(()=>!window.fluidLab.emissionPending&&window.fluidLab.count===34000,null,{timeout:60000});
    const fill=await page.evaluate(async()=>{
      const f=window.fluidLab,stats=await f.step(1),state=await f.readState();
      let maxSpeed=0;for(let i=0;i<state.particles.length;i+=16)maxSpeed=Math.max(maxSpeed,Math.hypot(state.particles[i+2],state.particles[i+3]));
      return {count:f.count,finite:state.particles.every(Number.isFinite),maxSpeed,stats};
    });
    assert.equal(fill.count,34000);assert.ok(fill.finite);assert.deepEqual(errors,[]);
    console.log(JSON.stringify({status:'passed',checks:['startup preset','all visible settings and native slider roundtrip','hidden boundary/color preservation',
      '34000-particle GPU Fill and one physics tick'],fill,software},null,2));
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
