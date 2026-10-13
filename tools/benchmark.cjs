// Compare scheduling on identical equations/storage. This is not the pre-SI/old linked-list baseline.
const { chromium } = require('playwright');
const software = process.env.WEBGPU_SOFTWARE === '1';
const url = process.env.FLUID_URL || 'http://127.0.0.1:8002';
const counts = (process.env.FLUID_COUNTS || (software ? '1000' : '1000,8000,20000,50000')).split(',').map(Number);
const gridLayout=process.env.FLUID_GRID_LAYOUT || 'auto';
const warmup = Number(process.env.FLUID_WARMUP || 60), samples = Number(process.env.FLUID_SAMPLES || 180), repeats = Number(process.env.FLUID_REPEATS || 3);
const quantile=(values,p)=>{const sorted=[...values].sort((a,b)=>a-b);return sorted[Math.max(0,Math.ceil(sorted.length*p)-1)]??null;};
(async()=>{
  const browser=await chromium.launch({executablePath:process.env.CHROMIUM_BIN||'/usr/bin/chromium',args:['--no-sandbox',...(software?['--enable-unsafe-webgpu','--use-angle=swiftshader','--enable-features=Vulkan','--use-vulkan=swiftshader']:[])]});
  const errors=[];
  try {
    const page=await browser.newPage({viewport:{width:1280,height:900},deviceScaleFactor:1});
    page.on('pageerror',e=>errors.push(e.message));
    page.on('console',m=>{if(m.type()==='error'&&!m.text().startsWith('Failed to load resource'))errors.push(m.text());});
    await page.goto(`${url}/?benchmark=1&seed=987654321`);
    await page.waitForFunction(()=>window.fluidLab?.count===320);
    const adapter=await page.evaluate(async()=>{const a=await navigator.gpu.requestAdapter();return {vendor:a.info.vendor,architecture:a.info.architecture,device:a.info.device,description:a.info.description};});
    const rows=[];
    for(const count of counts) for(let repeat=0;repeat<repeats;repeat++) {
      // Alternate order across repetitions to reduce warm-cache/order bias.
      for(const referenceSchedule of (repeat%2?[false,true]:[true,false])) {
        const data=await page.evaluate(async({count,warmup,samples,referenceSchedule,gridLayout})=>{
          const f=window.fluidLab;
          Object.assign(f.settings,{budget:50000,spacing:0.002,restDensity:1000,gravity:0,surfaceTension:0,viscosity:0.001,
            substeps:4,solverIterations:3,diffusivity:1e-9,neighborMode:1,renderScale:1,referenceSchedule,sampleDiagnostics:false,gridLayout});
          const columns=Math.ceil(Math.sqrt(count)), rows=Math.ceil(count/columns), positions=[];
          for(let i=0;i<count;i++)positions.push([0.3+(i%columns-(columns-1)/2)*0.002,0.3+(Math.floor(i/columns)-(rows-1)/2)*0.002]);
          await f.setScene(positions);
          for(let i=0;i<warmup;i++)await f.step(1);
          const timing=[];
          for(let i=0;i<samples;i++){
            f.metrics.frames=0;f.metrics.samples.length=0;
            const start=performance.now(), stats=await f.step(1), completionMs=performance.now()-start;
            // Flush async timestamp-map callbacks without stalling the production loop.
            await new Promise(resolve=>setTimeout(resolve,0));
            timing.push({...stats,completionMs,...f.metrics.samples.at(-1)});
          }
          const state=await f.readState();let maxSpeed=0,maxCompression=0;
          for(let i=0;i<state.particles.length;i+=16){
            maxSpeed=Math.max(maxSpeed,Math.hypot(state.particles[i+2],state.particles[i+3]));
            maxCompression=Math.max(maxCompression,state.particles[i+8]/f.settings.restDensity-1);
          }
          return {timing,maxSpeed,maxCompression,finite:state.particles.every(Number.isFinite),count:f.count};
        },{count,warmup,samples,referenceSchedule,gridLayout});
        if(!data.finite||data.count!==count)throw Error('Invalid benchmark state or particle loss');
        const summary={count,repeat,mode:referenceSchedule?'unfused':'optimized',measuredTicks:samples,
          neighborSweeps:data.timing[0].neighborSweeps,gridBuilds:data.timing[0].gridBuilds,
          maxSpeed:data.maxSpeed,maxCompression:data.maxCompression};
        for(const field of ['cpuMs','compute','splat','composite','completionMs']) {
          const values=data.timing.map(x=>x[field]).filter(Number.isFinite);
          summary[field]={median:quantile(values,0.5),p95:quantile(values,0.95),samples:values.length};
        }
        rows.push(summary);
      }
    }
    if(errors.length)throw Error(errors.join('\n'));
    console.log(JSON.stringify({adapter,browser:browser.version(),software,warmup,samples,repeats,gridLayout,
      scope:'Serialized one-tick workload with rendered frames. Same corrected equations and selected grid; unfused versus fused/cached schedule. GPU timestamps exclude uploads/presentation; completion includes queue latency. No realtime FPS claim.',rows},null,2));
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
