// Fluid shader set: si-smooth-flow-v6
// Planar SPH kernel with per-depth particle mass (kg/m), returning volumetric density (kg/m³).

struct Particle {
  pos: vec2<f32>, vel: vec2<f32>, color: vec4<f32>,
  density: f32, pressure: f32, massPerDepth: f32, neighborCount: u32,
  lambda: f32, surfaceLaplacian: f32, _pad: vec2<f32>,
};
struct Params {
  gravityX: f32, gravityY: f32, h: f32, h2: f32,
  restDensity: f32, bulkModulus: f32, viscosity: f32, surfaceTension: f32,
  diffusivity: f32, restitution: f32, wallRetention: f32, dt: f32,
  time: f32, canvasW: f32, canvasH: f32, count: u32,
  gridCols: u32, gridRows: u32, neighborMode: u32, debugView: u32,
  splatRadius: f32, kernelCoefficient: f32, gradientCoefficient: f32,
  viscosityCoefficient: f32, laplacianCoefficient: f32, referenceKernel: f32, mass: f32,
  _p6: f32, sampleStats: f32, implicitViscosity: f32, contiguousGrid: f32, sliceDepth: f32,
};
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cellOffsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> accelerations: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> sourceColors: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> targetColors: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> previousPositions: array<vec2<f32>>;
const INVALID_PARTICLE: u32 = 0xffffffffu;
const PI: f32 = 3.141592653589793;

@group(0) @binding(8) var<storage, read_write> sortedIds: array<u32>;
struct Emission {
  origin: vec2<f32>, spacing: f32, count: u32,
  color: vec4<f32>, columns: u32, seed: u32, _pad: vec2<u32>,
};
@group(0) @binding(9) var<uniform> emission: Emission;
const MAX_CELLS: u32 = 32768u;
const EMITTED_COUNT: u32 = 65536u;
var<workgroup> scan: array<u32, 256>;

fn kernel(r2: f32) -> f32 {
  let q = max(params.h2-r2, 0.0);
  return params.kernelCoefficient*q*q*q;
}
fn kernelGrad(rij: vec2<f32>, r2: f32) -> vec2<f32> {
  let q = max(params.h2-r2,0.0);
  return params.gradientCoefficient*q*q*rij;
}
fn cellIndex(pos: vec2<f32>) -> u32 {
  let c=vec2<u32>(clamp(vec2<i32>(floor(pos/params.h)),vec2<i32>(0),
    vec2<i32>(i32(params.gridCols)-1,i32(params.gridRows)-1)));
  return c.y*params.gridCols+c.x;
}
fn cellStart(cell: u32) -> u32 {
  let cells=params.gridCols*params.gridRows;
  if(cell>=cells){return params.count;}
  let blocks=(cells+255u)/256u;
  return cellOffsets[cell]+cellOffsets[cells+blocks+cell/256u];
}
struct Cursor { box: vec4<i32>, x:i32, y:i32, index:u32, end:u32 };
fn neighborsAt(pos: vec2<f32>) -> Cursor {
  var c:Cursor;
  if(params.neighborMode==0u){c.index=0u;c.end=params.count;return c;}
  let cell=clamp(vec2<i32>(floor(pos/params.h)),vec2<i32>(0),vec2<i32>(i32(params.gridCols)-1,i32(params.gridRows)-1));
  c.box=vec4<i32>(max(cell-vec2<i32>(1),vec2<i32>(0)),min(cell+vec2<i32>(1),vec2<i32>(i32(params.gridCols)-1,i32(params.gridRows)-1)));
  c.x=c.box.x;c.y=c.box.y;
  let first=u32(c.y)*params.gridCols+u32(c.x);
  if(params.contiguousGrid<0.5){c.index=atomicLoad(&cellCounts[first]);}
  else{c.index=cellStart(first);c.end=cellStart(first+1u);}
  return c;
}
fn nextNeighbor(c:ptr<function,Cursor>) -> u32 {
  if(params.neighborMode==0u){
    if((*c).index>=(*c).end){return INVALID_PARTICLE;}
    let j=(*c).index;(*c).index++;return j;
  }
  loop {
    if(params.contiguousGrid<0.5){
      if((*c).index!=INVALID_PARTICLE){let j=(*c).index;(*c).index=sortedIds[j];return j;}
    }else{
    if((*c).index<(*c).end){let j=sortedIds[(*c).index];(*c).index++;return j;}
    }
    (*c).x++;
    if((*c).x>(*c).box.z){(*c).x=(*c).box.x;(*c).y++;}
    if((*c).y>(*c).box.w){return INVALID_PARTICLE;}
    let cell=u32((*c).y)*params.gridCols+u32((*c).x);
    if(params.contiguousGrid<0.5){(*c).index=atomicLoad(&cellCounts[cell]);}
    else{(*c).index=cellStart(cell);(*c).end=cellStart(cell+1u);}
  }
}
@compute @workgroup_size(64)
fn clearLinkedGrid(@builtin(global_invocation_id) gid:vec3<u32>){
  if(gid.x<params.gridCols*params.gridRows){atomicStore(&cellCounts[gid.x],INVALID_PARTICLE);}
}
@compute @workgroup_size(64)
fn buildLinkedGrid(@builtin(global_invocation_id) gid:vec3<u32>){
  if(gid.x<params.count){sortedIds[gid.x]=atomicExchange(&cellCounts[cellIndex(particles[gid.x].pos)],gid.x);}
}
@compute @workgroup_size(64)
fn clearGrid(@builtin(global_invocation_id) gid:vec3<u32>){
  if(gid.x<params.gridCols*params.gridRows){atomicStore(&cellCounts[gid.x],0u);atomicStore(&cellCounts[MAX_CELLS+gid.x],0u);}
}
@compute @workgroup_size(64)
fn countGrid(@builtin(global_invocation_id) gid:vec3<u32>){
  if(gid.x<params.count){atomicAdd(&cellCounts[cellIndex(particles[gid.x].pos)],1u);}
}
@compute @workgroup_size(256)
fn scanCells(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) block:vec3<u32>){
  let cells=params.gridCols*params.gridRows;
  var value=0u;if(gid.x<cells){value=atomicLoad(&cellCounts[gid.x]);}
  scan[lane]=value;workgroupBarrier();
  for(var offset=1u;offset<256u;offset*=2u){
    var add=0u;if(lane>=offset){add=scan[lane-offset];}
    workgroupBarrier();scan[lane]+=add;workgroupBarrier();
  }
  if(gid.x<cells){cellOffsets[gid.x]=scan[lane]-value;}
  if(lane==255u){cellOffsets[cells+block.x]=scan[lane];}
}
@compute @workgroup_size(256)
fn scanBlocks(@builtin(local_invocation_index) lane:u32){
  let cells=params.gridCols*params.gridRows;let blocks=(cells+255u)/256u;
  var value=0u;if(lane<blocks){value=cellOffsets[cells+lane];}
  scan[lane]=value;workgroupBarrier();
  for(var offset=1u;offset<256u;offset*=2u){
    var add=0u;if(lane>=offset){add=scan[lane-offset];}
    workgroupBarrier();scan[lane]+=add;workgroupBarrier();
  }
  if(lane<blocks){cellOffsets[cells+blocks+lane]=scan[lane]-value;}
}
@compute @workgroup_size(64)
fn scatterGrid(@builtin(global_invocation_id) gid:vec3<u32>){
  if(gid.x>=params.count){return;}
  let cell=cellIndex(particles[gid.x].pos);
  let offset=atomicAdd(&cellCounts[MAX_CELLS+cell],1u);
  sortedIds[cellStart(cell)+offset]=gid.x;
}
@compute @workgroup_size(64)
fn updateMasses(@builtin(global_invocation_id) gid:vec3<u32>){
  if(gid.x<params.count){particles[gid.x].massPerDepth=params.mass;}
}
@compute @workgroup_size(1)
fn clearEmission(){atomicStore(&cellCounts[EMITTED_COUNT],0u);}
fn hash(value:u32)->u32{var x=value;x^=x<<13u;x^=x>>17u;x^=x<<5u;return x;}
@compute @workgroup_size(64)
fn emitParticles(@builtin(global_invocation_id) gid:vec3<u32>){
  let k=gid.x;if(k>=emission.count){return;}
  let pos=emission.origin+vec2<f32>(f32(k%emission.columns),f32(k/emission.columns))*emission.spacing;
  var cursor=neighborsAt(pos);
  loop{let j=nextNeighbor(&cursor);if(j==INVALID_PARTICLE){break;}
    let d=pos-particles[j].pos;
    if(dot(d,d)<0.81*emission.spacing*emission.spacing){return;}
  }
  let i=params.count+atomicAdd(&cellCounts[EMITTED_COUNT],1u);
  var p:Particle;p.pos=pos;
  let seed=hash(emission.seed+k+1u);
  p.vel=vec2<f32>((f32(seed&65535u)/65536.0-0.5)*0.04,f32(seed>>16u)/65536.0*0.04);
  p.density=params.restDensity;p.massPerDepth=params.mass;p.neighborCount=1u;
  particles[i]=p;previousPositions[i]=pos;targetColors[i]=emission.color;
}
@compute @workgroup_size(64)
fn copyEmittedColors(@builtin(global_invocation_id) gid:vec3<u32>){
  if(gid.x<atomicLoad(&cellCounts[EMITTED_COUNT])){
    let i=params.count+gid.x;targetColors[i]=sourceColors[i];
  }
}
struct DensityResult { rho:f32, count:u32, gradient:vec2<f32>, gradientSquared:f32, dye:vec3<f32>, weights:f32 };
fn gatherDensity(i:u32,withLambda:bool,withDye:bool)->DensityResult{
  var result:DensityResult;let pos=particles[i].pos;
  var cursor=neighborsAt(pos);
  loop{let j=nextNeighbor(&cursor);if(j==INVALID_PARTICLE){break;}
    let d=pos-particles[j].pos;let r2=dot(d,d);
    if(r2<params.h2){
      let w=kernel(r2);result.rho+=particles[j].massPerDepth*w;result.count++;
      if(withLambda&&j!=i){let g=particles[j].massPerDepth/params.restDensity*kernelGrad(d,r2);result.gradient+=g;result.gradientSquared+=dot(g,g);}
      if(withDye){result.dye+=sourceColors[j].rgb*w;result.weights+=w;}
    }
  }
  return result;
}
fn writeDensity(i:u32,result:DensityResult){
  particles[i].density=result.rho;particles[i].neighborCount=result.count;
  particles[i].pressure=params.bulkModulus*max(result.rho/params.restDensity-1.0,0.0);
}
@compute @workgroup_size(64)
fn computeDensity(@builtin(global_invocation_id) gid:vec3<u32>){
  if(gid.x<params.count){writeDensity(gid.x,gatherDensity(gid.x,false,false));}
}
@compute @workgroup_size(64)
fn computeDensityLambda(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=params.count){return;}
  let result=gatherDensity(i,true,false);writeDensity(i,result);
  // Air-side missing support does not require pulling sparse particles together.
  let compression=max(result.rho/params.restDensity-1.0,0.0);
  particles[i].lambda=-compression/(dot(result.gradient,result.gradient)+result.gradientSquared+1e-4/params.h2);
}
@compute @workgroup_size(64)
fn computeDensityColors(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=params.count){return;}
  let result=gatherDensity(i,false,true);writeDensity(i,result);
  let alpha=clamp(params.diffusivity*params.dt/params.h2,0.0,0.25);
  let average=result.dye/max(result.weights,1e-20);
  targetColors[i]=vec4<f32>(mix(sourceColors[i].rgb,average,alpha),1.0);
}
// Benchmark reference kernels preserve the same formulas without fused accumulation.
@compute @workgroup_size(64)
fn lambdasReference(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=params.count){return;}
  let pos=particles[i].pos;var gradient=vec2<f32>(0.0);var squared=0.0;
  var cursor=neighborsAt(pos);
  loop{let j=nextNeighbor(&cursor);if(j==INVALID_PARTICLE){break;}if(j==i){continue;}
    let d=pos-particles[j].pos;let r2=dot(d,d);if(r2>=params.h2){continue;}
    let g=particles[j].massPerDepth/params.restDensity*kernelGrad(d,r2);gradient+=g;squared+=dot(g,g);
  }
  particles[i].lambda=-max(particles[i].density/params.restDensity-1.0,0.0)/(dot(gradient,gradient)+squared+1e-4/params.h2);
}
@compute @workgroup_size(64)
fn colorsReference(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=params.count){return;}
  let pos=particles[i].pos;var dye=vec3<f32>(0.0);var weights=0.0;var cursor=neighborsAt(pos);
  loop{let j=nextNeighbor(&cursor);if(j==INVALID_PARTICLE){break;}
    let d=pos-particles[j].pos;let r2=dot(d,d);if(r2>=params.h2){continue;}
    let w=kernel(r2);dye+=sourceColors[j].rgb*w;weights+=w;
  }
  let alpha=clamp(params.diffusivity*params.dt/params.h2,0.0,0.25);
  targetColors[i]=vec4<f32>(mix(sourceColors[i].rgb,dye/max(weights,1e-20),alpha),1.0);
}
@compute @workgroup_size(64)
fn computeCorrections(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=params.count){return;}let pos=particles[i].pos;var delta=vec2<f32>(0.0);
  var cursor=neighborsAt(pos);
  loop{let j=nextNeighbor(&cursor);if(j==INVALID_PARTICLE){break;}if(j==i){continue;}
    let d=pos-particles[j].pos;let r2=dot(d,d);
    if(r2<params.h2&&r2>1e-14){
      let ratio=kernel(r2)/params.referenceKernel;let squared=ratio*ratio;
      let scorr=-0.001*params.h2*squared*squared;
      delta+=(particles[i].lambda+particles[j].lambda+scorr)*particles[j].massPerDepth/params.restDensity*kernelGrad(d,r2);
    }
  }
  let limit=0.2*params.h;let lengthSquared=dot(delta,delta);
  if(lengthSquared>limit*limit){delta*=limit/sqrt(lengthSquared);if(params.sampleStats>0.5){atomicAdd(&cellCounts[65538u],1u);}}
  accelerations[i]=delta;
}
@compute @workgroup_size(64)
fn applyCorrections(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=params.count){return;}
  let oldPosition=particles[i].pos;
  let pos=clamp(oldPosition+accelerations[i],vec2<f32>(0.006),vec2<f32>(params.canvasW-0.006,params.canvasH-0.006));
  // Only the displacement accepted by the wall projection contributes to velocity.
  particles[i].vel+=(pos-oldPosition)/params.dt;particles[i].pos=pos;
}
@compute @workgroup_size(64)
fn beginViscosity(@builtin(global_invocation_id) gid:vec3<u32>){
  if(gid.x<params.count){particles[gid.x].color=vec4<f32>(particles[gid.x].vel,particles[gid.x].vel);}
}
fn viscosityVelocity(i:u32,fromScratch:bool)->vec2<f32>{
  if(fromScratch){return accelerations[i];}return particles[i].color.xy;
}
fn implicitViscosity(i:u32,fromScratch:bool)->vec2<f32>{
  let pos=particles[i].pos;let rhoi=max(particles[i].density,params.restDensity*0.1);
  var numerator=particles[i].color.zw;var denominator=1.0;
  var cursor=neighborsAt(pos);
  loop{let j=nextNeighbor(&cursor);if(j==INVALID_PARTICLE){break;}if(j==i){continue;}
    let d=pos-particles[j].pos;let r2=dot(d,d);if(r2>=params.h2){continue;}
    let q=params.h2-r2;
    let weight=params.dt*params.viscosity/rhoi*particles[j].massPerDepth/max(particles[j].density,rhoi*0.1)*params.viscosityCoefficient*q*q;
    numerator+=weight*viscosityVelocity(j,fromScratch);denominator+=weight;
  }
  return numerator/denominator;
}
@compute @workgroup_size(64)
fn viscosityA(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x<params.count){accelerations[gid.x]=implicitViscosity(gid.x,false);}}
@compute @workgroup_size(64)
fn viscosityB(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x<params.count){particles[gid.x].color=vec4<f32>(implicitViscosity(gid.x,true),particles[gid.x].color.zw);}}
@compute @workgroup_size(64)
fn applyViscosity(@builtin(global_invocation_id) gid:vec3<u32>){if(gid.x<params.count){particles[gid.x].vel=particles[gid.x].color.xy;}}
@compute @workgroup_size(64)
fn computeForces(@builtin(global_invocation_id) gid:vec3<u32>){
  let i=gid.x;if(i>=params.count){return;}
  let pos=particles[i].pos;let vi=particles[i].vel;let rhoi=max(particles[i].density,params.restDensity*0.1);
  var a=vec2<f32>(params.gravityX,params.gravityY);var grad=vec2<f32>(0.0);var lap=0.0;
  var cursor=neighborsAt(pos);
  loop{let j=nextNeighbor(&cursor);if(j==INVALID_PARTICLE){break;}if(j==i){continue;}
    let d=pos-particles[j].pos;let r2=dot(d,d);if(r2>=params.h2||r2<=1e-14){continue;}
    let volume=particles[j].massPerDepth/max(particles[j].density,rhoi*0.1);let q=params.h2-r2;
    grad+=volume*kernelGrad(d,r2);
    lap+=volume*params.laplacianCoefficient*q*(3.0*r2-params.h2);
    if(params.implicitViscosity<0.5){a+=params.viscosity/rhoi*volume*(particles[j].vel-vi)*params.viscosityCoefficient*q*q;}
  }
  // Classify a surface with a scale-aware gradient threshold; tiny bulk noise is not a surface.
  let gmag=length(grad);
  if(params.h*gmag>0.2){a-=params.surfaceTension*lap*grad/(rhoi*gmag);}
  let a2=dot(a,a);if(a2>40000.0){a*=200.0/sqrt(a2);if(params.sampleStats>0.5){atomicAdd(&cellCounts[65537u],1u);}}
  accelerations[i]=a;
}
@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i=gid.x;if(i>=params.count){return;}
  var v=particles[i].vel+accelerations[i]*params.dt;
  var p=particles[i].pos+v*params.dt;
  let pad=0.006;let W=params.canvasW;let H=params.canvasH;
  if(p.x<pad){p.x=pad;v.x=abs(v.x)*params.restitution;v.y*=params.wallRetention;}
  if(p.x>W-pad){p.x=W-pad;v.x=-abs(v.x)*params.restitution;v.y*=params.wallRetention;}
  if(p.y<pad){p.y=pad;v.y=abs(v.y)*params.restitution;v.x*=params.wallRetention;}
  if(p.y>H-pad){p.y=H-pad;v.y=-abs(v.y)*params.restitution;v.x*=params.wallRetention;}
  if(p.x!=p.x||p.y!=p.y||abs(p.x)>1e20||abs(p.y)>1e20){
    p=vec2<f32>(W*0.5,H*0.5);v=vec2<f32>(0.0);
  }
  particles[i].pos=p;particles[i].vel=v;
}
@compute @workgroup_size(64)
fn savePreviousPositions(@builtin(global_invocation_id) gid: vec3<u32>) {
  if(gid.x<params.count){previousPositions[gid.x]=particles[gid.x].pos;}
}

// Sampled reductions: no particle-state readback in ordinary runs.
const STATS_BASE:u32=33024u;
const STATS_FINAL:u32=39280u;
var<workgroup> statsA:array<vec4<f32>,64>;
var<workgroup> statsB:array<vec4<f32>,64>;
@compute @workgroup_size(1)
fn clearStats(){atomicStore(&cellCounts[65537u],0u);atomicStore(&cellCounts[65538u],0u);}
@compute @workgroup_size(64)
fn computeStats(@builtin(global_invocation_id) gid:vec3<u32>,@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) block:vec3<u32>){
  var a=vec4<f32>(0.0);var b=vec4<f32>(0.0);
  if(gid.x<params.count){
    let i=gid.x;let pos=particles[i].pos;let vel=particles[i].vel;
    let bits=bitcast<vec4<u32>>(vec4<f32>(pos,vel));
    let properties=bitcast<vec4<u32>>(vec4<f32>(particles[i].density,particles[i].lambda,particles[i].pressure,particles[i].massPerDepth));
    let dyeBits=bitcast<vec4<u32>>(sourceColors[i]);
    let invalid=any((bits&vec4<u32>(0x7f800000u))==vec4<u32>(0x7f800000u))||
      any((properties&vec4<u32>(0x7f800000u))==vec4<u32>(0x7f800000u))||
      any((dyeBits&vec4<u32>(0x7f800000u))==vec4<u32>(0x7f800000u));
    if(invalid){a.x=1.0;}else{
      let mass=particles[i].massPerDepth*params.sliceDepth;
      a.y=length(vel);a.z=max(particles[i].density/params.restDensity-1.0,0.0);a.w=0.5*mass*dot(vel,vel);
      b=vec4<f32>(sourceColors[i].rgb*mass,0.0);
    }
  }
  if(gid.x<params.gridCols*params.gridRows&&params.neighborMode==1u){
    if(params.contiguousGrid>0.5){b.w=f32(atomicLoad(&cellCounts[gid.x]));}
    else{var j=atomicLoad(&cellCounts[gid.x]);loop{if(j==INVALID_PARTICLE){break;}b.w+=1.0;j=sortedIds[j];}}
  }
  statsA[lane]=a;statsB[lane]=b;workgroupBarrier();
  for(var stride=32u;stride>0u;stride/=2u){
    if(lane<stride){
      let otherA=statsA[lane+stride];let otherB=statsB[lane+stride];
      statsA[lane]=vec4<f32>(statsA[lane].x+otherA.x,max(statsA[lane].y,otherA.y),max(statsA[lane].z,otherA.z),statsA[lane].w+otherA.w);
      statsB[lane]=vec4<f32>(statsB[lane].xyz+otherB.xyz,max(statsB[lane].w,otherB.w));
    }workgroupBarrier();
  }
  if(lane==0u){for(var k=0u;k<4u;k++){cellOffsets[STATS_BASE+block.x*8u+k]=bitcast<u32>(statsA[0][k]);cellOffsets[STATS_BASE+block.x*8u+4u+k]=bitcast<u32>(statsB[0][k]);}}
}
@compute @workgroup_size(1)
fn reduceStats(){
  let groups=(max(params.count,params.gridCols*params.gridRows)+63u)/64u;
  var a=vec4<f32>(0.0);var b=vec4<f32>(0.0);
  for(var group=0u;group<groups;group++){
    let offset=STATS_BASE+group*8u;
    let x=vec4<f32>(bitcast<f32>(cellOffsets[offset]),bitcast<f32>(cellOffsets[offset+1u]),bitcast<f32>(cellOffsets[offset+2u]),bitcast<f32>(cellOffsets[offset+3u]));
    let y=vec4<f32>(bitcast<f32>(cellOffsets[offset+4u]),bitcast<f32>(cellOffsets[offset+5u]),bitcast<f32>(cellOffsets[offset+6u]),bitcast<f32>(cellOffsets[offset+7u]));
    a=vec4<f32>(a.x+x.x,max(a.y,x.y),max(a.z,x.z),a.w+x.w);b=vec4<f32>(b.xyz+y.xyz,max(b.w,y.w));
  }
  for(var k=0u;k<4u;k++){cellOffsets[STATS_FINAL+k]=bitcast<u32>(a[k]);cellOffsets[STATS_FINAL+4u+k]=bitcast<u32>(b[k]);}
  cellOffsets[STATS_FINAL+8u]=bitcast<u32>(f32(atomicLoad(&cellCounts[65537u])));
  cellOffsets[STATS_FINAL+9u]=bitcast<u32>(f32(atomicLoad(&cellCounts[65538u])));
}
