// Fluid shader set: si-units-v1
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
  splatRadius: f32, _p0: f32, _p1: f32,
  _p2: f32, _p3: f32, _p4: f32, _p5: f32,
  _p6: f32, _p7: f32, _p8: f32, _p9: f32,
};
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> cellHeads: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> particleNext: array<u32>;
@group(0) @binding(4) var<storage, read_write> accelerations: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> sourceColors: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> targetColors: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> previousPositions: array<vec2<f32>>;
const INVALID_PARTICLE: u32 = 0xffffffffu;
const PI: f32 = 3.141592653589793;

fn kernel(r2: f32) -> f32 {
  if (r2 >= params.h2) { return 0.0; }
  let q = params.h2 - r2;
  return 4.0 / (PI * params.h2 * params.h2 * params.h2 * params.h2) * q * q * q;
}
fn kernelGrad(rij: vec2<f32>, r2: f32) -> vec2<f32> {
  if (r2 >= params.h2) { return vec2<f32>(0.0); }
  let q = params.h2 - r2;
  let scale = -24.0 / (PI * params.h2 * params.h2 * params.h2 * params.h2) * q * q;
  return scale * rij;
}
fn cellRange(i: u32) -> vec4<i32> {
  let c = vec2<i32>(floor(particles[i].pos / params.h));
  return vec4<i32>(max(c - vec2<i32>(1), vec2<i32>(0)),
    min(c + vec2<i32>(1), vec2<i32>(i32(params.gridCols)-1, i32(params.gridRows)-1)));
}
fn firstInCell(x: i32, y: i32) -> u32 {
  return atomicLoad(&cellHeads[u32(y) * params.gridCols + u32(x)]);
}

@compute @workgroup_size(64)
fn clearGrid(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x < params.gridCols * params.gridRows) { atomicStore(&cellHeads[gid.x], INVALID_PARTICLE); }
}
@compute @workgroup_size(64)
fn buildGrid(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= params.count) { return; }
  let cell = vec2<u32>(clamp(vec2<i32>(floor(particles[i].pos / params.h)), vec2<i32>(0),
    vec2<i32>(i32(params.gridCols)-1, i32(params.gridRows)-1)));
  let index = cell.y * params.gridCols + cell.x;
  particleNext[i] = atomicExchange(&cellHeads[index], i);
}

fn densityAt(i: u32) -> vec2<f32> {
  let pi = particles[i].pos;
  var rho = particles[i].massPerDepth * kernel(0.0);
  var count = 1.0;
  if (params.neighborMode == 0u) {
    for (var j=0u; j<params.count; j=j+1u) { if (j != i) {
      let r2 = dot(pi-particles[j].pos, pi-particles[j].pos);
      if (r2 < params.h2) { rho += particles[j].massPerDepth * kernel(r2); count += 1.0; }
    }}
  } else {
    let box = cellRange(i);
    for (var y=box.y; y<=box.w; y=y+1) { for (var x=box.x; x<=box.z; x=x+1) {
      var j = firstInCell(x,y); var hops=0u;
      while (j != INVALID_PARTICLE && hops < params.count) {
        if (j != i) { let d=pi-particles[j].pos; let r2=dot(d,d);
          if (r2 < params.h2) { rho += particles[j].massPerDepth*kernel(r2); count += 1.0; }
        }
        j=particleNext[j]; hops=hops+1u;
      }
    }}
  }
  return vec2<f32>(rho,count);
}
@compute @workgroup_size(64)
fn computeDensity(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i=gid.x; if(i>=params.count){return;}
  let dc=densityAt(i); particles[i].density=dc.x;
  // Pressure is a diagnostic estimate from water's bulk modulus; PBF enforces density.
  particles[i].pressure=params.bulkModulus*max(dc.x/params.restDensity-1.0,0.0);
  particles[i].neighborCount=u32(dc.y);
}

@compute @workgroup_size(64)
fn computeLambdas(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i=gid.x; if(i>=params.count){return;}
  let pi=particles[i].pos; let rho0=params.restDensity;
  let C=particles[i].density/rho0-1.0;
  var gradSum=vec2<f32>(0.0); var gradSq=0.0;
  if(params.neighborMode==0u){
    for(var j=0u;j<params.count;j=j+1u){ if(j!=i){let rij=pi-particles[j].pos;let r2=dot(rij,rij);if(r2<params.h2){
      let g=particles[j].massPerDepth/rho0*kernelGrad(rij,r2);gradSum+=g;gradSq+=dot(g,g);
    }}}
  } else {
    let box=cellRange(i);
    for(var y=box.y;y<=box.w;y=y+1){for(var x=box.x;x<=box.z;x=x+1){
      var j=firstInCell(x,y);var hops=0u;
      while(j!=INVALID_PARTICLE&&hops<params.count){if(j!=i){let rij=pi-particles[j].pos;let r2=dot(rij,rij);if(r2<params.h2){
        let g=particles[j].massPerDepth/rho0*kernelGrad(rij,r2);gradSum+=g;gradSq+=dot(g,g);
      }}j=particleNext[j];hops=hops+1u;}
    }}
  }
  let eps=1e-6/(params.h*params.h);
  particles[i].lambda=-C/(dot(gradSum,gradSum)+gradSq+eps);
}

@compute @workgroup_size(64)
fn computeCorrections(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i=gid.x;if(i>=params.count){return;}
  let pi=particles[i].pos;
  var delta=vec2<f32>(0.0);
  if(params.neighborMode==0u){
    for(var j=0u;j<params.count;j=j+1u){if(j!=i){let rij=pi-particles[j].pos;let r2=dot(rij,rij);if(r2<params.h2&&r2>1e-14){
      let scorr=-0.001*params.h2*pow(kernel(r2)/max(kernel(0.09*params.h2),1e-20),4.0);
      delta+=(particles[i].lambda+particles[j].lambda+scorr)*particles[j].massPerDepth/params.restDensity*kernelGrad(rij,r2);
    }}}
  }else{
    let box=cellRange(i);
    for(var y=box.y;y<=box.w;y=y+1){for(var x=box.x;x<=box.z;x=x+1){
      var j=firstInCell(x,y);var hops=0u;
      while(j!=INVALID_PARTICLE&&hops<params.count){if(j!=i){let rij=pi-particles[j].pos;let r2=dot(rij,rij);if(r2<params.h2&&r2>1e-14){
        let scorr=-0.001*params.h2*pow(kernel(r2)/max(kernel(0.09*params.h2),1e-20),4.0);
        delta+=(particles[i].lambda+particles[j].lambda+scorr)*particles[j].massPerDepth/params.restDensity*kernelGrad(rij,r2);
      }}j=particleNext[j];hops=hops+1u;}
    }}
  }
  let maxCorrection=0.2*params.h;let d2=dot(delta,delta);
  if(d2>maxCorrection*maxCorrection){delta*=maxCorrection/sqrt(d2);}
  accelerations[i]=delta;
}

@compute @workgroup_size(64)
fn applyCorrections(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i=gid.x;if(i>=params.count){return;}
  var p=particles[i].pos+accelerations[i];
  let dx=accelerations[i];var v=particles[i].vel+dx/params.dt;
  let pad=0.006;let W=params.canvasW;let H=params.canvasH;
  if(p.x<pad){p.x=pad;v.x=max(v.x,0.0);}if(p.x>W-pad){p.x=W-pad;v.x=min(v.x,0.0);}
  if(p.y<pad){p.y=pad;v.y=max(v.y,0.0);}if(p.y>H-pad){p.y=H-pad;v.y=min(v.y,0.0);}
  particles[i].pos=p;particles[i].vel=v;
}

@compute @workgroup_size(64)
fn diffuseColors(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i=gid.x;if(i>=params.count){return;}
  let pi=particles[i].pos;var sum=vec3<f32>(0.0);var weights=0.0;
  if(params.neighborMode==0u){for(var j=0u;j<params.count;j=j+1u){let d=pi-particles[j].pos;let r2=dot(d,d);if(r2<params.h2){let w=kernel(r2);sum+=sourceColors[j].rgb*w;weights+=w;}}}
  else{let box=cellRange(i);for(var y=box.y;y<=box.w;y=y+1){for(var x=box.x;x<=box.z;x=x+1){var j=firstInCell(x,y);var hops=0u;while(j!=INVALID_PARTICLE&&hops<params.count){let d=pi-particles[j].pos;let r2=dot(d,d);if(r2<params.h2){let w=kernel(r2);sum+=sourceColors[j].rgb*w;weights+=w;}j=particleNext[j];hops=hops+1u;}}}}
  let alpha=clamp(params.diffusivity*params.dt/(params.h2),0.0,0.25);
  let avg=select(sourceColors[i].rgb,sum/max(weights,1e-20),weights>0.0);
  targetColors[i]=vec4<f32>(mix(sourceColors[i].rgb,avg,alpha),1.0);
}

@compute @workgroup_size(64)
fn computeForces(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i=gid.x;if(i>=params.count){return;}
  let pi=particles[i].pos;let vi=particles[i].vel;let rhoi=max(particles[i].density,params.restDensity*0.1);
  var a=vec2<f32>(params.gravityX,params.gravityY);
  var colorGrad=vec2<f32>(0.0);var colorLap=0.0;var visc=vec2<f32>(0.0);
  if(params.neighborMode==0u){for(var j=0u;j<params.count;j=j+1u){if(j!=i){let rij=pi-particles[j].pos;let r2=dot(rij,rij);if(r2<params.h2&&r2>1e-14){
    let grad=kernelGrad(rij,r2);let mOverRho=particles[j].massPerDepth/max(particles[j].density,rhoi*0.1);
    colorGrad+=mOverRho*grad;let q=params.h2-r2;
    colorLap+=mOverRho*(48.0/(PI*pow(params.h,8.0)))*q*(3.0*r2-params.h2);
    let lap=40.0/(PI*pow(params.h,8.0))*q*q;
    visc+=params.viscosity/rhoi*particles[j].massPerDepth/max(particles[j].density,rhoi*0.1)*(particles[j].vel-vi)*lap;
  }}}}
  else{let box=cellRange(i);for(var y=box.y;y<=box.w;y=y+1){for(var x=box.x;x<=box.z;x=x+1){var j=firstInCell(x,y);var hops=0u;while(j!=INVALID_PARTICLE&&hops<params.count){if(j!=i){let rij=pi-particles[j].pos;let r2=dot(rij,rij);if(r2<params.h2&&r2>1e-14){
    let grad=kernelGrad(rij,r2);let mOverRho=particles[j].massPerDepth/max(particles[j].density,rhoi*0.1);
    colorGrad+=mOverRho*grad;let q=params.h2-r2;
    colorLap+=mOverRho*(48.0/(PI*pow(params.h,8.0)))*q*(3.0*r2-params.h2);
    let lap=40.0/(PI*pow(params.h,8.0))*q*q;
    visc+=params.viscosity/rhoi*particles[j].massPerDepth/max(particles[j].density,rhoi*0.1)*(particles[j].vel-vi)*lap;
  }}j=particleNext[j];hops=hops+1u;}}}}
  a+=visc;
  let gmag=length(colorGrad);
  if(gmag>1e-6){let curvature=-colorLap/gmag;a+=params.surfaceTension*curvature*colorGrad/rhoi;}
  let a2=dot(a,a);let aMax=200.0;if(a2>aMax*aMax){a*=aMax/sqrt(a2);}
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
  if(any(isNan(p))||any(isInf(p))){p=vec2<f32>(W*0.5,H*0.5);v=vec2<f32>(0.0);}
  particles[i].pos=p;particles[i].vel=v;
}
@compute @workgroup_size(64)
fn savePreviousPositions(@builtin(global_invocation_id) gid: vec3<u32>) {
  if(gid.x<params.count){previousPositions[gid.x]=particles[gid.x].pos;}
}
