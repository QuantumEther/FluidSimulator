// Fluid shader set: si-units-foundation-v1

struct Particle {
  pos          : vec2<f32>,
  vel          : vec2<f32>,
  color        : vec4<f32>,
  density      : f32,
  pressure     : f32,
  neighborCount: u32,
  _pad         : u32,
};

struct Params {
  gravityX      : f32,
  gravityY      : f32,
  h             : f32,
  h2            : f32,
  restDensity   : f32,
  stiffness     : f32,
  viscosity     : f32,
  surfaceTension: f32,
  damping       : f32,
  colorMix      : f32,
  bounce        : f32,
  wallFriction  : f32,
  dt            : f32,
  time          : f32,
  canvasW       : f32,
  canvasH       : f32,
  count         : u32,
  gridCols      : u32,
  gridRows      : u32,
  neighborMode  : u32,
  debugView     : u32,
  splatRadius   : f32,
  _p0 : f32, _p1 : f32, _p2 : f32, _p3 : f32,
  _p4 : f32, _p5 : f32, _p6 : f32, _p7 : f32,
};

@group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(1) var<uniform> params : Params;
@group(0) @binding(2) var<storage, read_write> cellHeads : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> particleNext : array<u32>;
@group(0) @binding(4) var<storage, read_write> accelerations : array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> sourceColors : array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> targetColors : array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> previousPositions : array<vec2<f32>>;

const INVALID_PARTICLE : u32 = 0xffffffffu;

/* ---------- Uniform grid construction ---------- */
@compute @workgroup_size(64)
fn clearGrid(@builtin(global_invocation_id) gid : vec3<u32>) {
  let cell = gid.x;
  let cellCount = params.gridCols * params.gridRows;
  if (cell >= cellCount) { return; }
  atomicStore(&cellHeads[cell], INVALID_PARTICLE);
}

@compute @workgroup_size(64)
fn buildGrid(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }

  let cell = vec2<u32>(floor(particles[i].pos / params.h));
  let x = min(cell.x, params.gridCols - 1u);
  let y = min(cell.y, params.gridRows - 1u);
  let cellIndex = y * params.gridCols + x;
  particleNext[i] = atomicExchange(&cellHeads[cellIndex], i);
}

/* ---------- 1. density / pressure ---------- */
@compute @workgroup_size(64)
fn computeDensity(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  let N = params.count;
  if (i >= N) { return; }

  let pi = particles[i].pos;
  let h2 = params.h2;
  var rho : f32 = 0.0;
  var neighbors : u32 = 0u;

  if (params.neighborMode == 0u) {
    for (var j : u32 = 0u; j < N; j = j + 1u) {
      let d = particles[j].pos - pi;
      let r2 = dot(d, d);
      if (r2 < h2) {
        let q = 1.0 - r2 / h2;
        rho = rho + q * q * q;
        neighbors = neighbors + 1u;
      }
    }
  } else {
    let center = vec2<i32>(floor(pi / params.h));
    for (var oy : i32 = -1; oy <= 1; oy = oy + 1) {
      for (var ox : i32 = -1; ox <= 1; ox = ox + 1) {
        let cx = center.x + ox;
        let cy = center.y + oy;
        if (cx < 0 || cy < 0 || cx >= i32(params.gridCols) || cy >= i32(params.gridRows)) { continue; }
        let cellIndex = u32(cy) * params.gridCols + u32(cx);
        var j = atomicLoad(&cellHeads[cellIndex]);
        var hops = 0u;
        while (j != INVALID_PARTICLE && hops < N) {
          let d = particles[j].pos - pi;
          let r2 = dot(d, d);
          if (r2 < h2) {
            let q = 1.0 - r2 / h2;
            rho = rho + q * q * q;
            neighbors = neighbors + 1u;
          }
          j = particleNext[j];
          hops = hops + 1u;
        }
      }
    }
  }

  particles[i].density = rho;
  particles[i].pressure = params.stiffness * (rho - params.restDensity);
  particles[i].neighborCount = neighbors;
}

/* ---------- 2. race-free colour diffusion ---------- */
@compute @workgroup_size(64)
fn diffuseColors(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  let N = params.count;
  if (i >= N) { return; }
  let pi = particles[i].pos;
  let h2 = params.h2;
  var sum = vec3<f32>(0.0);
  var weightSum = 0.0;

  if (params.neighborMode == 0u) {
    for (var j : u32 = 0u; j < N; j = j + 1u) {
      let d = particles[j].pos - pi;
      let r2 = dot(d, d);
      if (r2 < h2) {
        let q = 1.0 - r2 / h2;
        let w = q * q * q * params.colorMix;
        sum = sum + sourceColors[j].rgb * w;
        weightSum = weightSum + w;
      }
    }
  } else {
    let center = vec2<i32>(floor(pi / params.h));
    for (var oy : i32 = -1; oy <= 1; oy = oy + 1) {
      for (var ox : i32 = -1; ox <= 1; ox = ox + 1) {
        let cx = center.x + ox;
        let cy = center.y + oy;
        if (cx < 0 || cy < 0 || cx >= i32(params.gridCols) || cy >= i32(params.gridRows)) { continue; }
        let cellIndex = u32(cy) * params.gridCols + u32(cx);
        var j = atomicLoad(&cellHeads[cellIndex]);
        var hops = 0u;
        while (j != INVALID_PARTICLE && hops < N) {
          let d = particles[j].pos - pi;
          let r2 = dot(d, d);
          if (r2 < h2) {
            let q = 1.0 - r2 / h2;
            let w = q * q * q * params.colorMix;
            sum = sum + sourceColors[j].rgb * w;
            weightSum = weightSum + w;
          }
          j = particleNext[j];
          hops = hops + 1u;
        }
      }
    }
  }

  var color = sourceColors[i].rgb;
  if (weightSum > 1e-6) {
    let average = sum / weightSum;
    let blend = clamp(weightSum * 0.04, 0.0, 0.30);
    color = mix(color, average, blend);
  }
  targetColors[i] = vec4<f32>(color, 1.0);
}

/* ---------- 3. force calculation; velocity writes happen later ---------- */
@compute @workgroup_size(64)
fn computeForces(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  let N = params.count;
  if (i >= N) { return; }

  let pi = particles[i].pos;
  let vi = particles[i].vel;
  let rhoi = max(particles[i].density, 0.05);
  let presi = particles[i].pressure;
  let h = params.h;
  let h2 = params.h2;
  let invH = 1.0 / h;
  var ax : f32 = params.gravityX;
  var ay : f32 = params.gravityY;

  if (params.neighborMode == 0u) {
    for (var j : u32 = 0u; j < N; j = j + 1u) {
      if (j == i) { continue; }
      let d = particles[j].pos - pi;
      let r2 = dot(d, d);
      if (r2 >= h2 || r2 < 1e-5) { continue; }
      let r = sqrt(r2);
      let rhat = d / r;
      let hmr = h - r;
      let rhoj = max(particles[j].density, 0.05);
      let presj = particles[j].pressure;
      let gradmag = hmr * hmr;
      let pterm = (presi / (rhoi * rhoi) + presj / (rhoj * rhoj)) * gradmag;
      ax = ax - rhat.x * pterm;
      ay = ay - rhat.y * pterm;
      let vj = particles[j].vel;
      let viscW = (1.0 - r * invH) * params.viscosity;
      ax = ax + (vj.x - vi.x) * viscW;
      ay = ay + (vj.y - vi.y) * viscW;
      let t = r * invH;
      let cohW = params.surfaceTension * t * (1.0 - t) * (1.0 - t) * 4.0;
      ax = ax + rhat.x * cohW;
      ay = ay + rhat.y * cohW;
    }
  } else {
    let center = vec2<i32>(floor(pi / params.h));
    for (var oy : i32 = -1; oy <= 1; oy = oy + 1) {
      for (var ox : i32 = -1; ox <= 1; ox = ox + 1) {
        let cx = center.x + ox;
        let cy = center.y + oy;
        if (cx < 0 || cy < 0 || cx >= i32(params.gridCols) || cy >= i32(params.gridRows)) { continue; }
        let cellIndex = u32(cy) * params.gridCols + u32(cx);
        var j = atomicLoad(&cellHeads[cellIndex]);
        var hops = 0u;
        while (j != INVALID_PARTICLE && hops < N) {
          if (j != i) {
            let d = particles[j].pos - pi;
            let r2 = dot(d, d);
            if (r2 < h2 && r2 >= 1e-5) {
              let r = sqrt(r2);
              let rhat = d / r;
              let hmr = h - r;
              let rhoj = max(particles[j].density, 0.05);
              let presj = particles[j].pressure;
              let gradmag = hmr * hmr;
              let pterm = (presi / (rhoi * rhoi) + presj / (rhoj * rhoj)) * gradmag;
              ax = ax - rhat.x * pterm;
              ay = ay - rhat.y * pterm;
              let vj = particles[j].vel;
              let viscW = (1.0 - r * invH) * params.viscosity;
              ax = ax + (vj.x - vi.x) * viscW;
              ay = ay + (vj.y - vi.y) * viscW;
              let t = r * invH;
              let cohW = params.surfaceTension * t * (1.0 - t) * (1.0 - t) * 4.0;
              ax = ax + rhat.x * cohW;
              ay = ay + rhat.y * cohW;
            }
          }
          j = particleNext[j];
          hops = hops + 1u;
        }
      }
    }
  }

  let a2 = ax * ax + ay * ay;
  let aMax = 12000.0;
  if (a2 > aMax * aMax) {
    let scale = aMax / sqrt(a2);
    ax = ax * scale;
    ay = ay * scale;
  }
  accelerations[i] = vec2<f32>(ax, ay);
}

/* ---------- 4. integrate and handle walls ---------- */
@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }

  let oldV = particles[i].vel;
  let accel = accelerations[i];
  var v = oldV * params.damping + accel * params.dt;
  let sp2 = dot(v, v);
  let spMax = 2500.0;
  if (sp2 > spMax * spMax) { v = v * (spMax / sqrt(sp2)); }
  if (v.x != v.x) { v.x = 0.0; }
  if (v.y != v.y) { v.y = 0.0; }

  var p = particles[i].pos + v * params.dt;
  let PAD = 10.0;
  let W = params.canvasW;
  let H = params.canvasH;
  let bounce = params.bounce;
  let fric = params.wallFriction;

  if (p.x < PAD)     { p.x = PAD;     v.x =  v.x * -bounce; v.y = v.y * fric; }
  if (p.x > W - PAD) { p.x = W - PAD; v.x = -v.x *  bounce; v.y = v.y * fric; }
  if (p.y < PAD)     { p.y = PAD;     v.y =  v.y * -bounce; v.x = v.x * fric; }
  if (p.y > H - PAD) { p.y = H - PAD; v.y = -v.y *  bounce; v.x = v.x * fric; }

  if (p.x != p.x) { p.x = W * 0.5; v.x = 0.0; }
  if (p.y != p.y) { p.y = H * 0.5; v.y = 0.0; }
  if (v.x != v.x) { v.x = 0.0; }
  if (v.y != v.y) { v.y = 0.0; }

  particles[i].pos = p;
  particles[i].vel = v;
}

@compute @workgroup_size(64)
fn savePreviousPositions(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }
  previousPositions[i] = particles[i].pos;
}
