
struct Particle {
  pos      : vec2<f32>,
  vel      : vec2<f32>,
  color    : vec4<f32>,
  density  : f32,
  pressure : f32,
  _pad     : vec2<f32>,
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
  splatRadius   : f32,
  _p0 : f32, _p1 : f32, _p2 : f32, _p3 : f32,
  _p4 : f32, _p5 : f32, _p6 : f32, _p7 : f32,
};

@group(0) @binding(0) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(1) var<uniform> params : Params;

/* ---------- 1. density / pressure / colour diffusion ---------- */
@compute @workgroup_size(64)
fn computeDensity(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  let N = params.count;
  if (i >= N) { return; }

  let pi = particles[i].pos;
  let h2 = params.h2;

  var rho : f32 = 0.0;
  var cr  : f32 = 0.0; var cg : f32 = 0.0; var cb : f32 = 0.0; var cw : f32 = 0.0;
  let mixRate = params.colorMix;

  for (var j : u32 = 0u; j < N; j = j + 1u) {
    let d  = particles[j].pos - pi;
    let r2 = dot(d, d);
    if (r2 < h2) {
      let q = 1.0 - r2 / h2;
      let w = q * q * q;
      rho = rho + w;

      let jc = particles[j].color.rgb;
      let mw = w * mixRate;
      cr = cr + jc.r * mw;
      cg = cg + jc.g * mw;
      cb = cb + jc.b * mw;
      cw = cw + mw;
    }
  }

  particles[i].density  = rho;
  particles[i].pressure = params.stiffness * (rho - params.restDensity);

  if (cw > 1e-6) {
    let avg = vec3<f32>(cr, cg, cb) / cw;
    let t   = clamp(cw * 0.04, 0.0, 0.30);
    let oldc = particles[i].color.rgb;
    particles[i].color = vec4<f32>(mix(oldc, avg, t), 1.0);
  }
}

/* ---------- 2. forces ---------- */
@compute @workgroup_size(64)
fn computeForces(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  let N = params.count;
  if (i >= N) { return; }

  let pi   = particles[i].pos;
  let vi   = particles[i].vel;
  let rhoi = max(particles[i].density, 0.05);
  let presi= particles[i].pressure;
  let h    = params.h;
  let h2   = params.h2;
  let invH = 1.0 / h;

  var ax : f32 = params.gravityX;
  var ay : f32 = params.gravityY;

  for (var j : u32 = 0u; j < N; j = j + 1u) {
    if (j == i) { continue; }
    let d  = particles[j].pos - pi;
    let r2 = dot(d, d);
    if (r2 >= h2 || r2 < 1e-5) { continue; }
    let r    = sqrt(r2);
    let rhat = d / r;
    let hmr  = h - r;

    let rhoj  = max(particles[j].density, 0.05);
    let presj = particles[j].pressure;

    // Pressure (spiky gradient)
    let gradmag = hmr * hmr;
    let pterm   = (presi / (rhoi * rhoi) + presj / (rhoj * rhoj)) * gradmag;
    ax = ax - rhat.x * pterm;
    ay = ay - rhat.y * pterm;

    // XSPH viscosity
    let vj = particles[j].vel;
    let viscW = (1.0 - r * invH) * params.viscosity;
    ax = ax + (vj.x - vi.x) * viscW;
    ay = ay + (vj.y - vi.y) * viscW;

    // Cohesion / surface tension
    let t = r * invH;
    let cohW = params.surfaceTension * t * (1.0 - t) * (1.0 - t) * 4.0;
    ax = ax + rhat.x * cohW;
    ay = ay + rhat.y * cohW;
  }

  // Acceleration clamp
  let a2 = ax * ax + ay * ay;
  let aMax = 12000.0;
  if (a2 > aMax * aMax) { let sc = aMax / sqrt(a2); ax = ax * sc; ay = ay * sc; }

  // Velocity integrate with damping
  var nvx = vi.x * params.damping + ax * params.dt;
  var nvy = vi.y * params.damping + ay * params.dt;

  // Speed clamp
  let sp2 = nvx * nvx + nvy * nvy;
  let spMax = 2500.0;
  if (sp2 > spMax * spMax) { let sc = spMax / sqrt(sp2); nvx = nvx * sc; nvy = nvy * sc; }

  // NaN guards — use (x != x) since isNan() isn't portable across Tint versions
  if (nvx != nvx) { nvx = 0.0; }
  if (nvy != nvy) { nvy = 0.0; }

  particles[i].vel = vec2<f32>(nvx, nvy);
}

/* ---------- 3. integrate + walls ---------- */
@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.count) { return; }

  var p = particles[i].pos + particles[i].vel * params.dt;
  var v = particles[i].vel;

  let PAD    = 10.0;
  let W      = params.canvasW;
  let H      = params.canvasH;
  let bounce = params.bounce;
  let fric   = params.wallFriction;

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
