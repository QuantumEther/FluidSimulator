
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

@group(0) @binding(0) var<storage, read> particles : array<Particle>;
@group(0) @binding(1) var<uniform> params : Params;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0)       uv  : vec2<f32>,
  @location(1) @interpolate(flat) color : vec3<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi : u32,
      @builtin(instance_index) ii : u32) -> VSOut {
  var o : VSOut;
  if (ii >= params.count) {
    o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    o.uv = vec2<f32>(0.0); o.color = vec3<f32>(0.0);
    return o;
  }
  var q = array<vec2<f32>, 6>(
    vec2<f32>(-1.0,-1.0), vec2<f32>(1.0,-1.0), vec2<f32>(-1.0,1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0,-1.0), vec2<f32>( 1.0,1.0)
  );
  let corner  = q[vi];
  let worldPx = particles[ii].pos + corner * params.splatRadius;
  let ndc = vec2<f32>(
    (worldPx.x / params.canvasW) * 2.0 - 1.0,
    1.0 - (worldPx.y / params.canvasH) * 2.0
  );
  o.pos   = vec4<f32>(ndc, 0.0, 1.0);
  o.uv    = corner;
  o.color = particles[ii].color.rgb;
  return o;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }
  let w = exp(-r2 * 3.6) * 0.85;
  return vec4<f32>(w, in.color.r * w, in.color.g * w, in.color.b * w);
}
