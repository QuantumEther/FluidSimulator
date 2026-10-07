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

@group(0) @binding(0) var<storage, read> particles : array<Particle>;
@group(0) @binding(1) var<uniform> params : Params;
@group(0) @binding(2) var<storage, read> colors : array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> previousPositions : array<vec2<f32>>;

struct RenderParams {
  resolution    : vec2<f32>,
  texel         : vec2<f32>,
  normalStrength: f32,
  threshold     : f32,
  specular      : f32,
  fresnel       : f32,
  subsurface    : f32,
  time          : f32,
  debugView     : f32,
  gridSpacing   : f32,
  interpolationAlpha: f32,
  _p1 : f32, _p2 : f32, _p3 : f32, _p4 : f32,
  _p5 : f32, _p6 : f32, _p7 : f32,
};
@group(0) @binding(4) var<uniform> renderParams : RenderParams;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
  @location(1) @interpolate(flat) color : vec3<f32>,
};

fn heat(t0 : f32) -> vec3<f32> {
  let t = clamp(t0, 0.0, 1.0);
  let a = vec3<f32>(0.08, 0.16, 0.72);
  let b = vec3<f32>(0.05, 0.86, 0.88);
  let c = vec3<f32>(1.0, 0.82, 0.18);
  let d = vec3<f32>(0.95, 0.12, 0.12);
  if (t < 0.33) { return mix(a, b, t / 0.33); }
  if (t < 0.66) { return mix(b, c, (t - 0.33) / 0.33); }
  return mix(c, d, (t - 0.66) / 0.34);
}

fn particleColor(i : u32) -> vec3<f32> {
  let p = particles[i];
  switch params.debugView {
    case 1u: { return heat(p.density / max(params.restDensity * 2.0, 0.1)); }
    case 2u: {
      let signedPressure = clamp(0.5 + p.pressure / max(params.stiffness * params.restDensity, 1.0), 0.0, 1.0);
      return heat(signedPressure);
    }
    case 3u: { return heat(length(p.vel) / 900.0); }
    case 4u: {
      let id = f32(i);
      return 0.5 + 0.5 * sin(vec3<f32>(id * 0.017, id * 0.031 + 2.1, id * 0.047 + 4.2));
    }
    case 6u: { return heat(f32(p.neighborCount) / 80.0); }
    default: { return colors[i].rgb; }
  }
}

@vertex
fn vs(@builtin(vertex_index) vi : u32, @builtin(instance_index) ii : u32) -> VSOut {
  var o : VSOut;
  if (ii >= params.count) {
    o.pos = vec4<f32>(2.0, 2.0, 2.0, 1.0);
    o.uv = vec2<f32>(0.0); o.color = vec3<f32>(0.0);
    return o;
  }
  let q = array<vec2<f32>, 6>(
    vec2<f32>(-1.0,-1.0), vec2<f32>(1.0,-1.0), vec2<f32>(-1.0,1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0,-1.0), vec2<f32>(1.0,1.0)
  );
  let corner = q[vi];
  let center = mix(previousPositions[ii], particles[ii].pos, clamp(renderParams.interpolationAlpha, 0.0, 1.0));
  let worldPx = center + corner * params.splatRadius;
  let ndc = vec2<f32>(
    worldPx.x / params.canvasW * 2.0 - 1.0,
    1.0 - worldPx.y / params.canvasH * 2.0
  );
  o.pos = vec4<f32>(ndc, 0.0, 1.0);
  o.uv = corner;
  o.color = particleColor(ii);
  return o;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }
  let w = exp(-r2 * 3.6) * 0.85;
  return vec4<f32>(w, in.color.r * w, in.color.g * w, in.color.b * w);
}
