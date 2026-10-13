
// Fluid shader set: si-units-v4

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
@group(0) @binding(0) var accum  : texture_2d<f32>;
@group(0) @binding(1) var samp   : sampler;
@group(0) @binding(2) var<uniform> P : RenderParams;

struct VSOut {
  @builtin(position) pos : vec4<f32>,
  @location(0)       uv  : vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi : u32) -> VSOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0,-1.0), vec2<f32>(3.0,-1.0), vec2<f32>(-1.0,3.0)
  );
  let v = p[vi];
  var o : VSOut;
  o.pos = vec4<f32>(v, 0.0, 1.0);
  o.uv  = vec2<f32>((v.x + 1.0) * 0.5, 1.0 - (v.y + 1.0) * 0.5);
  return o;
}

fn bg(uv : vec2<f32>) -> vec3<f32> {
  let base = mix(vec3<f32>(0.024,0.033,0.062), vec3<f32>(0.043,0.062,0.12), uv.y);
  let d    = length(uv - vec2<f32>(0.5)) * 1.25;
  return base * (1.0 - d * d * 0.55);
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  let uv = in.uv;
  let tx = P.texel;

  let s  = textureSample(accum, samp, uv);
  let l  = textureSample(accum, samp, uv - vec2<f32>(tx.x, 0.0)).r;
  let r  = textureSample(accum, samp, uv + vec2<f32>(tx.x, 0.0)).r;
  let dn = textureSample(accum, samp, uv - vec2<f32>(0.0, tx.y)).r;
  let u  = textureSample(accum, samp, uv + vec2<f32>(0.0, tx.y)).r;

  let den   = s.r;
  let bgc   = bg(uv);
  let alpha = smoothstep(P.threshold, P.threshold + 0.30, den);
  let diagnosticAlpha = smoothstep(0.02, 0.12, den);

  if (P.debugView > 4.5 && P.debugView < 5.5) {
    let spacing = max(P.gridSpacing, 1.0);
    let pixel = uv * P.resolution;
    let cell = fract(pixel / spacing);
    let edge = min(min(cell.x, 1.0 - cell.x), min(cell.y, 1.0 - cell.y));
    let line = 1.0 - smoothstep(0.025, 0.085, edge);
    let fluid = s.gba / max(den, 1e-5);
    let cells = mix(bgc, vec3<f32>(0.10, 0.42, 0.72), line * 0.88);
    return vec4<f32>(mix(cells, fluid, diagnosticAlpha), 1.0);
  }

  let col = s.gba / max(den, 1e-5);

  if (P.debugView > 0.5) {
    return vec4<f32>(mix(bgc, col, diagnosticAlpha), 1.0);
  }

  if (alpha < 0.003) { return vec4<f32>(bgc, 1.0); }

  let gx = (r - l) * P.normalStrength;
  let gy = (u - dn) * P.normalStrength;
  var n = normalize(vec3<f32>(-gx, gy, 1.0));

  let thick = clamp(den * 0.55, 0.0, 1.2);
  let absorb = exp(-vec3<f32>(0.15, 0.35, 0.55) * thick);
  let tinted = col * absorb;

  let L1 = normalize(vec3<f32>( 0.45, -0.75, 0.55));
  let L2 = normalize(vec3<f32>(-0.55,  0.35, 0.75));
  let V  = vec3<f32>(0.0, 0.0, 1.0);

  let ndl1 = max(dot(n, L1), 0.0);
  let ndl2 = max(dot(n, L2), 0.0);

  let spec1 = pow(max(dot(n, normalize(L1 + V)), 0.0), 96.0);
  let spec2 = pow(max(dot(n, normalize(L2 + V)), 0.0), 48.0);
  let fres  = pow(1.0 - max(dot(n, V), 0.0), 3.0);

  let sss = mix(0.55, 0.10, thick);
  let sssCol = col * vec3<f32>(1.0, 0.55, 0.35);

  var lit = tinted * (0.30 + 0.70 * ndl1 + 0.32 * ndl2);
  lit = lit + sssCol * sss * (1.0 - clamp(thick, 0.0, 1.0)) * P.subsurface;
  lit = lit + vec3<f32>(1.0, 0.96, 0.90) * spec1 * 1.1 * P.specular;
  lit = lit + vec3<f32>(0.70, 0.85, 1.00) * spec2 * 0.55 * P.specular;
  lit = lit + vec3<f32>(0.60, 0.80, 1.00) * fres * 0.5 * P.fresnel;

  return vec4<f32>(mix(bgc, lit, alpha), 1.0);
}
