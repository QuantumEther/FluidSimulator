
// Fluid shader set: si-smooth-flow-v7

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
  let halo = exp(-dot((uv - vec2<f32>(0.73, 0.22)) * vec2<f32>(1.3, 1.8),
                      (uv - vec2<f32>(0.73, 0.22)) * vec2<f32>(1.3, 1.8)) * 4.0);
  return base * (1.0 - d * d * 0.55) + vec3<f32>(0.012, 0.036, 0.060) * halo;
}

fn softbox(uv : vec2<f32>, center : vec2<f32>, halfSize : vec2<f32>, softness : f32) -> f32 {
  let edge = max(abs(uv.x - center.x) - halfSize.x,
                 abs(uv.y - center.y) - halfSize.y);
  return 1.0 - smoothstep(0.0, softness, edge);
}

fn studio(uv : vec2<f32>) -> vec3<f32> {
  let sky = mix(vec3<f32>(0.10, 0.18, 0.26), vec3<f32>(0.035, 0.070, 0.12), uv.y);
  let key = softbox(uv, vec2<f32>(0.36, 0.34), vec2<f32>(0.16, 0.035), 0.055);
  let fill = softbox(uv, vec2<f32>(0.73, 0.56), vec2<f32>(0.07, 0.12), 0.06);
  return sky + vec3<f32>(0.58, 0.72, 0.83) * key
             + vec3<f32>(0.16, 0.32, 0.48) * fill;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  let uv = in.uv;
  let tx = P.texel;

  let filterStep = tx * 1.2;
  let s  = textureSample(accum, samp, uv);
  let l  = textureSample(accum, samp, uv - vec2<f32>(filterStep.x, 0.0));
  let r  = textureSample(accum, samp, uv + vec2<f32>(filterStep.x, 0.0));
  let dn = textureSample(accum, samp, uv - vec2<f32>(0.0, filterStep.y));
  let u  = textureSample(accum, samp, uv + vec2<f32>(0.0, filterStep.y));
  let tl = textureSample(accum, samp, uv - filterStep);
  let tr = textureSample(accum, samp, uv + vec2<f32>( filterStep.x, -filterStep.y));
  let bl = textureSample(accum, samp, uv + vec2<f32>(-filterStep.x,  filterStep.y));
  let br = textureSample(accum, samp, uv + filterStep);
  // Bilinear taps approximate a separable 5x5 Gaussian without adding fetches.
  let field = (s * 36.0 + (l + r + dn + u) * 30.0
             + (tl + tr + bl + br) * 25.0) / 256.0;

  let den   = field.r;
  let bgc   = bg(uv);
  let alpha = smoothstep(P.threshold, P.threshold + 0.30, den);
  let diagnosticAlpha = smoothstep(0.02, 0.12, den);

  if (P.debugView > 4.5 && P.debugView < 5.5) {
    let spacing = max(P.gridSpacing, 1.0);
    let pixel = uv * P.resolution;
    let cell = fract(pixel / spacing);
    let edge = min(min(cell.x, 1.0 - cell.x), min(cell.y, 1.0 - cell.y));
    let line = 1.0 - smoothstep(0.025, 0.085, edge);
    let fluid = field.gba / max(den, 1e-5);
    let cells = mix(bgc, vec3<f32>(0.10, 0.42, 0.72), line * 0.88);
    return vec4<f32>(mix(cells, fluid, diagnosticAlpha), 1.0);
  }

  let col = field.gba / max(den, 1e-5);

  if (P.debugView > 0.5) {
    return vec4<f32>(mix(bgc, col, diagnosticAlpha), 1.0);
  }

  if (alpha < 0.003) { return vec4<f32>(bgc, 1.0); }

  // The wider centered difference suppresses grid-scale normal shimmer.
  let gxSample = ((r.r - l.r) * 4.0 + (tr.r + br.r) - (tl.r + bl.r)) / 6.0;
  let gySample = ((u.r - dn.r) * 4.0 + (bl.r + br.r) - (tl.r + tr.r)) / 6.0;
  let gx = gxSample / 2.4 * P.normalStrength / (tx.x * 720.0);
  let gy = gySample / 2.4 * P.normalStrength / (tx.y * 720.0);
  let n = normalize(vec3<f32>(-gx, gy, 1.0));

  let thickness = clamp((den - P.threshold) * 1.4 + den * 0.28, 0.0, 1.8);
  let incident = vec3<f32>(0.0, 0.0, -1.0);
  let view = vec3<f32>(0.0, 0.0, 1.0);
  let transmissionRay = refract(incident, n, 0.75);
  let refractedUv = clamp(uv + transmissionRay.xy * thickness * 0.022,
                          vec2<f32>(0.001), vec2<f32>(0.999));
  // Beer-Lambert style absorption gives longer, deeper paths a blue-green tint.
  let absorb = exp(-vec3<f32>(0.90, 0.26, 0.10) * thickness * 2.2);
  let transmitted = bg(refractedUv) * absorb + col * (1.0 - absorb) * 0.72;

  let cosView = max(dot(n, view), 0.0);
  let fresnel = (0.02 + 0.98 * pow(1.0 - cosView, 5.0)) * clamp(P.fresnel, 0.0, 1.0);
  let reflectedRay = reflect(incident, n);
  let reflectionUv = vec2<f32>(0.5 + reflectedRay.x * 0.5,
                               0.5 - reflectedRay.y * 0.5);
  let reflection = studio(reflectionUv) * fresnel * P.specular;

  let subsurface = col * vec3<f32>(1.0, 0.82, 0.68)
                 * (1.0 - clamp(thickness, 0.0, 1.0))
                 * (1.0 - fresnel) * P.subsurface * 0.12;
  let lit = transmitted * (1.0 - fresnel * 0.35) + reflection + subsurface;

  return vec4<f32>(mix(bgc, lit, alpha), 1.0);
}
