/** A 2D particle is a column through the represented fluid depth. */
export function particleMassPerDepth(restDensityKgM3, spacingM) {
  return restDensityKgM3 * spacingM * spacingM;
}

export function particleMassKg(restDensityKgM3, spacingM, sliceDepthM) {
  return particleMassPerDepth(restDensityKgM3, spacingM) * sliceDepthM;
}

/** Normalized 2D poly6 kernel; its area integral is one. */
export function poly6_2D(distanceM, supportRadiusM) {
  if (distanceM < 0 || supportRadiusM <= 0 || distanceM >= supportRadiusM) return 0;
  const q = supportRadiusM ** 2 - distanceM ** 2;
  return 4 * q ** 3 / (Math.PI * supportRadiusM ** 8);
}
