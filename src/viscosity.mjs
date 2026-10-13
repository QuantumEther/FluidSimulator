import { FIXED_DT } from './fixed-step.mjs';

// Largest decay rate of the default periodic lattice's actual stencil.
// This policy is a reference-scale estimate, not a bound for arbitrary particle arrangements.
export function usesImplicitViscosity(settings) {
  const rate = 869.6450252151678 * settings.viscosity * (1000 / settings.restDensity) * (0.002 / settings.spacing) ** 2;
  return rate * FIXED_DT / Math.max(1, Math.round(settings.substeps)) > 1.5;
}
