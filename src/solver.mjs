// Device-wide dependencies remain separate ordered dispatches, even in one pass.
import { usesImplicitViscosity } from './viscosity.mjs';
export function gridLayout(settings, count) {
  return ['linked', 'contiguous'].includes(settings.gridLayout) ? settings.gridLayout : (count < 2048 ? 'linked' : 'contiguous');
}

export function encodeGrid(pass, gpu, count, cells, bindGroup, stats, layout = 'contiguous') {
  const dispatch = (pipeline, groups) => {
    pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(groups); stats.dispatches++;
  };
  if (layout === 'linked') {
    dispatch(gpu.pipeClearLinkedGrid, Math.ceil(cells / 64));
    dispatch(gpu.pipeBuildLinkedGrid, Math.ceil(count / 64));
    stats.gridBuilds++;
    return;
  }
  dispatch(gpu.pipeClearGrid, Math.ceil(cells / 64));
  dispatch(gpu.pipeCountGrid, Math.ceil(count / 64));
  dispatch(gpu.pipeScanCells, Math.ceil(cells / 256));
  dispatch(gpu.pipeScanBlocks, 1);
  dispatch(gpu.pipeScatterGrid, Math.ceil(count / 64));
  stats.gridBuilds++;
}

export function encodeSimulation({ pass, gpu, settings, count, cells, ticks, colorIndex, valid, stats }) {
  const groups = Math.ceil(count / 64);
  const dispatch = pipeline => {
    pass.setPipeline(pipeline); pass.setBindGroup(0, gpu.physBGs[colorIndex]);
    pass.dispatchWorkgroups(groups); stats.dispatches++;
  };
  const grid = () => {
    if (settings.neighborMode === 1) encodeGrid(pass, gpu, count, cells, gpu.physBGs[colorIndex], stats, gridLayout(settings, count));
  };
  if (!count) return { colorIndex, valid: false };
  for (let tick = 0; tick < ticks; tick++) {
    dispatch(gpu.pipeSavePrevious);
    if (!valid && !settings.referenceSchedule) { grid(); dispatch(gpu.pipeDensity); stats.neighborSweeps++; }
    for (let step = 0; step < Math.max(1, Math.round(settings.substeps)); step++) {
      // Opt-in benchmark control: same equations/storage, with the redundant original schedule.
      if (settings.referenceSchedule) { grid(); dispatch(gpu.pipeDensity); stats.neighborSweeps++; }
      if (usesImplicitViscosity(settings)) {
        dispatch(gpu.pipeBeginViscosity);
        for (let iteration = 0; iteration < 8; iteration++) {
          dispatch(iteration % 2 === 0 ? gpu.pipeViscosityA : gpu.pipeViscosityB);
          stats.neighborSweeps++;
        }
        dispatch(gpu.pipeApplyViscosity);
      }
      dispatch(gpu.pipeForces); stats.neighborSweeps++;
      dispatch(gpu.pipeIntegrate);
      for (let iteration = 0; iteration < settings.solverIterations; iteration++) {
        grid();
        if (settings.referenceSchedule) {
          dispatch(gpu.pipeDensity); stats.neighborSweeps++;
          dispatch(gpu.pipeLambdasReference); stats.neighborSweeps++;
        } else { dispatch(gpu.pipeDensityLambda); stats.neighborSweeps++; }
        dispatch(gpu.pipeCorrections); stats.neighborSweeps++;
        dispatch(gpu.pipeApplyCorrections);
      }
      grid();
      if (settings.referenceSchedule) {
        dispatch(gpu.pipeDensity); stats.neighborSweeps++;
        if (settings.diffusivity > 0) {
          dispatch(gpu.pipeColorsReference); stats.neighborSweeps++; colorIndex = 1 - colorIndex;
        }
      } else if (settings.diffusivity > 0) {
        dispatch(gpu.pipeDensityColors); colorIndex = 1 - colorIndex;
      } else { dispatch(gpu.pipeDensity); }
      if (!settings.referenceSchedule) stats.neighborSweeps++;
      valid = true;
    }
  }
  return { colorIndex, valid };
}
