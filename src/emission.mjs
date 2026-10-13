export const WALL_MARGIN = 0.006;
export const REFERENCE_LIMIT = 1000;

export function particleCapacity(width, height, spacing, budget) {
  const columns = Math.max(0, Math.floor((width - 2 * WALL_MARGIN) / spacing + 1e-9));
  const rows = Math.max(0, Math.floor((height - 2 * WALL_MARGIN) / spacing + 1e-9));
  return Math.min(budget, columns * rows);
}

// Translate a fitting lattice as a whole. Never clamp individual particles onto a wall.
export function emissionPatch({ width, height, spacing, x, y, count }) {
  const maxColumns = Math.floor((width - 2 * WALL_MARGIN) / spacing + 1e-9);
  const maxRows = Math.floor((height - 2 * WALL_MARGIN) / spacing + 1e-9);
  count = Math.min(Math.max(0, Math.floor(count)), maxColumns * maxRows);
  if (!count) return null;
  const columns = Math.min(maxColumns, Math.ceil(Math.sqrt(count)));
  const rows = Math.ceil(count / columns);
  // Wide tanks/very large batches may need a wider patch to fit vertically.
  const fittedColumns = rows > maxRows ? Math.ceil(count / maxRows) : columns;
  const fittedRows = Math.ceil(count / fittedColumns);
  const inset = WALL_MARGIN + spacing * 0.5;
  const spanX = (fittedColumns - 1) * spacing;
  const spanY = (fittedRows - 1) * spacing;
  return { count, columns: fittedColumns,
    x: Math.max(inset, Math.min(width - inset - spanX, x - spanX / 2)),
    y: Math.max(inset, Math.min(height - inset - spanY, y - spanY / 2)) };
}
