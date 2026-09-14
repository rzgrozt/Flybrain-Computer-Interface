export function parseIndices(value, {required = false, label = 'indices'} = {}) {
  if (value.trim() === '') {
    if (required) throw new Error(`${label} is required.`);
    return [];
  }
  return value.split(',').map(raw => {
    const token = raw.trim();
    if (token === '') throw new Error(`${label} contains an empty value.`);
    if (!/^\d+$/.test(token)) {
      throw new Error(`${label} must contain nonnegative integer indices.`);
    }
    const index = Number(token);
    if (!Number.isSafeInteger(index)) {
      throw new Error(`${label} contains an index outside the safe integer range.`);
    }
    return index;
  });
}
