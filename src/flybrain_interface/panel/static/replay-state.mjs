// Pure, bounded recorded-episode display helpers. Never infer neural clocks.
export function clampNormalized(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 0.5;
  return Math.min(1, Math.max(0, value));
}

export function selectRecordedStep(recording, episodeIndex, stepIndex) {
  const episodes = recording?.episodes;
  if (!Array.isArray(episodes) || episodes.length === 0) return null;
  const episode = episodes[Math.min(Math.max(0, episodeIndex), episodes.length - 1)];
  const steps = episode.steps;
  if (!Array.isArray(steps) || steps.length === 0) return null;
  const index = Math.min(Math.max(0, stepIndex), steps.length - 1);
  return { episode, step: steps[index], index, total: steps.length };
}

export function replayDirection(step) {
  const dx = step?.motor_command?.dx;
  const dy = step?.motor_command?.dy;
  if (typeof dx !== 'number' || typeof dy !== 'number') return null;
  if (!Number.isFinite(dx) || !Number.isFinite(dy)) return null;
  if (dx === 0 && dy === 0) return null;
  return Math.atan2(dy, dx) * 180 / Math.PI;
}

export function scoreRows(probe) {
  if (!probe || !Array.isArray(probe.scores) || !Array.isArray(probe.score_labels)) return [];
  if (probe.scores_are_probabilities !== false) return [];
  if (probe.scores.length !== probe.score_labels.length) return [];
  const maximum = Math.max(1e-12, ...probe.scores.map(value => Math.abs(value)));
  return probe.scores.map((value, index) => ({
    label: String(probe.score_labels[index]),
    value,
    // Relative magnitude only. Deliberately not a probability or confidence.
    magnitude: Number.isFinite(value) ? Math.abs(value) / maximum : 0,
  }));
}
