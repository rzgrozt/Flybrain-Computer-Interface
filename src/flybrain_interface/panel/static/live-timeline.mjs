// Latest-only neural observation history: not a motor-action or guest-event log.
export function appendNeuralSample(history, payload, limit = 80) {
  if (payload?.kind !== 'telemetry' || !Number.isFinite(payload.simulated_time_s)) return history;
  if (!Number.isInteger(payload.chunks) || !payload.experiment_id) return history;
  const last = history.at(-1);
  const fresh = last && last.session_id === payload.experiment_id ? history : [];
  if (fresh.at(-1)?.step_id === payload.chunks) return fresh;
  const sample = {
    session_id: payload.experiment_id,
    step_id: payload.chunks,
    simulated_time_s: payload.simulated_time_s,
    chunk_spikes: typeof payload.chunk_spikes === 'number' ? payload.chunk_spikes : null,
    status: payload.status,
    source: 'live_simulation',
    motor: null,
    reward: null,
    timestamp_utc: null,
  };
  return [...fresh, sample].slice(-Math.min(Math.max(1, limit), 120));
}
