// Small bounded browser-side history of live *simulated* per-route observations.
// Static anatomical connections never become measurements through this join.
export const MAX_PATHWAY_SAMPLES = 60;

export function appendPathwaySample(history, payload, selected, limit = MAX_PATHWAY_SAMPLES) {
  const measurement = payload?.pathway_measurement;
  if (!selected || payload?.kind !== 'telemetry' ||
      measurement?.source !== 'simulated_neural_measurement' ||
      !Number.isInteger(payload.chunks) || measurement.sampled_chunk !== payload.chunks ||
      !payload.experiment_id ||
      measurement.target_neuron_index !== selected.target_neuron_index ||
      measurement.path_index !== selected.path_index) return history;
  const ids = measurement.neuron_indices;
  const voltage = measurement.voltage_mv;
  const drive = measurement.synaptic_drive_mv;
  const spikes = measurement.spike_counts;
  if (!Array.isArray(ids) || ids.length < 3 || ids.length > 4 ||
      selected.neuron_indices?.length !== ids.length ||
      ids.some(id => !Number.isSafeInteger(id) || id < 0 || id >= 166700) ||
      ![voltage, drive, spikes].every(values => Array.isArray(values) && values.length === ids.length) ||
      !voltage.every(Number.isFinite) || !drive.every(Number.isFinite) ||
      !spikes.every(value => Number.isSafeInteger(value) && value >= 0) ||
      !Number.isFinite(measurement.simulated_time_s) ||
      !Number.isFinite(measurement.bin_duration_s) ||
      !(measurement.bin_duration_s > 0 && measurement.bin_duration_s <= .25) ||
      ids.some((id, index) => id !== selected.neuron_indices[index])) return history;
  const existing = history.at(-1)?.session_id === payload.experiment_id ? history : [];
  if (existing.at(-1)?.chunk === payload.chunks) return existing;
  return [...existing, {
    session_id: payload.experiment_id,
    chunk: payload.chunks,
    simulated_time_s: measurement.simulated_time_s,
    bin_duration_s: measurement.bin_duration_s,
    voltage_mv: [...voltage],
    drive_mv: [...drive],
    spike_counts: [...spikes],
    population_voltage_mv: voltage.reduce((a, b) => a + b, 0) / ids.length,
    population_drive_mv: drive.reduce((a, b) => a + b, 0) / ids.length,
    population_spike_rate_hz: spikes.reduce((a, b) => a + b, 0) / (ids.length * measurement.bin_duration_s),
  }].slice(-Math.min(Math.max(1, limit), MAX_PATHWAY_SAMPLES));
}
