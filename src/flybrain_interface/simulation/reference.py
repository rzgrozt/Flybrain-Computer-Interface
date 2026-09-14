"""Transparent Brian2 reference simulation for small validation networks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import brian2 as b2
import numpy as np
from brian2.codegen.runtime.numpy_rt import NumpyCodeObject

from flybrain_interface.connectome_data.sparse import SparseConnectivity
from flybrain_interface.contracts import NeuralReadout
from flybrain_interface.sensory.spikes import DeterministicSpikeInput
from flybrain_interface.simulation.config import ShiuLIFConfig
from flybrain_interface.simulation.trace import SimulationTrace


@dataclass(frozen=True, slots=True)
class Brian2ReferenceSimulator:
    """CPU reference backend; this is not yet a full MaleCNS simulation."""

    connectivity: SparseConnectivity
    populations: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    config: ShiuLIFConfig = field(default_factory=ShiuLIFConfig)

    def step(
        self,
        stimulus: DeterministicSpikeInput,
        *,
        duration_s: float,
    ) -> NeuralReadout:
        return self.trace(stimulus, duration_s=duration_s).readout

    def trace(
        self,
        stimulus: DeterministicSpikeInput,
        *,
        duration_s: float,
        watched_indices: tuple[int, ...] = (),
    ) -> SimulationTrace:
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        if any(
            neuron < 0 or neuron >= self.connectivity.neuron_count
            for neuron in stimulus.neuron_indices
        ):
            raise ValueError("stimulus neuron index outside network")
        if any(time >= duration_s for time in stimulus.times_s):
            raise ValueError("stimulus spike time must be inside the run duration")
        if len(set(watched_indices)) != len(watched_indices):
            raise ValueError("watched_indices must be unique")
        if any(
            neuron < 0 or neuron >= self.connectivity.neuron_count
            for neuron in watched_indices
        ):
            raise ValueError("watched neuron index outside network")

        cfg = self.config
        clock = b2.Clock(dt=cfg.dt_ms * b2.ms)
        equations = """
            dv/dt = (v_rest - v + g) / tau_membrane : volt (unless refractory)
            dg/dt = -g / tau_synapse : volt (unless refractory)
        """
        namespace = {
            "v_rest": cfg.resting_mv * b2.mV,
            "v_reset": cfg.reset_mv * b2.mV,
            "v_threshold": cfg.threshold_mv * b2.mV,
            "tau_membrane": cfg.membrane_tau_ms * b2.ms,
            "tau_synapse": cfg.synapse_tau_ms * b2.ms,
            "refractory_period": cfg.refractory_ms * b2.ms,
        }

        neurons = b2.NeuronGroup(
            self.connectivity.neuron_count,
            equations,
            threshold="v > v_threshold",
            reset="v = v_reset; g = 0*mV",
            refractory="refractory_period",
            method="linear",
            namespace=namespace,
            clock=clock,
            codeobj_class=NumpyCodeObject,
        )
        neurons.v = cfg.resting_mv * b2.mV
        neurons.g = 0 * b2.mV

        recurrent = b2.Synapses(
            neurons,
            neurons,
            model="w : volt",
            on_pre="g_post += w",
            delay=cfg.delay_ms * b2.ms,
            clock=clock,
            codeobj_class=NumpyCodeObject,
        )
        sources, targets, signed_counts = self.connectivity.edge_arrays()
        recurrent.connect(i=sources, j=targets)
        recurrent.w = signed_counts * cfg.synapse_scale_mv * b2.mV

        unsorted_indices = np.asarray(stimulus.neuron_indices, dtype=np.int64)
        unsorted_times = np.asarray(stimulus.times_s, dtype=np.float64)
        order = np.lexsort((unsorted_indices, unsorted_times))
        input_indices = unsorted_indices[order]
        input_times = unsorted_times[order] * b2.second
        spike_input = b2.SpikeGeneratorGroup(
            self.connectivity.neuron_count,
            indices=input_indices,
            times=input_times,
            sorted=True,
            clock=clock,
            codeobj_class=NumpyCodeObject,
        )
        input_synapses = b2.Synapses(
            spike_input,
            neurons,
            on_pre="v_post += input_amplitude",
            namespace={"input_amplitude": stimulus.amplitude_mv * b2.mV},
            clock=clock,
            codeobj_class=NumpyCodeObject,
        )
        input_synapses.connect(j="i")

        monitor = b2.SpikeMonitor(neurons, codeobj_class=NumpyCodeObject)
        state_monitor = b2.StateMonitor(
            neurons,
            variables=("v", "g"),
            record=np.asarray(watched_indices, dtype=np.int64),
            when="end",
            clock=clock,
            codeobj_class=NumpyCodeObject,
        )
        network = b2.Network(
            neurons,
            recurrent,
            spike_input,
            input_synapses,
            monitor,
            state_monitor,
        )
        network.run(duration_s * b2.second)

        trains = monitor.spike_trains()
        spike_times = tuple(
            tuple(float(time / b2.second) for time in trains[index])
            for index in range(self.connectivity.neuron_count)
        )
        counts = tuple(len(times) for times in spike_times)
        rates = {
            name: self._population_rate(indices, counts, duration_s)
            for name, indices in self.populations.items()
        }
        readout = NeuralReadout(
            duration_s=duration_s,
            neuron_spike_counts=counts,
            population_rates_hz=rates,
            spike_times_s=spike_times,
        )
        return SimulationTrace(
            readout=readout,
            watched_indices=watched_indices,
            sample_times_s=np.asarray(state_monitor.t / b2.second, dtype=np.float64),
            voltage_mv=np.asarray(state_monitor.v / b2.mV, dtype=np.float64).T,
            synaptic_drive_mv=np.asarray(state_monitor.g / b2.mV, dtype=np.float64).T,
        )

    def _population_rate(
        self,
        indices: tuple[int, ...],
        counts: tuple[int, ...],
        duration_s: float,
    ) -> float:
        if not indices:
            raise ValueError("population cannot be empty")
        if any(index < 0 or index >= len(counts) for index in indices):
            raise ValueError("population index outside network")
        return sum(counts[index] for index in indices) / (len(indices) * duration_s)
