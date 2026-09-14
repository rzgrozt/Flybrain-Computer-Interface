"""Compiled numerical kernels for the sparse LIF runtime."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@njit(cache=True, fastmath=False, nogil=True)
def advance_state_numba(
    voltage_mv: FloatArray,
    synaptic_drive_mv: FloatArray,
    refractory_steps_left: IntArray,
    spike_buffer: IntArray,
    refractory_index_buffer: IntArray,
    refractory_drive_buffer: FloatArray,
    resting_mv: float,
    threshold_mv: float,
    membrane_decay: float,
    synapse_decay: float,
    drive_coupling: float,
) -> tuple[int, int]:
    """Fuse decay, threshold detection, and refractory bookkeeping."""

    spike_count = 0
    refractory_count = 0
    for neuron in range(voltage_mv.size):
        if refractory_steps_left[neuron] == 0:
            old_voltage = voltage_mv[neuron]
            old_drive = synaptic_drive_mv[neuron]
            voltage_mv[neuron] = (
                resting_mv
                + (old_voltage - resting_mv) * membrane_decay
                + old_drive * drive_coupling
            )
            synaptic_drive_mv[neuron] = old_drive * synapse_decay
            if voltage_mv[neuron] > threshold_mv:
                spike_buffer[spike_count] = neuron
                spike_count += 1
        else:
            refractory_index_buffer[refractory_count] = neuron
            refractory_drive_buffer[refractory_count] = synaptic_drive_mv[neuron]
            refractory_count += 1
            refractory_steps_left[neuron] -= 1
    return spike_count, refractory_count


@njit(cache=True, fastmath=False, nogil=True)
def advance_state_numba_zero_subnormal(
    voltage_mv: FloatArray,
    synaptic_drive_mv: FloatArray,
    refractory_steps_left: IntArray,
    spike_buffer: IntArray,
    refractory_index_buffer: IntArray,
    refractory_drive_buffer: FloatArray,
    resting_mv: float,
    threshold_mv: float,
    membrane_decay: float,
    synapse_decay: float,
    drive_coupling: float,
    minimum_preserved_drive_mv: float,
) -> tuple[int, int]:
    """Fuse state updates while preventing subnormal drive inputs/results."""

    spike_count = 0
    refractory_count = 0
    for neuron in range(voltage_mv.size):
        if refractory_steps_left[neuron] == 0:
            old_voltage = voltage_mv[neuron]
            old_drive = synaptic_drive_mv[neuron]
            if old_drive != 0.0 and abs(old_drive) < minimum_preserved_drive_mv:
                old_drive = 0.0
            voltage_mv[neuron] = (
                resting_mv
                + (old_voltage - resting_mv) * membrane_decay
                + old_drive * drive_coupling
            )
            synaptic_drive_mv[neuron] = old_drive * synapse_decay
            if voltage_mv[neuron] > threshold_mv:
                spike_buffer[spike_count] = neuron
                spike_count += 1
        else:
            refractory_index_buffer[refractory_count] = neuron
            refractory_drive_buffer[refractory_count] = synaptic_drive_mv[neuron]
            refractory_count += 1
            refractory_steps_left[neuron] -= 1
    return spike_count, refractory_count
