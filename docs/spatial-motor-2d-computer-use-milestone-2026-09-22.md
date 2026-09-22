# Factorized 2D MaleCNS cursor control and OS pointer bridge — 2026-09-22

## Milestone

The project now demonstrates factorized 2D virtual cursor target seeking driven by
MaleCNS descending-neuron temporal state.

Two independently validated readouts are composed:

- horizontal: temporal voltage → whitened PCA → RBF LEFT/RIGHT
- vertical: temporal voltage/drive concat → whitened PCA → RBF UP/DOWN

At each 2D control step, horizontal and vertical cursor-relative visual probes are
simulated separately through the same MaleCNS visual pathway. The two frozen motor
readouts are then combined into one MotorCommand(dx, dy).

This is a factorized 2D controller. It is not yet a unified single-frame 2D retinal
observation.

## 2D generalization result

Four diagonal start/target configurations were chosen outside the static readout grids:

- (0.52, 0.48) → (0.14, 0.16)
- (0.48, 0.52) → (0.86, 0.84)
- (0.18, 0.82) → (0.57, 0.43)
- (0.82, 0.18) → (0.43, 0.57)

Result:

- episode success: 4/4
- success fraction: 1.0
- mean Euclidean distance reduction: 0.4600
- every executed step was non-increasing in Euclidean target distance
- all configured gates passed

The readouts observe only broad descending-neuron temporal state. No screen coordinate,
target coordinate, or topographic-decoder value is supplied to the motor readout.

## OS pointer bridge

A local Linux pointer adapter was added.

Supported modes:

- dry_run: converts MotorCommand values to pixel deltas without touching the OS
- uinput: emits relative REL_X / REL_Y events through /dev/uinput

The current Linux sandbox reports:

- session type: Wayland
- /dev/uinput: writable
- Python evdev: available in the project environment after optional installation
- uinput virtual-device create/close smoke test: passed

No pointer motion was emitted during the smoke test.

The computer-use dependency remains optional:

    pip/uv extra: computer-use
    dependency: evdev>=1.7

## Safety boundary

OS pointer control is intentionally double gated.

1. The committed 2D validation config uses backend=dry_run and enabled=false.
2. Even if a config enables uinput, the CLI must also receive --allow-os-pointer.

Therefore importing modules, running unit tests, or running the normal 2D validation
cannot move the host pointer.

The web panel exposes only a read-only capability endpoint:

    GET /api/computer-use/capabilities

It reports whether uinput is available. It does not perform pointer actions.

## Current control chain

synthetic cursor-relative target position
→ measured/graded visual frontend
→ real MaleCNS connectivity
→ pruned graded relay propagation
→ 1,310 reachable descending neurons
→ temporal state
→ horizontal/vertical whitened-PCA RBF readouts
→ 2D MotorCommand
→ virtual cursor
→ optional dry-run/uinput OS pointer bridge

## Remaining gap to real computer use

The motor side is now sufficient for a first local pointer-control experiment.

The major remaining gap is sensory realism. The controller still receives a synthetic
cursor-relative target probe rather than pixels captured from the live desktop.

The next increment should therefore build a local screen-capture interface and convert
a bounded screen region into the same retinotopic visual frontend. Initial validation
should use a high-contrast target in a sandbox window before adding OCR, semantic target
selection, clicking, or Laya-based reward shaping.
