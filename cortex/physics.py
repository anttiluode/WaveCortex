"""The wave field as what it is: a linear system, so one impulse response per write site.

The vendored PerceptionLab WaveFieldNode (vendor/up/nodes/wavefieldnode.py) integrates

    F_tt = c^2 lap(F) - gamma F_t + drive

on a 96x96 grid, with an absorbing edge band, and reads each aperture as the mean of a 7x7 window.
Every step of that is linear. The only nonlinearity is a clip at +-4, and no run in QualiaClaude ever
reached it (zero clipped frames in every receipt; the largest field here is about 0.1).

So the field is fully described by its impulse response: write 1.0 at site s on one tick, record the
eight readouts on every later tick. Any history of writes is then a sum of shifted, scaled copies.
This file takes that kernel FROM the vendored node itself (it is not re-derived), caches it, and
offers a Field object with the node's tick contract:

    on tick t the host sets writes w_s(t); the node steps; readouts r(t) include those writes.

The equivalence with the node, tick by tick and through whole trained bodies, is checked in
tests/test_equivalence.py.
"""
from __future__ import annotations
import pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
N = 8                                       # write/read sites on the aperture line
STIM_AMP, PING_AMP = 0.12, 0.24
WAVE = {"size": 96, "speed": 0.5, "damping": 0.006, "substeps": 4, "drive_gain": 0.6, "source_radius": 4,
        "edge_absorb": 0.06, "read_aperture": 6, "line_y": 0.5, "first_x": 0.2, "last_x": 0.92, "source_x": 0.08}
KLEN = 480                                  # ticks of impulse response kept (runs are 240; regen uses 440)
CACHE = ROOT / "cortex" / "kernel.npz"


def vendored_field():
    """A fresh vendored WaveFieldNode with the QualiaClaude parameters (the reference physics)."""
    sys.path.insert(0, str(ROOT / "vendor" / "tests"))
    from host_shim import load
    cls = load(ROOT / "vendor/up/nodes/wavefieldnode.py", "WaveFieldNode")
    f = cls()
    f.set_config_options(dict(WAVE))         # the node's own path: resizes arrays and rebuilds geometry
    for k, v in WAVE.items():                # then, as QualiaClaude's make() did, set any attribute left over
        if hasattr(f, k) and getattr(f, k) != v:
            setattr(f, k, v)
    f._render = lambda *a: None
    return f


def build_kernel(amp=1.0):
    H = np.zeros((N, KLEN, N))
    for s in range(N):
        f = vendored_field()
        for j in range(KLEN):
            f.pre_step()
            if j == 0:
                f.set_input("write_%d" % s, amp)
            f.step()
            H[s, j] = f.read_values
        H[s] /= amp
    return H


_H = None
def kernel():
    global _H
    if _H is None:
        if CACHE.exists():
            _H = np.load(CACHE)["H"]
        else:
            _H = build_kernel()
            np.savez_compressed(CACHE, H=_H)
    return _H


class Field:
    """Linear wave field with the node's tick contract. writes(w) then read() for the same tick."""
    def __init__(self, steps):
        self.H = kernel()
        self.buf = np.zeros((steps + KLEN, N))
        self.t = -1
        self.read_values = np.zeros(N)

    def step(self, w):
        self.t += 1
        t = self.t
        for s in range(N):
            if abs(w[s]) > 1e-9:                     # the node ignores |write| <= 1e-9
                self.buf[t:t + KLEN] += w[s] * self.H[s]
        self.read_values = self.buf[t]
        return self.read_values


EXACT_SENSORS = False          # tests/test_receipts.py: sensor waves bit-identical to the float32 node
_SENSOR = {}


def _node_response(site, scale, steps):
    """One stimulus on a silent vendored field, onset 0. A silent float32 field stays exactly zero until
    written, and IEEE rounding is sign-symmetric, so any onset and either sign is this array shifted or
    negated, bit for bit."""
    key = (site, abs(scale), steps)
    if key not in _SENSOR:
        f = vendored_field(); rows = []
        for tick in range(steps):
            f.pre_step()
            if 1 <= tick < 3:
                f.set_input("write_%d" % site, STIM_AMP * abs(scale))
            f.step(); rows.append(f.read_values.copy())
        _SENSOR[key] = np.asarray(rows, dtype=float)
    return _SENSOR[key]


def passive(stimuli, steps):
    """Sensor waves (steps x 8) of a field driven by the stimuli alone: [(site, onset, scale)],
    each a two-frame pulse of STIM_AMP * scale on ticks onset+1 and onset+2 (as in QualiaClaude)."""
    if EXACT_SENSORS and len(stimuli) == 1:
        site, onset, scale = stimuli[0]
        out = np.zeros((steps, N))
        if onset < steps and abs(STIM_AMP * scale) > 1e-9:
            r = _node_response(site, scale, steps)
            out[max(onset, 0):] = np.sign(scale) * r[:steps - max(onset, 0)]
        return out
    H = kernel()
    out = np.zeros((steps + KLEN, N))
    for site, onset, scale in stimuli:
        for t in (onset + 1, onset + 2):
            if 0 <= t < steps and abs(STIM_AMP * scale) > 1e-9:
                out[t:t + KLEN] += STIM_AMP * scale * H[site]
    return out[:steps]
