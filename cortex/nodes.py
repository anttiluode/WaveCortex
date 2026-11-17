"""Banks of the two vendored node types, vectorized, with the same arithmetic as the node code.

FadeClocks  = many FadingHistorySpikeNodes  (vendor/nodes/fading_history_spike_node.py)
Branches    = many DampingReturnNodes       (vendor/nodes/damping_return_node.py)

Each step() below is the node's step(), line for line, applied to an array of units. Nothing about the
behaviour is changed; tests/test_equivalence.py runs whole trained bodies through both versions.
"""
from __future__ import annotations
import numpy as np


class FadeClocks:
    """A unit's trace is a leaky average of |input|. It ARMS when the trace reaches arm, and fires one
    pulse when the trace has then faded to release. It cannot start from silence."""
    def __init__(self, decay, arm, release, amplitude, refractory):
        self.decay = np.clip(np.asarray(decay, float), 0.0, 0.9999)
        self.arm = np.maximum(1e-12, np.asarray(arm, float))
        self.release = np.minimum(np.maximum(0.0, np.asarray(release, float)), self.arm * 0.9999)
        self.amplitude = np.asarray(amplitude, float) * np.ones_like(self.decay)
        self.refractory = int(refractory)
        n = len(self.decay)
        self.trace = np.zeros(n); self.armed = np.zeros(n, bool)
        self.cooldown = np.zeros(n, int); self.pulse = np.zeros(n)

    def step(self, x):
        amp = np.abs(np.asarray(x, float))
        self.trace = self.decay * self.trace + (1.0 - self.decay) * amp
        self.pulse = np.zeros_like(self.trace)
        self.cooldown = np.where(self.cooldown > 0, self.cooldown - 1, self.cooldown)
        self.armed |= (~self.armed) & (self.trace >= self.arm)
        fire = self.armed & (self.trace <= self.release) & (self.cooldown == 0)
        self.pulse[fire] = self.amplitude[fire]
        self.cooldown[fire] = max(0, self.refractory)
        self.armed &= ~fire
        return fire


class Branches:
    """Eight signed return branches per unit. When the unit's pulse arrives, branch s writes pulse*w_s at
    site s; it then listens to its own tip for listen_from..listen_to steps and moves w_s against the mean
    it heard (a one-tap LMS step; fixed point = the tip is quiet just after this unit fires)."""
    def __init__(self, W, learning_rate, listen_from, listen_to, self_response, max_weight=2.0, sign=1.0,
                 learn=False):
        self.W = np.array(W, float)
        n = self.W.shape[0]
        self.lf, self.lt = int(listen_from), int(listen_to)
        self.k = float(sign) * float(learning_rate) / ((self.lt - self.lf + 1) * max(1e-9, float(self_response)))
        self.lim = float(max_weight)
        self.learn = bool(learn)
        self.age = -np.ones(n, int); self.acc = np.zeros((n, self.W.shape[1])); self.out = np.zeros_like(self.W)

    def step(self, pre, site):
        pre = np.asarray(pre, float)
        listening = self.age >= 0
        self.age = np.where(listening, self.age + 1, self.age)
        win = listening & (self.age >= self.lf) & (self.age <= self.lt)
        self.acc[win] += site[win]
        done = listening & (self.age == self.lt)
        if self.learn and done.any():
            self.W[done] = np.clip(self.W[done] - self.k * self.acc[done], -self.lim, self.lim)
        self.age[done] = -1
        new = np.abs(pre) > 1e-12
        self.age[new] = 0
        self.acc[new] = 0.0
        self.out = pre[:, None] * self.W
        return self.out
