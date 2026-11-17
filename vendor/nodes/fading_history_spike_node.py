"""FadingHistorySpikeNode: a minimal, explicitly state-dependent pulse generator.

A filtered absolute input `trace` rises when field activity is present.
A node is ARMED only once trace reaches `arm_threshold`. It emits a single
one-tick pulse when the trace subsequently fades to `release_threshold`.
The node cannot spontaneously start from silence. Its pulse output can be
routed back to a *separate* WaveField write port; no wave physics is modified.

This is a testable dynamical toy, NOT the GeometricNeuron V26-V29 equations,
NOT a biological action potential simulator, and NOT a learned neuron.

2026-11-15, PerceptionLabAI; MIT.
"""
import __main__
import math
from collections import deque

import cv2
import numpy as np

BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class FadingHistorySpikeNode(BaseNode):
    NODE_CATEGORY = "Cognition"
    NODE_COLOR = QtGui.QColor(83, 147, 119)

    def __init__(self, decay=0.90, arm_threshold=0.20,
                 release_threshold=0.09, pulse_amplitude=2.0,
                 refractory=3, enabled=True):
        super().__init__()
        self.node_title = "Fading History Spike"
        self.inputs = {"signal_in": "signal"}
        self.outputs = {
            "pulse": "signal",
            "trace": "signal",
            "amplitude": "signal",
            "armed": "signal",
            "fire_count": "signal",
        }
        self.decay = float(decay)
        self.arm_threshold = float(arm_threshold)
        self.release_threshold = float(release_threshold)
        self.pulse_amplitude = float(pulse_amplitude)
        self.refractory = int(refractory)
        self.enabled = bool(enabled)
        self._reset()

    def _reset(self):
        self.trace = 0.0
        self.amplitude = 0.0
        self.pulse = 0.0
        self.armed = False
        self.fire_count = 0
        self.cooldown = 0
        self.history = deque(maxlen=220)
        self.display_img = np.zeros((132, 300, 3), np.uint8)

    def step(self):
        inp = self.get_blended_input("signal_in", "sum")
        val = float(inp) if inp is not None else 0.0
        if not math.isfinite(val):
            val = 0.0
        self.amplitude = abs(val)
        decay = min(0.9999, max(0.0, self.decay))
        self.trace = decay * self.trace + (1.0 - decay) * self.amplitude
        self.pulse = 0.0

        if self.cooldown > 0:
            self.cooldown -= 1
        arm = max(1e-12, self.arm_threshold)
        release = min(max(0.0, self.release_threshold), arm * 0.9999)

        if not self.armed and self.trace >= arm:
            self.armed = True
        if self.armed and self.trace <= release and self.cooldown == 0:
            if self.enabled:
                self.pulse = self.pulse_amplitude
                self.fire_count += 1
                self.cooldown = max(0, int(self.refractory))
            self.armed = False

        self.history.append((self.trace, self.amplitude, self.pulse))
        self._render(arm, release)

    def _render(self, arm, release):
        img = np.zeros((132, 300, 3), np.uint8)
        cv2.putText(img, "FADE / ARM / FIRE", (6, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, .44, (215, 225, 225), 1)
        values = list(self.history)
        if values:
            vmax = max(arm * 1.3, max(v[0] for v in values), 1e-9)
            for threshold, color in ((arm, (50, 210, 220)),
                                     (release, (200, 130, 230))):
                y = int(112 - min(1.0, threshold / vmax) * 85)
                cv2.line(img, (5, y), (295, y), color, 1)
            last = None
            for i, (trace, amp, fired) in enumerate(values):
                x = int(5 + i * 290 / max(1, len(values) - 1))
                y = int(112 - min(1.0, trace / vmax) * 85)
                if last is not None:
                    cv2.line(img, last, (x, y), (100, 245, 120), 1)
                if abs(fired) > 1e-12:
                    cv2.line(img, (x, 111), (x, 120), (230, 230, 255), 2)
                last = (x, y)
        cv2.putText(img, "n=%d  armed=%s" % (self.fire_count, int(self.armed)),
                    (7, 130), cv2.FONT_HERSHEY_SIMPLEX,
                    .38, (220, 220, 220), 1)
        self.display_img = np.ascontiguousarray(img)

    def get_display_image(self):
        h, w = self.display_img.shape[:2]
        return QtGui.QImage(self.display_img.data, w, h, 3 * w,
                            QtGui.QImage.Format.Format_BGR888)

    def get_output(self, port_name):
        if port_name not in self.outputs:
            return None
        if port_name == "armed":
            return 1.0 if self.armed else 0.0
        return float(getattr(self, port_name))

    def get_config_options(self):
        return [
            ("Trace decay (0..1)", "decay", self.decay, "float"),
            ("Arm threshold", "arm_threshold", self.arm_threshold, "float"),
            ("Release threshold", "release_threshold", self.release_threshold, "float"),
            ("Pulse amplitude", "pulse_amplitude", self.pulse_amplitude, "float"),
            ("Min gap frames", "refractory", self.refractory, None),
            ("Enabled (1/0)", "enabled", int(self.enabled), None),
        ]

    def set_config_options(self, options):
        reset = False
        for name in ("decay", "arm_threshold", "release_threshold", "pulse_amplitude"):
            if name in options:
                new = float(options[name])
                reset |= new != getattr(self, name)
                setattr(self, name, new)
        if "refractory" in options:
            new = max(0, int(float(options["refractory"])))
            reset |= new != self.refractory
            self.refractory = new
        if "enabled" in options:
            self.enabled = bool(int(float(options["enabled"])))
        if reset:
            self._reset()

