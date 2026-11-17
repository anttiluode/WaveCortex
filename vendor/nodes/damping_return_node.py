"""DampingReturnNode — signed return branches that learn to flatten the wave at their own tip.

Gemini's proposal ("local predictive damping"), made concrete. One unit's pulse can be written back
into the WaveField at up to eight sites, each through a branch with a SIGNED weight starting at zero.

When the unit fires, each branch injects  pulse * w_s  at its site. The branch then listens to the
field at its own tip for a short window just after the injection and moves its weight against what
it hears:

    m_s  = mean of site_s over the listening window
    w_s <- clip( w_s - learning_rate * m_s / self_response )

`self_response` is the mean field one unit of weight would itself produce at the tip in that window
(a calibrated constant). So the update is a one-tap LMS step whose fixed point is the weight that
makes the tip quiet on average right after this unit fires:

    w_s* = - E[ what else is at the tip then ] / self_response

A branch whose tip hears nothing consistent after its unit fires has fixed point zero: it removes
itself without any pruning rule or budget. A branch whose unit reliably fires just as something
arrives at its tip grows negative and cancels that arrival.

Local information only: the unit's own pulse and the field at each branch tip. No label, reader,
reward, target waveform or schedule.

This is a least-mean-squares adaptive canceller (Widrow and Hoff; adaptive noise cancelling,
Widrow et al. 1975) and an anti-Hebbian rule. It is a test component, NOT a biological model.

15 November 2026, Claude Fable 5.2, for anttiluode/StateShapedPingRecall. MIT.
"""
import math
import __main__
import numpy as np
import cv2

BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

N_SITES = 8


class DampingReturnNode(BaseNode):
    NODE_CATEGORY = "Cognition"
    NODE_COLOR = QtGui.QColor(70, 110, 140)

    def __init__(self):
        super().__init__()
        self.node_title = "Damping Return"
        self.inputs = {"pre_spike": "signal"}
        for s in range(N_SITES):
            self.inputs["site_%d" % s] = "signal"
        self.outputs = {"largest_abs_weight": "signal"}
        for s in range(N_SITES):
            self.outputs["out_%d" % s] = "signal"
            self.outputs["w_%d" % s] = "signal"
        self.learning_rate = 0.3
        self.listen_from = 3          # node steps after the pulse arrives here
        self.listen_to = 5            # inclusive
        self.self_response = 0.05     # calibrated: mean tip field per unit weight in the listening window
        self.max_weight = 2.0
        self.sign = 1.0               # +1 damping (the rule); -1 amplifying control
        self.learn_enabled = 1
        self.weights = [0.0] * N_SITES
        self.out = [0.0] * N_SITES
        self.age = -1                 # node steps since the last pulse arrived; -1 = not listening
        self.acc = [0.0] * N_SITES
        self.update_count = 0
        self.display_img = np.zeros((120, 310, 3), np.uint8)

    def step(self):
        inp = self.get_blended_input("pre_spike", "sum")
        pre = float(inp) if inp is not None else 0.0
        if not math.isfinite(pre):
            pre = 0.0
        site = []
        for s in range(N_SITES):
            v = self.get_blended_input("site_%d" % s, "sum")
            site.append(float(v) if v is not None and math.isfinite(float(v)) else 0.0)

        if self.age >= 0:
            self.age += 1
            if int(self.listen_from) <= self.age <= int(self.listen_to):
                self.acc = [a + x for a, x in zip(self.acc, site)]
            if self.age == int(self.listen_to):
                if int(self.learn_enabled):
                    n = int(self.listen_to) - int(self.listen_from) + 1
                    k = float(self.sign) * float(self.learning_rate) / (n * max(1e-9, float(self.self_response)))
                    lim = float(self.max_weight)
                    self.weights = [min(lim, max(-lim, w - k * a)) for w, a in zip(self.weights, self.acc)]
                    self.update_count += 1
                self.age = -1

        if abs(pre) > 1e-12:
            self.age = 0                      # (re)start listening; an unfinished window is dropped
            self.acc = [0.0] * N_SITES
        self.out = [pre * w for w in self.weights]
        self._render()

    def _render(self):
        im = np.zeros((120, 310, 3), np.uint8)
        cv2.putText(im, "DAMPING RETURN  updates=%d" % self.update_count, (5, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, (225, 230, 235), 1)
        top = max(0.05, max(abs(w) for w in self.weights))
        for s in range(N_SITES):
            x0 = 10 + s * 37
            h = int(40 * self.weights[s] / top)
            col = (200, 120, 90) if self.weights[s] < 0 else (90, 200, 120)
            cv2.rectangle(im, (x0, 66 - max(h, 0)), (x0 + 28, 66 - min(h, 0)), col, -1)
            cv2.putText(im, str(s), (x0 + 9, 119), cv2.FONT_HERSHEY_SIMPLEX, .32, (200, 200, 200), 1)
        cv2.line(im, (5, 66), (305, 66), (120, 120, 120), 1)
        self.display_img = np.ascontiguousarray(im)

    def get_output(self, name):
        if name == "largest_abs_weight":
            return float(max(abs(w) for w in self.weights))
        if name.startswith("out_"):
            return float(self.out[int(name[4:])])
        if name.startswith("w_"):
            return float(self.weights[int(name[2:])])
        return None

    def get_display_image(self):
        h, w = self.display_img.shape[:2]
        return QtGui.QImage(self.display_img.data, w, h, w * 3, QtGui.QImage.Format.Format_BGR888)

    def get_config_options(self):
        opts = [("Learning rate", "learning_rate", self.learning_rate, "float"),
                ("Listen from (steps)", "listen_from", self.listen_from, None),
                ("Listen to (steps)", "listen_to", self.listen_to, None),
                ("Self response", "self_response", self.self_response, "float"),
                ("Max |weight|", "max_weight", self.max_weight, "float"),
                ("Sign (+1 damp, -1 control)", "sign", self.sign, "float"),
                ("Enable learning (1/0)", "learn_enabled", self.learn_enabled, None)]
        for s in range(N_SITES):
            opts.append(("Weight to site %d" % s, "w%d" % s, self.weights[s], "float"))
        return opts

    def set_config_options(self, options):
        for f in ("learning_rate", "self_response", "max_weight", "sign"):
            if f in options:
                setattr(self, f, float(options[f]))
        for f in ("listen_from", "listen_to", "learn_enabled"):
            if f in options:
                setattr(self, f, int(float(options[f])))
        if "weights" in options:
            self.weights = [float(x) for x in options["weights"]][:N_SITES]
        for s in range(N_SITES):
            if "w%d" % s in options:
                self.weights[s] = float(options["w%d" % s])
