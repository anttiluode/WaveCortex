"""The wave computer: the A -> B -> C body of QualiaClaude, with every part that survived, in one run().

Geometry and populations are QualiaClaude relay/'s, so earlier numbers carry over:

  the world      pulses at sites A = 0, B = 4, C = 7 of a 96x96 wave tank. It is sensed through
                 separate receptors (the passive field of the stimuli): the body cannot move its own clock
  the prediction one shared wave field (Field). Every unit writes its expectation here, positive sign
  AB units       hear A, write at B's place (sites 3-5): the held expectation of B
  AC units       hear A, write at C's place (sites 6-7): the direct path (a prior, or a shortcut)
  B units        hear B (how, depends on the B-unit kind below), write at C's place (sites 6-7)

Parts that survived the Qualia day, all on by construction:
  two fields     world and prediction are kept apart (QualeX 2.2): a medium of anti-waves carries only
                 surprise, so the expectation is held in its own field with a positive sign
  residual rule  each branch learns from what the WHOLE body still leaves unexplained at its tip
                 (prediction - world), so redundant routes are screened off (QualeX 2.3)
  latched clocks once a unit's trace starts to fall, nothing can recharge it until it fires, so the
                 predicted event cannot reset the clock that predicts it (QualeX 2.4)

The B-unit kinds (bkind):
  "content"  relay/: eyes open they hear the world's B; eyes closed, the prediction field at B's place,
             scaled by a body-measured gain G to the size of a seen B. Sends what B IS.
  "regen"    relay/ follow-up: eyes closed, the imagined B is replaced, once it crosses half a seen B's
             peak, by the rest of a canonical seen-B wave (a local regenerative event): on time, but certain.
  "residue"  innovation/ (Antti's rule): eyes open they hear B minus the body's own prediction of it; eyes
             closed they hear nothing. Sends what was NOT expected.
  "twoform"  NEW, the unit QualeX section 3 proposed ("the damper becomes a transistor"). Two
             compartments on one clock:
               basal  : the world's B. Sets the clock when B is there.
               apical : the prediction field at B's place, the body's own expectation, unscaled.
             It learns from the residue (as all kinds do) and SENDS CONTENT. What the transistor sets is
             the size of what it sends:
               evidence present (basal): size 1. The clock runs on the world's B.
               evidence absent (imagining): the clock runs on the regenerated event, so imagination is
               on time, and the ping is scaled by apical/seen: how high the unit's own fade climbed on
               its apical input, against how high a seen B drives it. So imagination keeps its
               uncertainty instead of arriving as a certain event.
             Why no formula for noisy evidence: with +-1 events, the exact E[B | A, b] is linear in A and
             b, so residual (least-squares) learning at C's tip already weights each by its reliability.
             The one place a size is needed is where nothing is sensed: imagination.
"""
from __future__ import annotations
import json, pathlib
import numpy as np
from .physics import N, PING_AMP, Field, passive
from .nodes import FadeClocks, Branches

ROOT = pathlib.Path(__file__).resolve().parent.parent
SITES = {"A": 0, "B": 4, "C": 7}
READ = {"A": 0, "B": 4}
REGION = {"AB": (3, 4, 5), "AC": (6, 7), "B": (6, 7)}
HEARS = {"AB": "A", "AC": "A", "B": "B"}
REFRACTORY = 300
RULE = {"learning_rate": 0.03, "listen_from": 3, "listen_to": 5, "self_response": 0.71, "max_weight": 2.0, "sign": 1.0}
CROSS = 0.5                                   # regenerative threshold: half a seen B's peak (relay/, fixed)


class Units:
    """The 72 units of relay/: for AB, AC, B, a + and a - half-wave bank of 12 latched fade clocks each.
    Calibration (decay, arm, release) is QualiaClaude's units_relay.json, label blind: each clock's arm
    and release are 0.8 and 0.2 of its own peak trace under one seen pulse of its own input."""
    def __init__(self, path=ROOT / "cortex" / "units.json"):
        cal = json.loads(pathlib.Path(path).read_text())
        self.pop, self.half, self.decay, self.arm, self.rel = [], [], [], [], []
        for pop in ("AB", "AC", "B"):
            for half in (1, -1):
                for d, th, _t in cal[pop]:
                    self.pop.append(pop); self.half.append(half); self.decay.append(d)
                    self.arm.append(th[0]); self.rel.append(th[1])
        self.pop = np.array(self.pop); self.half = np.array(self.half, float)
        self.decay = np.array(self.decay); self.arm = np.array(self.arm); self.rel = np.array(self.rel)
        self.n = len(self.pop)
        self.mask = np.zeros((self.n, N), bool)
        for k, p in enumerate(self.pop):
            self.mask[k, list(REGION[p])] = True
        self.isB = self.pop == "B"

    def index(self, pop, half=None):
        return [k for k in range(self.n) if self.pop[k] == pop and (half is None or self.half[k] == half)]


def events(sa, sb, sc, onset, g, g2, amp=1.0):
    ev = [(SITES["A"], onset, amp * sa)]
    if sb:
        ev.append((SITES["B"], onset + g, amp * sb))
    if sc:
        ev.append((SITES["C"], onset + g + g2, amp * sc))
    return ev


_CANON = {}
def canonical(steps):
    """A seen B of amplitude 1: its tail from the half-peak crossing on, its peak, and the crossing delay."""
    if steps not in _CANON:
        onset = 20
        w = passive([(SITES["B"], onset, 1.0)], steps + 200)[:, READ["B"]]
        pk = float(np.max(np.abs(w)))
        th = int(np.argmax(np.abs(w) >= CROSS * pk))
        _CANON[steps] = (w[th:], pk, th - onset)
    return _CANON[steps]


def run(U, W, ev, steps, learn=False, rule=None, bkind="content", eyes="open", gain=1.0, cut=(),
        b_arm=0.8, ping_scale=1.0):
    """One episode. U: Units. W: (n, 8) weights. ev: world events [(site, onset, scale)].
    gain: body-measured size gain G (content/regen eyes closed). cut: populations whose writes are removed.
    Noisy evidence lives in the world, not here: ev holds the B the body SEES (its sign and size can be
    wrong) and the C that the TRUE B caused.
    Returns prediction field (steps x 8), weights, spike ticks, the B-place residue, and B ping sizes."""
    rule = dict(RULE, **(rule or {}))
    tb = passive([e for e in ev if e[0] == SITES["B"]], steps)
    tc = passive([e for e in ev if e[0] == SITES["C"]], steps)
    target = np.zeros_like(tb); target[:, 3:6] = tb[:, 3:6]; target[:, 6:8] = tc[:, 6:8]
    sense_a = passive([e for e in ev if e[0] == SITES["A"]], steps)[:, READ["A"]]
    sense_b = tb[:, READ["B"]]

    closed_eyes = eyes == "closed"
    arm = U.arm.copy()
    if bkind == "content" and closed_eyes:
        arm[U.isB] = 0.8 * U.arm[U.isB] / 0.8                    # normalized: unchanged perception threshold
    if bkind == "residue":
        arm[U.isB] = b_arm * U.arm[U.isB] / 0.8
    amp = np.full(U.n, PING_AMP)
    if bkind == "regen":
        amp[U.isB] *= ping_scale
    clocks = FadeClocks(U.decay, arm, U.rel, amp, REFRACTORY)
    br = Branches(W, rule["learning_rate"], rule["listen_from"], rule["listen_to"], rule["self_response"],
                  rule["max_weight"], rule["sign"], learn=learn)
    field = Field(steps)
    cutmask = np.array([p in cut for p in U.pop])

    tail, pk, d0 = canonical(steps)
    regen_timing = closed_eyes and bkind in ("regen", "twoform")
    t_cross, rsign = None, 0
    cal_peak = U.arm / 0.8                                         # each clock's trace peak under a seen B
    apical = np.zeros(U.n); apical_peak = np.zeros(U.n)

    closed = np.zeros(U.n, bool); last = np.zeros(U.n)
    prev_pred = np.zeros(N); prev_pulse = np.zeros(U.n); prev_out = np.zeros((U.n, N))
    out_wave = np.zeros((steps, N)); spikes = [[] for _ in range(U.n)]
    resid = np.zeros(steps); bsize = np.full(U.n, np.nan)
    for tick in range(steps):
        prev_target = target[tick - 1] if tick > 0 else np.zeros(N)
        # --- what each unit hears this tick ---------------------------------------------------------
        src = np.zeros(U.n)
        if tick > 0:
            src[~U.isB] = sense_a[tick - 1]
            r_b = (sense_b[tick - 1] - prev_pred[READ["B"]]) if not closed_eyes else 0.0
            resid[tick] = r_b
            raw = float(prev_pred[READ["B"]])                     # the apical compartment: own expectation
            if bkind == "residue":
                src[U.isB] = r_b
            elif not closed_eyes:
                src[U.isB] = sense_b[tick - 1]
            elif bkind == "content":
                src[U.isB] = gain * raw
            else:                                                 # regen and twoform: regenerative clock
                im = gain * raw
                if t_cross is None and abs(im) >= CROSS * pk:
                    t_cross, rsign = tick, (1 if im > 0 else -1)
                if t_cross is None:
                    src[U.isB] = im
                else:
                    j = tick - t_cross
                    src[U.isB] = rsign * float(tail[j]) if j < len(tail) else 0.0
            if bkind == "twoform":
                a = np.maximum(0.0, U.half * raw)
                apical = U.decay * apical + (1 - U.decay) * a     # the unit's own fade, on its apical input
                apical_peak = np.maximum(apical_peak, apical)
        x = np.maximum(0.0, U.half * src)
        x[closed] = 0.0
        # --- the transistor: set this tick's ping size of the B units --------------------------------
        if bkind == "twoform":
            size = np.minimum(1.0, apical_peak / cal_peak) if closed_eyes else np.ones(U.n)
            clocks.amplitude = np.where(U.isB, PING_AMP * size, PING_AMP)
        # --- branches: pre = last tick's pulse; tips hear prediction - world at their own sites --------
        tip = np.where(U.mask, (prev_pred - prev_target)[None, :], 0.0)
        wr = np.where(U.mask & (prev_out != 0) & ~cutmask[:, None], prev_out, 0.0).sum(0)
        pred = field.step(wr)
        fire = clocks.step(x)
        out = br.step(prev_pulse, tip)
        for k in np.flatnonzero(fire):
            spikes[k].append(tick)
            if U.isB[k]:
                bsize[k] = clocks.pulse[k] / PING_AMP
        closed = np.where(fire, False, closed | (clocks.armed & (clocks.trace < last)))
        last = clocks.trace.copy()
        prev_pred = pred.copy(); prev_pulse = clocks.pulse.copy(); prev_out = out.copy()
        out_wave[tick] = pred
    weights = np.where(U.mask, br.W, 0.0)
    return {"prediction": out_wave, "weights": weights, "spikes": spikes, "residue": resid,
            "b_size": bsize, "regen_onset": (t_cross - 1 - d0) if t_cross is not None else None}

