"""WaveCortex: the A -> B -> C wave body of WaveComputer with the apical gate of ApicalRuleReversalMachine.

From WaveComputer (machine.py), unchanged:
  the geometry (A = site 0, B = site 4, C = site 7), the 72 calibrated latched fade clocks (AB, AC, B; a +
  and a - half-wave bank each), two fields (world sensed through receptors; one prediction field), residual
  learning at every branch tip, and the B-unit kinds for imagination:
     content  imagined B scaled by a body gain G (relay/)
     regen    imagined B regenerated into a seen B's shape once it crosses half a seen B's peak: on time, certain
     twoform  the regenerated clock, but the ping is sized by apical/seen: how high the unit's own fade climbed
              on the prediction at B's place. On time AND as unsure as the body is.
From ApicalRuleReversalMachine (rulemachine.py), unchanged:
  every branch has a basal weight W and an apical-gated weight V; a unit writes pulse * (W + g V), with one
  apical state g in [-1, 1] shared by every tuft. Three-factor learning:
     W <- W - k acc,   V <- V - k g_fire acc,
     g <- g - eta * sum(acc V) / (s (sum V^2 + kappa))   once per trial, pooled over the tufts that fired.
  Slow weights (rate 0.006), fast state.

New is only that the two meet: g gates the A -> B branches, so the imagined B, and through it C, carries
the body's belief about a hidden rule. The twoform unit's size reads that belief off the apical fade.
"""
from __future__ import annotations
import json, pathlib
import numpy as np
from .physics import N, PING_AMP, Field, passive
from .nodes import FadeClocks

ROOT = pathlib.Path(__file__).resolve().parent.parent
SITES = {"A": 0, "B": 4, "C": 7}
READ = {"A": 0, "B": 4}
REGION = {"AB": (3, 4, 5), "AC": (6, 7), "B": (6, 7)}
REFRACTORY = 300
RULE = {"listen_from": 3, "listen_to": 5, "self_response": 0.71, "max_weight": 2.0}
CROSS = 0.5


class Units:
    def __init__(self, path=ROOT / "cortex" / "units.json"):
        cal = json.loads(pathlib.Path(path).read_text())
        rows = [(pop, half, d, th) for pop in ("AB", "AC", "B") for half in (1, -1) for d, th, _ in cal[pop]]
        self.pop = np.array([r[0] for r in rows]); self.half = np.array([r[1] for r in rows], float)
        self.decay = np.array([r[2] for r in rows]); self.arm = np.array([r[3][0] for r in rows])
        self.rel = np.array([r[3][1] for r in rows]); self.n = len(rows)
        self.mask = np.zeros((self.n, N), bool)
        for k, p in enumerate(self.pop):
            self.mask[k, list(REGION[p])] = True
        self.isB = self.pop == "B"

    def index(self, pop, half=None):
        return [k for k in range(self.n) if self.pop[k] == pop and (half is None or self.half[k] == half)]


def events(sa, sb, sc, onset, g1, g2, amp=1.0):
    ev = [(SITES["A"], onset, amp * sa)]
    if sb:
        ev.append((SITES["B"], onset + g1, amp * sb))
    if sc:
        ev.append((SITES["C"], onset + g1 + g2, amp * sc))
    return ev


_CANON = {}
def canonical(steps):
    if steps not in _CANON:
        w = passive([(SITES["B"], 20, 1.0)], steps + 200)[:, READ["B"]]
        pk = float(np.max(np.abs(w))); th = int(np.argmax(np.abs(w) >= CROSS * pk))
        _CANON[steps] = (w[th:], pk)
    return _CANON[steps]


def run(U, W, V, g, ev, steps, learn=False, lr=0.006, bkind="twoform", eyes="open", gain=1.0, cut=(),
        update_g=False, eta_g=0.5, kappa=0.05, apical=True):
    W = np.array(W, float); V = np.array(V, float)
    tb = passive([e for e in ev if e[0] == SITES["B"]], steps)
    tc = passive([e for e in ev if e[0] == SITES["C"]], steps)
    target = np.zeros_like(tb); target[:, 3:6] = tb[:, 3:6]; target[:, 6:8] = tc[:, 6:8]
    sense_a = passive([e for e in ev if e[0] == SITES["A"]], steps)[:, READ["A"]]
    sense_b = tb[:, READ["B"]]
    closed_eyes = eyes == "closed"
    clocks = FadeClocks(U.decay, U.arm, U.rel, PING_AMP, REFRACTORY)
    field = Field(steps)
    cutmask = np.array([p in cut for p in U.pop])
    n_win = RULE["listen_to"] - RULE["listen_from"] + 1
    k = lr / (n_win * RULE["self_response"])
    tail, pk = canonical(steps)
    t_cross, rsign = None, 0
    cal_peak = U.arm / 0.8
    apic = np.zeros(U.n); apic_peak = np.zeros(U.n)
    age = -np.ones(U.n, int); acc = np.zeros((U.n, N)); gfire = np.zeros(U.n); num = den = 0.0
    closed = np.zeros(U.n, bool); last = np.zeros(U.n)
    prev_pred = np.zeros(N); prev_pulse = np.zeros(U.n); prev_out = np.zeros((U.n, N))
    out_wave = np.zeros((steps, N))
    geff = g if apical else 0.0
    for tick in range(steps):
        prev_target = target[tick - 1] if tick > 0 else np.zeros(N)
        src = np.zeros(U.n)
        if tick > 0:
            src[~U.isB] = sense_a[tick - 1]
            raw = float(prev_pred[READ["B"]])
            if not closed_eyes:
                src[U.isB] = sense_b[tick - 1]
            elif bkind == "content":
                src[U.isB] = gain * raw
            else:
                im = gain * raw
                if t_cross is None and abs(im) >= CROSS * pk:
                    t_cross, rsign = tick, (1 if im > 0 else -1)
                if t_cross is None:
                    src[U.isB] = im
                else:
                    j = tick - t_cross
                    src[U.isB] = rsign * float(tail[j]) if j < len(tail) else 0.0
            if bkind == "twoform":
                apic = U.decay * apic + (1 - U.decay) * np.maximum(0.0, U.half * raw)
                apic_peak = np.maximum(apic_peak, apic)
        x = np.maximum(0.0, U.half * src); x[closed] = 0.0
        if bkind == "twoform" and closed_eyes:
            clocks.amplitude = np.where(U.isB, PING_AMP * np.minimum(1.0, apic_peak / cal_peak), PING_AMP)
        wr = np.where(U.mask & ~cutmask[:, None], prev_out, 0.0).sum(0)
        pred = field.step(wr)
        fire = clocks.step(x)
        tip = np.where(U.mask, (prev_pred - prev_target)[None, :], 0.0)
        listening = age >= 0
        age = np.where(listening, age + 1, age)
        win = listening & (age >= RULE["listen_from"]) & (age <= RULE["listen_to"])
        acc[win] += tip[win]
        done = listening & (age == RULE["listen_to"])
        if done.any():
            a = acc[done]
            if update_g and apical:
                num += float(np.sum(a * V[done])) / n_win; den += float(np.sum(V[done] ** 2))
            if learn:
                W[done] = np.clip(W[done] - k * a, -RULE["max_weight"], RULE["max_weight"])
                V[done] = np.clip(V[done] - k * gfire[done][:, None] * a, -RULE["max_weight"], RULE["max_weight"])
            age[done] = -1
        new = np.abs(prev_pulse) > 1e-12
        age[new] = 0; acc[new] = 0.0; gfire[new] = geff
        out = prev_pulse[:, None] * (W + geff * V)
        closed = np.where(fire, False, closed | (clocks.armed & (clocks.trace < last)))
        last = clocks.trace.copy()
        prev_pred = pred.copy(); prev_pulse = clocks.pulse.copy(); prev_out = out
        out_wave[tick] = pred
    if update_g and apical:
        g = float(np.clip(g - eta_g * num / (RULE["self_response"] * (den + kappa)), -1.0, 1.0))
    return {"prediction": out_wave, "W": np.where(U.mask, W, 0.0), "V": np.where(U.mask, V, 0.0), "g": g}


_TPL = {}
def proj(P, site, t0, n, steps):
    key = (site, t0, n, steps)
    if key not in _TPL:
        _TPL[key] = passive([(SITES[site], t0, 1.0)], steps)[t0:t0 + n].ravel()
    tpl = _TPL[key]
    return float(P[t0:t0 + n].ravel() @ tpl / (tpl @ tpl))
