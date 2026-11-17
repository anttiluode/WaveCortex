"""The fused world: a hidden rule on the FIRST link of a chain, read out through imagination.

  A = +-1 (balanced)
  true B = rule * A, flipped with probability 1 - Q1
  C = rho2 * true B, flipped with probability 1 - Q2        (rho2 = +-1 per body: a FIXED second link)
  B and C are seen in training and in every lived test trial; the rule flips, uncued, after 20-40 trials.

The question, every test trial, weights frozen: with eyes closed and A alone, what does the body expect C
to be? The only route is A -> imagined B -> C, and the imagined B must carry the body's belief about the
rule. The exact answer, along A * rho2:   E[C] = (2Q2 - 1)(2Q1 - 1)(2 P(rule = +1) - 1).

Readouts (each test trial, before living it, nothing updates):
  closed_content, closed_regen, closed_twoform   eyes closed, A alone -> C, with each B-unit kind
  cut_closed                                      the same with the apical compartment silenced (g = 0)
  cut_seenB                                       apical silenced, A and a seen B -> C: the fixed link
Baselines on the same sequences: bayes (exact filter over rule and block position), plastic (per-A-sign
delta rule on C, best rate on its own training), scalar (the fused algorithm with no waves).
"""
from __future__ import annotations
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cortex import cortex as M
from cortex.physics import passive

Q1, Q2 = 0.9, 0.9
BLOCK = (20, 40)
TRAIN, TEST = 1500, 400
STEPS, PROBE, GAP2_EXTRA = 240, 23, 20
LR, ETA_G, KAPPA = 0.006, 0.5, 0.05          # ApicalRuleReversalMachine's, unchanged


def sequence(rng, n):
    A, B, C, R = [], [], [], []
    r = int(rng.choice([1, -1])); left = int(rng.integers(BLOCK[0], BLOCK[1] + 1))
    for _ in range(n):
        if left == 0:
            r = -r; left = int(rng.integers(BLOCK[0], BLOCK[1] + 1))
        a = int(rng.choice([1, -1])); b = r * a * (1 if rng.random() < Q1 else -1)
        A.append(a); B.append(b); R.append(r); C.append(b * (1 if rng.random() < Q2 else -1)); left -= 1
    return np.array(A), np.array(B), np.array(C), np.array(R)       # C here is along rho2 (multiply later)


def bayes_rule(A, B):
    """P(rule = +1) before each trial, exact over (rule, trials into the block); learns from (A, B)."""
    lo, hi = BLOCK; D = hi + 1
    dur = np.zeros(D + 1); dur[lo:hi + 1] = 1.0 / (hi - lo + 1); surv = 1 - np.cumsum(dur)
    haz = np.array([dur[d] / max(surv[d - 1] if d > 0 else 1.0, 1e-12) for d in range(D)])
    p = np.zeros((2, D)); p[:, 0] = 0.5; out = []
    for a, b in zip(A, B):
        new = np.zeros_like(p)
        for d in range(D):
            h = min(1.0, haz[d]) if d >= lo else 0.0
            if d + 1 < D:
                new[:, d + 1] += p[:, d] * (1 - h)
            new[::-1, 1] += p[:, d] * h
        p = new / new.sum(); out.append(p[0].sum())
        like = np.array([[Q1 if r * a == b else 1 - Q1] for r in (1, -1)]); p = p * like; p /= p.sum()
    return np.array(out)


def plastic(A, C, alpha, w=None):
    w = np.zeros(2) if w is None else w.copy(); out = []
    for a, c in zip(A, C):
        i = 0 if a > 0 else 1; out.append(w[i]); w[i] += alpha * (c - w[i])
    return np.array(out), w


def scalar(A, B, C, eta=ETA_G, lr=0.01, kappa=0.1, state=None, learn=True):
    """The fused algorithm with no waves. b_hat = w_a + g v_a; imagined C = u * b_hat (sized imagination)."""
    if state is None:
        w, v, u, g = np.zeros(2), np.zeros(2), 0.0, 1.0
    else:
        w, v, u, g = state[0].copy(), state[1].copy(), state[2], state[3]
    out = []
    for a, b, c in zip(A, B, C):
        i = 0 if a > 0 else 1
        bh = w[i] + g * v[i]; out.append(u * bh)
        e = bh - b; gn = float(np.clip(g - eta * e * v[i] / (v[i] ** 2 + kappa), -1, 1))
        if learn:
            w[i] -= lr * e; v[i] -= lr * g * e; u -= lr * (u * b - c) * b
        g = gn
    return np.array(out), (w, v, u, g)


def body(seed, n_test=TEST):
    rng = np.random.default_rng(seed)
    U = M.Units()
    gap = int(rng.integers(28, 43)); amp = float(rng.uniform(0.9, 1.12)); rho2 = 1 if seed % 2 else -1
    tr = sequence(rng, TRAIN); te = sequence(rng, n_test)
    jit = rng.integers(-5, 6, TRAIN + n_test); gj = rng.integers(-3, 4, TRAIN + n_test)
    W = np.zeros((U.n, 8)); V = np.zeros((U.n, 8)); g = 1.0
    for i, (a, b, c) in enumerate(zip(*tr[:3])):
        on = 20 + int(jit[i]); g1 = gap + int(gj[i])
        r = M.run(U, W, V, g, M.events(a, b, rho2 * c, on, g1, gap + GAP2_EXTRA, amp), STEPS,
                  learn=True, lr=LR, update_g=True, eta_g=ETA_G, kappa=KAPPA)
        W, V, g = r["W"], r["V"], r["g"]
    g2 = gap + GAP2_EXTRA; b0 = PROBE + gap; c0 = b0 + g2
    # body gain G for content/regen: seen-B peak over the imagined-B peak at full belief (as relay/)
    pk = []
    for a in (1, -1):
        q = M.run(U, W, V, 1.0, M.events(a, 0, 0, PROBE, gap, g2), STEPS, cut=("B",))
        pk.append(float(np.max(np.abs(q["prediction"][b0 - 5:b0 + 30, M.READ["B"]]))))
    seen_pk = float(np.max(np.abs(passive([(M.SITES["B"], b0, 1.0)], STEPS)[:, M.READ["B"]])))
    G = seen_pk / max(np.mean(pk), 1e-9)
    A, B, C, R = te
    rows = {k: [] for k in ("closed_content", "closed_regen", "closed_twoform", "cut_closed", "cut_seenB", "g")}
    for i, (a, b, c) in enumerate(zip(A, B, C)):
        ev0 = M.events(a, 0, 0, PROBE, gap, g2); al = a * rho2 / amp       # read "along A * rho2"
        for kind in ("content", "regen", "twoform"):
            q = M.run(U, W, V, g, ev0, STEPS, bkind=kind, eyes="closed", gain=G)
            rows["closed_" + kind].append(M.proj(q["prediction"], "C", c0, 40, STEPS) * al)
        q = M.run(U, W, V, g, ev0, STEPS, bkind="twoform", eyes="closed", gain=G, apical=False)
        rows["cut_closed"].append(M.proj(q["prediction"], "C", c0, 40, STEPS) * al)
        q = M.run(U, W, V, g, M.events(a, b, 0, PROBE, gap, g2), STEPS, apical=False)
        rows["cut_seenB"].append(M.proj(q["prediction"], "C", c0, 40, STEPS) * b * rho2 / amp)   # along the seen B
        rows["g"].append(g)
        j = TRAIN + i; on = 20 + int(jit[j]); g1 = gap + int(gj[j])
        g = M.run(U, W, V, g, M.events(a, b, rho2 * c, on, g1, g2, amp), STEPS, update_g=True, eta_g=ETA_G, kappa=KAPPA)["g"]
    P = bayes_rule(A, B)
    bay = (2 * Q2 - 1) * (2 * Q1 - 1) * (2 * P - 1)            # along A * rho2
    grid = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7]
    alpha = grid[int(np.argmin([np.mean((plastic(tr[0], tr[2], al_)[0][100:] - tr[2][100:]) ** 2) for al_ in grid]))]
    w0 = plastic(tr[0], tr[2], alpha)[1]
    pp = plastic(A, C, alpha, w0)[0] * A                         # its C prediction, read along A
    _, st = scalar(tr[0], tr[1], tr[2])
    ps = scalar(A, B, C, state=st, learn=False)[0] * A
    by = lambda pop, half=None: U.index(pop, half)
    return {"seed": seed, "gap": gap, "amp": amp, "rho2": rho2, "G": G, "alpha": alpha,
            "A": A.tolist(), "B": B.tolist(), "C": C.tolist(), "rule": R.tolist(), "bayes": bay.tolist(), "P": P.tolist(),
            "plastic": pp.tolist(), "scalar": ps.tolist(), **{k: list(map(float, v)) for k, v in rows.items()},
            "absW": {p: float(np.abs(W[by(p)]).sum()) for p in ("AB", "AC", "B")},
            "absV": {p: float(np.abs(V[by(p)]).sum()) for p in ("AB", "AC", "B")}}
