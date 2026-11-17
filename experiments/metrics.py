"""Per-body measures (development and batch alike). Everything is read along A * rho2, where the exact
answer is (2Q2-1)(2Q1-1)(2P(rule=+1)-1) and the right sign is the current rule."""
import numpy as np

KINDS = ("closed_content", "closed_regen", "closed_twoform", "cut_closed", "plastic", "scalar", "bayes")


def transfer_events(A, R):
    """First trial of each A sign after a rule switch, when the other A sign was already lived since it."""
    ev = []; start = 0
    for i in range(1, len(A) + 1):
        if i == len(A) or R[i] != R[i - 1]:
            seen = set()
            for j in range(start, i):
                if A[j] not in seen:
                    if -A[j] in seen and start > 0:
                        ev.append(j)
                    seen.add(A[j])
            start = i
    return ev


def measures(b):
    A, R = np.array(b["A"]), np.array(b["rule"]); bay = np.array(b["bayes"])
    ev = transfer_events(A, R); unsure = np.abs(bay) < np.median(np.abs(bay))
    out = {"n_transfer": len(ev)}
    for k in KINDS:
        x = np.array(b[k])
        out[k + "_acc"] = float(np.mean(np.sign(x) == R))
        out[k + "_mse"] = float(np.mean((x - bay) ** 2))
        out[k + "_transfer"] = float(np.mean(np.sign(x[ev]) == R[ev])) if ev else float("nan")
        out[k + "_abs"] = float(np.mean(np.abs(x)))
        out[k + "_slope"] = float(np.polyfit(bay, x, 1)[0])
        # how much smaller the prediction is when the ideal observer is unsure than when it is sure
        if k.startswith("closed") or k == "scalar":
            # how closely the size of the imagined C follows the body's own belief |g|
            out[k + "_own"] = float(np.corrcoef(np.abs(np.array(b["g"])), np.abs(x))[0, 1])
        out[k + "_shrink"] = float(np.mean(np.abs(x[unsure])) / max(np.mean(np.abs(x[~unsure])), 1e-9))
    s = np.array(b["cut_seenB"]); out["cut_seenB_acc"] = float(np.mean(s > 0)); out["cut_seenB_mean"] = float(np.mean(s))
    out["sorted"] = bool(b["absV"]["AB"] > b["absW"]["AB"] and b["absW"]["B"] > b["absV"]["B"])
    return out
