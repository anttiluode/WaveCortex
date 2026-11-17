"""experiments/wavecortex.png from results.json (or a given file)."""
import json, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from metrics import measures
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
C = {"twoform": "#1a7f64", "regen": "#cf6a2e", "content": "#8c959f", "cut": "#b54708", "bayes": "#1f2328",
     "plastic": "#afb8c1", "scalar": "#5b6cc4"}
path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results.json"
d = json.loads(path.read_text())["results"]; ms = [measures(b) for b in d]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED})
fig = plt.figure(figsize=(13, 8)); gsp = fig.add_gridspec(2, 3, height_ratios=[1.1, 1])
b = d[0]; R = np.array(b["rule"]); t = np.arange(len(R)); lo, hi = 0, 160
a = fig.add_subplot(gsp[0, :])
a.fill_between(t[lo:hi], -1.1, 1.1, where=R[lo:hi] > 0, color="#eef6f3", step="mid", lw=0)
a.plot(t[lo:hi], np.array(b["g"])[lo:hi], color=C["twoform"], lw=1, alpha=.35, label="apical state g")
a.plot(t[lo:hi], np.array(b["bayes"])[lo:hi], color=C["bayes"], lw=1.4, label="ideal observer")
a.plot(t[lo:hi], np.array(b["closed_regen"])[lo:hi], ".", color=C["regen"], ms=4, label="imagined B certain (regen)")
a.plot(t[lo:hi], np.array(b["closed_twoform"])[lo:hi], "o", color=C["twoform"], ms=3.5, label="imagined B sized by g (two-form)")
a.axhline(0, color=GRID, lw=.8); a.set_xlim(lo, hi); a.set_ylim(-1.1, 1.1)
a.set_xlabel("test trial (weights frozen; shaded: rule = +1)"); a.set_ylabel("eyes closed, A alone:\nexpected C (along A·ρ₂)")
a.set_title("One body: the rule on link A → B is hidden; with eyes closed, C is reached only through an imagined B", loc="left", color=INK, fontsize=10)
a.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.17))
rng = np.random.default_rng(0)
def strip(ax, cols, ylab, title, chance=True, ylim=None):
    for i, (k, lab, col) in enumerate(cols):
        v = np.array([m[k] for m in ms]); ax.scatter(i + rng.uniform(-.15, .15, len(v)), v, s=8, color=col, alpha=.5, lw=0)
        ax.plot([i - .3, i + .3], [np.nanmedian(v)] * 2, color=col, lw=2.5)
    if chance:
        ax.axhline(.5, color=INK, lw=.8, ls="--")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[1] for c in cols]); ax.set_ylabel(ylab)
    ax.set_title(title, loc="left", color=INK, fontsize=10)
    if ylim: ax.set_ylim(*ylim)
strip(fig.add_subplot(gsp[1, 0]), [("closed_twoform_acc", "chain\nintact", C["twoform"]), ("cut_closed_acc", "chain\ncut", C["cut"]),
      ("cut_seenB_acc", "fixed link\ncut", C["cut"])], "sign accuracy", "Cut the apical compartment", ylim=(0.3, 1.04))
strip(fig.add_subplot(gsp[1, 1]), [("closed_twoform_transfer", "two-form", C["twoform"]), ("scalar_transfer", "scalar\n(no waves)", C["scalar"]),
      ("bayes_transfer", "ideal", C["bayes"]), ("plastic_transfer", "plastic", C["plastic"])],
      "sign accuracy on transfer trials", "The other A sign, after a switch", ylim=(-0.03, 1.05))
strip(fig.add_subplot(gsp[1, 2]), [("closed_content_own", "content", C["content"]), ("closed_regen_own", "regen", C["regen"]),
      ("closed_twoform_own", "two-form", C["twoform"]), ("scalar_own", "scalar", C["scalar"])],
      "corr( |g| , |imagined C| )", "Imagination as unsure as the body", chance=False, ylim=(-0.5, 1.02))
fig.text(.995, .005, "%d bodies · dots: bodies · bars: medians" % len(d), ha="right", fontsize=8, color=MUTED)
fig.tight_layout(); out = HERE / ("wavecortex.png" if path.name == "results.json" else "dev_wavecortex.png"); fig.savefig(out, dpi=150); print(out)
