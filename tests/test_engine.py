"""WaveCortex reduces exactly to WaveComputer's verified machine when the apical gate is off.

With V = 0 and the apical compartment silenced, cortex.run must give the same prediction field and the
same learned W as WaveComputer's machine.run (kept as cortex/wavecomputer_machine.py, itself retrained
against all 48 published QualiaClaude bodies to 1e-7), for training and for each eyes-closed B-unit kind.
    python tests/test_engine.py
"""
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from cortex import cortex as C, wavecomputer_machine as WC

U, UW = C.Units(), WC.Units(); ok = True
rng = np.random.default_rng(3)
W = np.where(U.mask, rng.normal(0, 0.1, (U.n, 8)), 0.0); V0 = np.zeros((U.n, 8))
for n, ev in enumerate([C.events(1, -1, 1, 20, 35, 55), C.events(-1, -1, -1, 18, 33, 55), C.events(1, 1, 1, 22, 36, 55)]):
    a = C.run(U, W, V0, 0.0, ev, 240, learn=True, lr=0.03, apical=False)
    b = WC.run(UW, W, ev, 240, learn=True, bkind="content")
    same = np.allclose(a["prediction"], b["prediction"], atol=1e-12) and np.allclose(a["W"], b["weights"], atol=1e-12)
    ok &= same; print("training episode %d: field and W identical: %s" % (n + 1, same)); W = a["W"]
for kind in ("content", "regen", "twoform"):
    ev = C.events(1, 0, 0, 23, 35, 55)
    a = C.run(U, W, V0, 0.0, ev, 240, bkind=kind, eyes="closed", gain=1.8, apical=False)
    b = WC.run(UW, W, ev, 240, bkind=kind, eyes="closed", gain=1.8)
    same = np.allclose(a["prediction"], b["prediction"], atol=1e-12); ok &= same
    print("eyes closed, %-8s field identical: %s" % (kind, same))
print("REDUCES EXACTLY" if ok else "MISMATCH"); sys.exit(0 if ok else 1)
