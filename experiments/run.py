"""Pre-registered batch (PROTOCOL.md): 32 bodies, seeds 500001-500032. About 15 minutes on two cores."""
import hashlib, json, pathlib, sys, time
from concurrent.futures import ProcessPoolExecutor
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent; sys.path.insert(0, str(HERE))
import world as Wd
SEEDS = list(range(500001, 500033))
if __name__ == "__main__":
    files = ["PROTOCOL.md", "experiments/run.py", "experiments/world.py", "experiments/metrics.py", "cortex/cortex.py",
             "cortex/nodes.py", "cortex/physics.py", "cortex/units.json", "cortex/kernel.npz"]
    rec = {"label": "WaveCortex batch (pre-registered)", "seeds": SEEDS,
           "hashes": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}, "results": []}
    t0 = time.time()
    with ProcessPoolExecutor(2) as p:
        for n, r in enumerate(p.map(Wd.body, SEEDS), 1):
            rec["results"].append(r); rec["elapsed_seconds"] = time.time() - t0
            (HERE / "results.json").write_text(json.dumps(rec)); print("%d/%d %.0fs" % (n, len(SEEDS), time.time() - t0), flush=True)
