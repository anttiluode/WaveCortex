"""Development bodies, seeds 981-986 (not evidence)."""
import json, pathlib, sys
from concurrent.futures import ProcessPoolExecutor
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import world as Wd
if __name__ == "__main__":
    with ProcessPoolExecutor(2) as p:
        res = list(p.map(Wd.body, range(981, 987)))
    (HERE / "dev_results.json").write_text(json.dumps({"label": "development (not evidence)", "results": res}))
