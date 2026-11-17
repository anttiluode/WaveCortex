"""Score against PROTOCOL.md.   python experiments/analyze.py [results.json]"""
import json, math, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from metrics import measures, KINDS


def summarize(path):
    ms = [measures(b) for b in json.loads(pathlib.Path(path).read_text())["results"]]
    n = len(ms); need = math.ceil(0.9 * n)
    col = lambda k: np.array([m[k] for m in ms]); med = lambda k: float(np.nanmedian(col(k)))
    R = {}
    R["P1_sorted"] = {"bodies": int(col("sorted").sum()), "pass": bool(col("sorted").sum() >= need)}
    R["P2_chain_follows_rule"] = {"twoform_acc": med("closed_twoform_acc"), "bayes_acc": med("bayes_acc"),
                                  "pass": med("closed_twoform_acc") >= 0.80}
    above = int((col("closed_twoform_transfer") > col("plastic_transfer")).sum())
    R["P3_transfer"] = {"twoform": med("closed_twoform_transfer"), "plastic": med("plastic_transfer"), "bayes": med("bayes_transfer"),
                        "twoform_above_plastic": above,
                        "pass": bool(med("closed_twoform_transfer") >= 0.65 and med("plastic_transfer") <= 0.25 and above >= need)}
    R["P4_double_dissociation"] = {"cut_chain_acc": med("cut_closed_acc"), "cut_fixed_link_acc": med("cut_seenB_acc"),
                                   "pass": bool(0.35 <= med("cut_closed_acc") <= 0.65 and med("cut_seenB_acc") >= 0.95)}
    R["P5_own_belief"] = {"twoform_corr": med("closed_twoform_own"), "pass": med("closed_twoform_own") >= 0.70}
    bc = int((col("closed_twoform_mse") < col("closed_content_mse")).sum()); bp = int((col("closed_twoform_mse") < col("plastic_mse")).sum())
    R["P6_closer_to_ideal"] = {"twoform_mse": med("closed_twoform_mse"), "content_mse": med("closed_content_mse"),
                               "plastic_mse": med("plastic_mse"), "better_than_content": bc, "better_than_plastic": bp,
                               "pass": bool(bc >= need and bp >= need)}
    rep = {k: {x: med(k + "_" + x) for x in ("acc", "mse", "transfer", "slope", "shrink", "abs")} for k in KINDS}
    for k in ("closed_content", "closed_regen", "closed_twoform", "scalar"):
        rep[k]["own"] = med(k + "_own")
    rep["twoform_mse_below_regen"] = int((col("closed_twoform_mse") < col("closed_regen_mse")).sum())
    rep["twoform_own_above_regen"] = int((col("closed_twoform_own") > col("closed_regen_own")).sum())
    rep["scalar_mse_below_twoform"] = int((col("scalar_mse") < col("closed_twoform_mse")).sum())
    rep["cut_fixed_link_mean"] = med("cut_seenB_mean")
    return {"n": n, "need_90pct": need, "rules": R, "report": rep, "all_rules_pass": all(v["pass"] for v in R.values())}


if __name__ == "__main__":
    path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results.json"
    S = summarize(path)
    (HERE / ("summary.json" if path.name == "results.json" else "dev_summary.json")).write_text(json.dumps(S, indent=1))
    for k, v in S["rules"].items():
        print("%-24s %s %s" % (k, "PASS" if v["pass"] else "FAIL", {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a != "pass"}))
    print("all rules pass:", S["all_rules_pass"])
