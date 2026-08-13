"""Recover the branching comparison from results/c16_budget4096.log.

That run completed both branching rules at the full max_boxes=4096 budget and
then died partway through the prompt sweep. The distribution sweep was rerun at
a smaller budget, so the two results must never be conflated -- provenance and
box budget are recorded explicitly in the output.
"""
import json, re, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "results", "c16_budget4096.log")
OUT = os.path.join(ROOT, "results", "c16_branching.json")

log = open(LOG).read()
pairs = re.findall(r"(widest|relu_guided)\s+max rho = ([\d.]+)\s+\((\d+)s\)", log)
rho = {k: float(v) for k, v, _ in pairs}
sec = {k: float(s) for k, _, s in pairs}
bw, br = rho["widest"], rho["relu_guided"]

out = {"branching": {
    "widest": {"max_certified_rho": bw, "seconds": sec["widest"], "per_rho": None},
    "relu_guided": {"max_certified_rho": br, "seconds": sec["relu_guided"],
                    "per_rho": None},
    "max_boxes": 4096,
    "grid_top": 0.05,
    "provenance": f"recovered from {os.path.basename(LOG)} (max_boxes=4096)",
    "verdict": {
        "improved": bool(br > bw),
        "radius_ratio": br / bw,
        "gap_before": 10.0 / bw,
        "gap_after": 10.0 / br,
        "slowdown": sec["relu_guided"] / sec["widest"],
    },
}}
json.dump(out, open(OUT, "w"), indent=2)
print(json.dumps(out["branching"]["verdict"], indent=2))
print(f"wrote {OUT}")
