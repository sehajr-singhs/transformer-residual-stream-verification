"""Rebuild results/c14_z3.json from results/c14_z3.log.

The first full run computed every query correctly but crashed in its summary
step (margin rows key their result as `soundness_query`, not `query`). The bug
is fixed in z3_primitives.py; this recovers that run's results rather than
spending another 28 minutes of solver time to reproduce them.
"""
import re, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log = open(os.path.join(ROOT, "results", "c14_z3.log")).read()

margin, relu, ln = [], [], []
for m in re.finditer(r"box (\d+): margin ([-+\d.]+)\s+soundness=(\w+)\s+tightness=(\w+)", log):
    b, v, s, t = int(m[1]), float(m[2]), m[3], m[4]
    margin.append({"box": b, "bound": v, "soundness_query": s, "tightness_query": t,
                   "sound": s == "unsat", "tight": t == "sat"})
for m in re.finditer(r"box (\d+): relu containment = (\w+)\s+\((\d+) units, (\d+) symbols\)", log):
    relu.append({"box": int(m[1]), "query": m[2], "sound": m[2] == "unsat",
                 "conclusive": m[2] in ("sat", "unsat"),
                 "n_units": int(m[3]), "n_symbols": int(m[4])})
for m in re.finditer(r"box (\d+): layernorm containment = (\w+)", log):
    ln.append({"box": int(m[1]), "query": m[2], "sound": m[2] == "unsat",
               "conclusive": m[2] in ("sat", "unsat")})

def tally(rows, key, qkey="query"):
    return {"conclusive": sum(1 for r in rows if r[qkey] in ("sat", "unsat")),
            "attempted": len(rows), "passed": sum(1 for r in rows if r.get(key)),
            "queries": [r[qkey] for r in rows]}

rep = {"config": {"rho": 0.02, "n_boxes": 8, "tol": 1e-6, "timeout_ms": 180000,
                  "z3_version": "5.0.0", "recovered_from": "results/c14_z3.log"},
       "margin_readout": margin, "relu": relu, "layernorm": ln,
       "summary": {
           "margin_soundness": tally(margin, "sound", "soundness_query"),
           "margin_tightness": {"passed": sum(1 for r in margin if r["tight"]),
                                "attempted": len(margin),
                                "queries": [r["tightness_query"] for r in margin]},
           "relu": tally(relu, "sound"), "layernorm": tally(ln, "sound"),
           "note": ("`unknown` is a solver timeout, not a refutation. Only `unsat` "
                    "on a soundness query is a positive result; only `sat` on a "
                    "tightness query is.")},
       "runtime_sec": float(re.findall(r"\[\s*([\d.]+)s\]", log)[-1])}
p = os.path.join(ROOT, "results", "c14_z3.json")
json.dump(rep, open(p, "w"), indent=2)
print(json.dumps(rep["summary"], indent=2))
