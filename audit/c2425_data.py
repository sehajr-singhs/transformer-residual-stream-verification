"""c26 -- aggregate the raw c24/c25 cell files for the manuscript generators.

c14 through c23 each wrote ONE summary JSON, so the generators could just
`load("c23.json")`. c24 and c25 do not: they write one file per cell into
`results/c24_parts/`, `results/c25_certified_parts/` and
`results/c25_control_parts/`, because both sweeps were run in chunks across a
dying free-tier GPU and a local CPU and had to survive being interrupted.

This module is the single place that knows how to read those directories, so
`make_manuscript_table.py` and `report_manuscript.py` cannot drift apart on how
a cell is summarised.

TWO REPORTING RULES ENCODED HERE, both learned the hard way:

  MEDIANS, NOT MEANS, for relaxation gaps. The c25 control's standard arm spans
  ~750x across three seeds and its fixnorm arm spans ~1700x across nine runs. A
  mean is dominated by its unluckiest seed. Every prior c-number in this project
  quotes a median.

  RATIOS ARE WITHHELD when either side is unstable or underpowered. c24 already
  withheld gap_gain whenever the denominator exceeded GAP_MEANINGFUL; c25's
  control denominators are worse. `tightening()` returns the ratio together with
  the n it rests on and a flag, so a caller cannot print a bare number.
"""
import glob
import json
import math
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")

GAP_MEANINGFUL = 1e3       # matches c24_scaling.GAP_MEANINGFUL
SEED_NOISE = 2.4           # measured c24 run-to-run gap spread at fixed config
MISSING = "--"             # LaTeX/markdown-safe placeholder


# ------------------------------------------------------------------ c24 cells

def load_c24(sub="c24_parts"):
    """-> {(n_layers, d_model, variant): summary dict}, plus device provenance.

    Each part file holds one cell under summary["L{L}_d{d}"][variant].
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, sub, "*.json"))):
        r = json.load(open(f))
        cfg = r.get("config", {})
        for cell, byvar in r.get("summary", {}).items():
            L = int(cell.split("_")[0][1:])
            d = int(cell.split("_")[1][1:])
            for v, s in byvar.items():
                s = dict(s)
                s["device"] = cfg.get("device")
                s["file"] = os.path.basename(f)
                out[(L, d, v)] = s
    return out


def c24_cell(c24, L, d, v, key):
    s = c24.get((L, d, v))
    if not s:
        return None
    return s.get(key)


def c24_pairing(c24, L, d):
    """'valid' only if BOTH arms exist and trained on the SAME device.

    c24_scaling.py measured CUDA-vs-CPU drift at L=4/d=128 fixnorm as ppl 6.4245
    (T4) vs 6.3851 (CPU), ~0.62% of baseline, against reported penalties of
    1-2.5%. A cross-device pair is not a valid perplexity read.
    """
    a, b = c24.get((L, d, "standard")), c24.get((L, d, "fixnorm"))
    if not a or not b:
        return "incomplete"
    return "valid" if a.get("device") == b.get("device") else "cross-device"


# ------------------------------------------------------------------ c25 cells

def load_c25(sub):
    """-> {(L, d, variant, lr): [run dicts]} for one c25 parts directory."""
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, sub, "*.json"))):
        r = json.load(open(f))
        out.setdefault((r["n_layers"], r["d_model"], r["variant"],
                        r["lr"]), []).append(r)
    return out


def arm(runs):
    """Summarise one c25 arm. Diverged runs are counted, never averaged."""
    d = {"n_total": len(runs), "n_conv": 0}
    ok = [r for r in runs if not r.get("diverged")]
    d["n_conv"] = len(ok)
    if not ok:
        return d
    ppl = [r["val_ppl"] for r in ok]
    d["ppl_mean"] = st.mean(ppl)
    d["ppl_sem"] = (st.stdev(ppl) / math.sqrt(len(ppl))) if len(ppl) > 1 else None
    g = [r["certified"]["relaxation_gap"] for r in ok
         if "relaxation_gap" in r.get("certified", {})]
    u = [r["certified"]["unstable_relus"] for r in ok
         if "unstable_relus" in r.get("certified", {})]
    if g:
        d["gap_median"] = st.median(g)
        d["gap_mean"] = st.mean(g)
        d["gap_sem"] = (st.stdev(g) / math.sqrt(len(g))) if len(g) > 1 else None
        d["gap_min"], d["gap_max"] = min(g), max(g)
        d["gap_spread"] = max(g) / min(g) if min(g) > 0 else float("inf")
        d["n_meaningful"] = sum(1 for x in g if x <= GAP_MEANINGFUL)
        d["n_gap"] = len(g)
    if u:
        d["unstable_median"] = st.median(u)
        d["unstable_max"] = max(u)
    return d


def pooled_spread(runs_by_key, keys):
    """max/min gap over every converged run in `keys`, with n.

    This is the variance-collapse statistic. Pooling across learning rates is
    deliberate and must be stated as such: it answers "how reproducible is the
    bound for this training recipe", not "how does the bound depend on lr".
    """
    g = []
    for k in keys:
        for r in runs_by_key.get(k, []):
            if not r.get("diverged") and "relaxation_gap" in r.get("certified", {}):
                g.append(r["certified"]["relaxation_gap"])
    if not g:
        return None
    return {"n": len(g), "min": min(g), "max": max(g),
            "spread": max(g) / min(g) if min(g) > 0 else float("inf"),
            "n_meaningful": sum(1 for x in g if x <= GAP_MEANINGFUL)}


def tightening(ctrl_arm, cert_arm):
    """Control/certified gap-median ratio, WITH the caveats attached.

    Never returns a bare float. `withheld` is set whenever the number would
    mislead: either arm underpowered (n<3), or the control denominator itself
    unstable enough that the ratio is a statement about one unlucky seed.
    """
    if not ctrl_arm.get("n_conv") or not cert_arm.get("n_conv"):
        return {"ratio": None, "withheld": "no paired arm"}
    if "gap_median" not in ctrl_arm or "gap_median" not in cert_arm:
        return {"ratio": None, "withheld": "no gap"}
    r = ctrl_arm["gap_median"] / cert_arm["gap_median"]
    n = min(ctrl_arm["n_conv"], cert_arm["n_conv"])
    why = None
    if n < 3:
        why = f"underpowered (n={n})"
    elif r <= SEED_NOISE:
        why = f"inside {SEED_NOISE}x seed noise"
    elif ctrl_arm.get("gap_spread", 1.0) > 10.0:
        why = (f"control denominator spans "
               f"{ctrl_arm['gap_spread']:.0f}x across seeds")
    return {"ratio": r, "n": n, "withheld": why,
            "ppl_cost_pct": ((cert_arm["ppl_mean"] - ctrl_arm["ppl_mean"])
                             / ctrl_arm["ppl_mean"] * 100)
            if "ppl_mean" in cert_arm and "ppl_mean" in ctrl_arm else None}


def fmt(x, spec="{:.4g}", missing=MISSING):
    return missing if x is None else spec.format(x)
