"""c22 -- capacity-normalised Pareto scaling, unattended.

c21 could not answer whether the accuracy penalty shrinks with capacity because
both of its cells were confounded, in opposite directions: the narrow model was
undertrained at 800 steps, and the wide model's baseline sat above the
discriminatory band. This fixes both.

Design
------
For each width, a PRE-FLIGHT CALIBRATION searches the task difficulty knob until
the `standard` baseline lands mid-band (~0.85), then the full comparison runs at
that difficulty. Every model trains for the full step budget, so no cell is
undertrained, and no cell is on the ceiling. That is the level playing field the
capacity question needs.

Difficulty knob
---------------
TWO_HOP_P, the fraction of examples that use the two-hop rule (the rest are
one-hop and much easier). Continuous, monotone, and it leaves the class count
fixed so accuracies stay comparable across difficulty levels.

n_feat was tried first and is a bad knob: a pilot sweep measured 0.952 accuracy
at n_feat=8 and 0.214 at n_feat=70 (preserved in results/c22_coarseknob.log),
a cliff with no usable resolution inside an 0.80-0.90 band. The first launch of
this task was killed for that reason.

Calibration pilots run at the SAME step budget as the final runs. Calibrating at
a shorter budget would be worthless: c21 showed accuracy at 800 steps (0.72) and
3000 steps (0.897) differ by 18 points at the same width and difficulty.

Cost
----
Roughly 5-6 hours on this CPU: ~60 full runs plus up to 5 pilots per width.
Intended to be launched and left alone. Partial progress is written to the log
after every width, and the JSON is written only at the end.
"""
import sys, os, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import c20_expressivity as c20

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--widths", default="8,16,32")
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--target", type=float, default=0.85)
ap.add_argument("--band", default="0.80,0.90")
ap.add_argument("--pilot-seeds", type=int, default=1)
ap.add_argument("--max-probes", type=int, default=5)
ap.add_argument("--nfeat-fixed", type=int, default=20)
ap.add_argument("--knob-lo", type=float, default=0.0)
ap.add_argument("--knob-hi", type=float, default=1.0)
ap.add_argument("--distractor-frac", type=float, default=0.2)
ap.add_argument("--variants", default="standard,fixnorm")
ap.add_argument("--out", default=os.path.join(ROOT, "results", "c22.json"))
args = ap.parse_args()

t0 = time.time()
def stamp(m): print(f"[{time.time()-t0:8.1f}s] {m}", flush=True)

BAND = tuple(float(x) for x in args.band.split(","))
VARIANTS = args.variants.split(",")
c20.args.steps = args.steps
c20.args.batch = args.batch
c20.args.lr = args.lr


def set_difficulty(p_two_hop):
    """Point c20's task globals at a difficulty level.

    The knob is the fraction of two-hop examples, which is continuous and
    monotone. n_feat is deliberately NOT the knob: a pilot sweep measured 0.952
    at n_feat=8 and 0.214 at n_feat=70 (results/c22_coarseknob.log), a cliff
    with no usable resolution inside an 0.80-0.90 band.
    """
    p = float(min(1.0, max(0.0, p_two_hop)))
    c20.TWO_HOP_P = p
    c20.N_FEAT = args.nfeat_fixed
    c20.N_REAL = max(2, int(round(args.nfeat_fixed * (1.0 - args.distractor_frac))))
    c20.VOCAB = c20.N_FEAT + c20.N_SLOT
    return {"two_hop_p": p, "n_feat": c20.N_FEAT, "n_real": c20.N_REAL,
            "distractor_features": c20.N_FEAT - c20.N_REAL, "vocab": c20.VOCAB}


def pilot(d, knob):
    """Mean `standard` accuracy at this width and difficulty, full step budget."""
    cfg = set_difficulty(knob)
    accs = []
    for s in range(args.pilot_seeds):
        _, r = c20.train_one("standard", 1000 + s, d)
        accs.append(r["test_accuracy"])
    return float(np.mean(accs)), cfg


def calibrate(d):
    """Bisect the two-hop fraction for a mid-band baseline.

    Accuracy is decreasing in the knob, so standard bisection applies. The knob
    is continuous, which is the whole reason for switching to it.
    """
    lo, hi = args.knob_lo, args.knob_hi
    probes = []
    a_lo, cfg_lo = pilot(d, lo)
    probes.append({"knob": lo, "accuracy": a_lo})
    stamp(f"    pilot d={d} two_hop_p={lo:.3f} acc={a_lo:.4f}")
    if a_lo < BAND[0]:
        return lo, cfg_lo, probes, False
    a_hi, cfg_hi = pilot(d, hi)
    probes.append({"knob": hi, "accuracy": a_hi})
    stamp(f"    pilot d={d} two_hop_p={hi:.3f} acc={a_hi:.4f}")
    best = None
    if BAND[0] <= a_hi <= BAND[1]:
        best = (hi, cfg_hi)
    elif a_hi > BAND[1]:
        return hi, cfg_hi, probes, False
    for _ in range(args.max_probes - 2):
        mid = 0.5 * (lo + hi)
        a_mid, cfg_mid = pilot(d, mid)
        probes.append({"knob": mid, "accuracy": a_mid})
        stamp(f"    pilot d={d} two_hop_p={mid:.3f} acc={a_mid:.4f}")
        if BAND[0] <= a_mid <= BAND[1]:
            best = (mid, cfg_mid)
            if abs(a_mid - args.target) < 0.02:
                break
        if a_mid > args.target:
            lo = mid
        else:
            hi = mid
    if best is None:
        k = min(probes, key=lambda x: abs(x["accuracy"] - args.target))
        return k["knob"], set_difficulty(k["knob"]), probes, False
    return best[0], best[1], probes, True


results = {}
for d in [int(x) for x in args.widths.split(",")]:
    stamp(f"width d_model={d}: calibrating difficulty")
    knob, cfg, probes, in_band = calibrate(d)
    set_difficulty(knob)
    stamp(f"  chosen two_hop_p={knob:.3f} (vocab {cfg['vocab']}, "
          f"{cfg['distractor_features']} distractors) in_band={in_band}")

    cells = {}
    for v in VARIANTS:
        rs = []
        for s in range(args.seeds):
            model, r = c20.train_one(v, s, d)
            r["trained_gap"] = c20.gap_on_trained(model, v, d, seed=s)
            rs.append(r)
        accs = np.array([r["test_accuracy"] for r in rs])
        gaps = np.array([r["trained_gap"]["relaxation_gap"] for r in rs])
        sem = float(accs.std(ddof=1) / np.sqrt(len(accs))) if len(accs) > 1 else 0.0
        cells[v] = {
            "n_seeds": int(len(accs)),
            "test_accuracy_mean": float(accs.mean()),
            "test_accuracy_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "test_accuracy_sem": sem,
            "test_ce_mean": float(np.mean([r["test_ce"] for r in rs])),
            "relaxation_gap_median": float(np.median(gaps)),
            "relaxation_gap_iqr": [float(np.percentile(gaps, 25)),
                                   float(np.percentile(gaps, 75))],
            "frac_margin_negative_mean": float(np.mean(
                [r["frac_margin_negative"] for r in rs])),
            "max_containment_violation": float(np.max(
                [r["trained_gap"]["max_containment_violation"] for r in rs])),
            "runs": rs,
        }
        stamp(f"  d={d:3d} {v:9s} acc={accs.mean():.4f}+-{sem:.4f} "
              f"gap={np.median(gaps):.3f}")

    b = cells["standard"]
    row = {"d_model": d, "two_hop_p": knob, "n_feat": cfg["n_feat"], "task": cfg,
           "calibration_probes": probes,
           "calibration_in_band": bool(in_band),
           "baseline_accuracy": b["test_accuracy_mean"],
           "baseline_in_band": bool(BAND[0] <= b["test_accuracy_mean"] <= BAND[1]),
           "cells": cells}
    if "fixnorm" in cells:
        f = cells["fixnorm"]
        dacc = f["test_accuracy_mean"] - b["test_accuracy_mean"]
        sed = float(np.sqrt(b["test_accuracy_sem"] ** 2 + f["test_accuracy_sem"] ** 2))
        row["penalty_points"] = -dacc * 100
        row["penalty_ci95_points"] = [(-dacc - 1.96 * sed) * 100,
                                      (-dacc + 1.96 * sed) * 100]
        row["penalty_significant"] = bool((-dacc - 1.96 * sed) > 0)
        row["gap_gain"] = (b["relaxation_gap_median"]
                           / max(f["relaxation_gap_median"], 1e-300))
    results[str(d)] = row
    stamp(f"  d={d}: baseline {row['baseline_accuracy']:.4f} "
          f"in_band={row['baseline_in_band']} "
          f"penalty={row.get('penalty_points', float('nan')):.2f}pts "
          f"gap_gain={row.get('gap_gain', float('nan')):.1f}x")

widths = [int(x) for x in args.widths.split(",")]
clean = [w for w in widths if results[str(w)]["baseline_in_band"]]
trend = {
    "widths_in_band": clean,
    "all_in_band": len(clean) == len(widths),
    "penalty_points": {str(w): results[str(w)].get("penalty_points")
                       for w in widths},
    "gap_gain": {str(w): results[str(w)].get("gap_gain") for w in widths},
    "penalty_significant": {str(w): results[str(w)].get("penalty_significant")
                            for w in widths},
}
if len(clean) >= 2:
    xs = np.log(np.array(clean, float))
    ps = np.array([results[str(w)]["penalty_points"] for w in clean], float)
    gs = np.array([results[str(w)]["gap_gain"] for w in clean], float)
    trend["penalty_slope_per_log2_width"] = float(np.polyfit(xs, ps, 1)[0]
                                                  * np.log(2))
    trend["gap_gain_slope_per_log2_width"] = float(np.polyfit(xs, gs, 1)[0]
                                                   * np.log(2))
    trend["interpretation"] = (
        f"Across widths {clean}, all with the baseline held inside {BAND}, the "
        f"penalty moves by {trend['penalty_slope_per_log2_width']:+.2f} points "
        f"per doubling of d_model and the verifiability gain by "
        f"{trend['gap_gain_slope_per_log2_width']:+.2f}x per doubling.")
else:
    trend["interpretation"] = (
        "NOT INTERPRETABLE: fewer than two widths achieved a mid-band "
        "baseline, so no capacity trend can be read. The difficulty knob could "
        "not normalise capacity at the remaining widths.")

rep = {"config": {"widths": widths, "seeds": args.seeds, "steps": args.steps,
                  "batch": args.batch, "lr": args.lr, "target": args.target,
                  "band": list(BAND), "variants": VARIANTS,
                  "distractor_frac": args.distractor_frac,
                  "pilot_seeds": args.pilot_seeds},
       "results": results, "trend": trend,
       "runtime_sec": time.time() - t0,
       "note": ("Capacity-normalised: task difficulty is re-tuned per width so "
                "the standard baseline sits mid-band everywhere, and every "
                "model trains for the full step budget. This removes the two "
                "confounds that made c21 uninterpretable (undertrained narrow "
                "cell, saturated wide cell). Absolute accuracies are NOT "
                "comparable across widths by construction -- they are all "
                "pinned near the target -- which is the point: what is "
                "compared across widths is the PENALTY and the GAP GAIN.")}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(rep, open(args.out, "w"), indent=2)
stamp(f"wrote {args.out}")
print(json.dumps(trend, indent=2))
