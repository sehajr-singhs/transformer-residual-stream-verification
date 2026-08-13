"""c21 -- does the accuracy penalty shrink or grow with capacity?

c20 measured a Pareto exchange at d_model=8: `fixnorm` costs 5.7 accuracy
points and buys 17.1x verifiability. The obvious next question is whether the
penalty closes as the model gets wider. This runs the same non-saturated task
at d_model 8 and 16 with 10 seeds each, dropping the dominated
`fixnorm_tanh`.

A confound this experiment must not walk into
---------------------------------------------
c20's whole lesson was that a saturated baseline reports a penalty of exactly
zero whether or not one exists. A width sweep is exposed to precisely that
failure: if the task saturates at the larger width, `standard` sits on the
ceiling, the measured penalty collapses, and "the penalty shrinks with
capacity" becomes indistinguishable from "the task ran out of headroom".

So this script reports, per width, whether the baseline is inside the
discriminatory band, and refuses to interpret the trend when it is not. The
saturation status is a first-class output, not a footnote.
"""
import sys, os, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import c20_expressivity as c20

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--widths", default="8,16")
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--n-feat", type=int, default=20)
ap.add_argument("--variants", default="standard,fixnorm")
ap.add_argument("--band", default="0.70,0.95")
ap.add_argument("--out", default=os.path.join(ROOT, "results", "c21.json"))
args = ap.parse_args()

t0 = time.time()
def stamp(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

# drive c20's module-level task/config globals
c20.args.steps = args.steps
c20.args.batch = args.batch
c20.args.lr = args.lr
c20.N_FEAT = args.n_feat
c20.VOCAB = c20.N_FEAT + c20.N_SLOT

BAND = tuple(float(x) for x in args.band.split(","))
VARIANTS = args.variants.split(",")
widths = [int(x) for x in args.widths.split(",")]

runs = []
for d in widths:
    for v in VARIANTS:
        for s in range(args.seeds):
            model, r = c20.train_one(v, s, d)
            r["d_model"] = d
            r["trained_gap"] = c20.gap_on_trained(model, v, d, seed=s)
            runs.append(r)
        acc = [x["test_accuracy"] for x in runs
               if x["d_model"] == d and x["variant"] == v]
        stamp(f"  d={d:3d} {v:9s} acc={np.mean(acc):.4f}+-{np.std(acc):.4f} "
              f"n={len(acc)}")

summary = {}
for d in widths:
    row = {}
    for v in VARIANTS:
        rs = [r for r in runs if r["d_model"] == d and r["variant"] == v]
        accs = np.array([r["test_accuracy"] for r in rs])
        gaps = np.array([r["trained_gap"]["relaxation_gap"] for r in rs])
        # 95% CI on the mean, normal approximation
        se = float(accs.std(ddof=1) / np.sqrt(len(accs))) if len(accs) > 1 else 0.0
        row[v] = {
            "n_seeds": len(accs),
            "test_accuracy_mean": float(accs.mean()),
            "test_accuracy_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
            "test_accuracy_sem": se,
            "test_accuracy_ci95": [float(accs.mean() - 1.96 * se),
                                   float(accs.mean() + 1.96 * se)],
            "test_ce_mean": float(np.mean([r["test_ce"] for r in rs])),
            "relaxation_gap_median": float(np.median(gaps)),
            "frac_margin_negative_mean": float(np.mean(
                [r["frac_margin_negative"] for r in rs])),
            "max_containment_violation": float(np.max(
                [r["trained_gap"]["max_containment_violation"] for r in rs])),
        }
    b = row["standard"]
    row["baseline_accuracy"] = b["test_accuracy_mean"]
    row["baseline_in_band"] = bool(BAND[0] <= b["test_accuracy_mean"] <= BAND[1])
    if "fixnorm" in row:
        f = row["fixnorm"]
        d_acc = f["test_accuracy_mean"] - b["test_accuracy_mean"]
        # unpaired 95% CI on the difference of means
        sed = float(np.sqrt(b["test_accuracy_sem"] ** 2 + f["test_accuracy_sem"] ** 2))
        row["penalty_points"] = -d_acc * 100
        row["penalty_ci95_points"] = [(-d_acc - 1.96 * sed) * 100,
                                      (-d_acc + 1.96 * sed) * 100]
        row["penalty_significant"] = bool((-d_acc - 1.96 * sed) > 0)
        row["gap_gain"] = (b["relaxation_gap_median"]
                           / max(f["relaxation_gap_median"], 1e-300))
    summary[str(d)] = row

trend = {}
if len(widths) >= 2 and all("penalty_points" in summary[str(d)] for d in widths):
    lo, hi = str(widths[0]), str(widths[-1])
    trend = {
        "penalty_points": {lo: summary[lo]["penalty_points"],
                           hi: summary[hi]["penalty_points"]},
        "gap_gain": {lo: summary[lo]["gap_gain"], hi: summary[hi]["gap_gain"]},
        "both_widths_in_band": bool(summary[lo]["baseline_in_band"]
                                    and summary[hi]["baseline_in_band"]),
    }
    if not trend["both_widths_in_band"]:
        oob = [w for w in (lo, hi) if not summary[w]["baseline_in_band"]]
        oob_acc = ", ".join(f"{summary[w]['baseline_accuracy']:.4f}" for w in oob)
        trend["interpretation"] = (
            "NOT INTERPRETABLE as a capacity trend. The baseline is outside the "
            f"discriminatory band {BAND} at d_model={','.join(oob)} "
            f"(accuracy {oob_acc}). "
            "A saturated baseline reports a shrinking penalty whether or not "
            "one exists -- this is exactly the artifact c20 exposed. To read a "
            "capacity trend the task must be re-tuned so the baseline stays "
            "off the ceiling at every width.")
    else:
        d0 = summary[lo]["penalty_points"]; d1 = summary[hi]["penalty_points"]
        trend["interpretation"] = (
            f"Penalty moves from {d0:.1f} to {d1:.1f} points between "
            f"d_model={lo} and {hi}, with both baselines inside the band.")

rep = {"config": {"widths": widths, "seeds": args.seeds, "steps": args.steps,
                  "batch": args.batch, "lr": args.lr, "n_feat": args.n_feat,
                  "variants": VARIANTS, "band": list(BAND),
                  "task": {"N_FEAT": c20.N_FEAT, "N_REAL": c20.N_REAL,
                           "N_SLOT": c20.N_SLOT, "SEQ_LEN": c20.SEQ_LEN,
                           "VOCAB": c20.VOCAB}},
       "runs": runs, "summary": summary, "trend": trend,
       "note": ("fixnorm_tanh dropped: c20 showed it dominated (3 more accuracy "
                "points lost, no additional verifiability). Step count is 2000 "
                "here versus 3000 in c20, so absolute accuracies are not "
                "directly comparable across the two experiments; the "
                "within-experiment comparison is.")}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(rep, open(args.out, "w"), indent=2)
stamp(f"wrote {args.out}")
print(json.dumps({"summary": {k: {"baseline_accuracy": v["baseline_accuracy"],
                                  "in_band": v["baseline_in_band"],
                                  "penalty_points": v.get("penalty_points"),
                                  "gap_gain": v.get("gap_gain")}
                              for k, v in summary.items()},
                  "trend": trend}, indent=2))
