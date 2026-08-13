"""c16 -- (1) certification over a prompt distribution, (3) ReLU-guided branching.

Experiment 1: from anchor to distribution
-----------------------------------------
The baseline certificate covers one prompt anchor. Two ways to go beyond it:

  per-prompt  certify each of N anchors separately. Exact, N times the cost, and
              the guarantee is the MINIMUM certified radius over the anchors.
  hull        replace the N nominal streams by a single box hull and certify
              once. One proof for all N, but the hull contains streams that no
              prompt produces, so it is strictly more conservative.

The conservation tax is the ratio between them. Reporting only the cheap one
would be dishonest in either direction, so both run.

Experiment 3: ReLU-guided branching
-----------------------------------
c14 showed unstable ReLUs, not the LayerNorm bracket, drive the bound explosion.
The BaB branches in ALPHA space, so "split on an unstable ReLU" is not directly
expressible -- you cannot bisect a ReLU. The faithful translation is to choose
WHICH alpha coordinate to bisect by how much it would reduce layer-0 unstable
ReLU width, rather than always taking the widest coordinate.

Score for coordinate k: sum over layer-0 MLP pre-activations that are currently
unstable of |G_k|, the generator magnitude that coordinate contributes. Halving
that coordinate halves its contribution to those neurons' radii, so the score is
a direct estimate of unstable-ReLU mass removed.
"""
import sys, os, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from src import toy_transformer as tt, task, sae as sae_mod, bounds as bd, verifier as vf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "checkpoints")

ap = argparse.ArgumentParser()
ap.add_argument("--n-prompts", type=int, default=5)
ap.add_argument("--max-boxes", type=int, default=4096)
ap.add_argument("--out", default=os.path.join(ROOT, "results", "c16.json"))
ap.add_argument("--skip-branching", action="store_true",
                help="reuse a previous branching result instead of recomputing")
ap.add_argument("--branching-from", default="",
                help="path to a c16.json whose 'branching' block to carry over")
args = ap.parse_args()

t0 = time.time()
def stamp(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

model = tt.ToyTransformer()
model.load_state_dict(torch.load(os.path.join(CK, "model.pt")))
model.eval()
w = model.export_weights()
m64 = tt.ToyTransformer()
m64.load_state_dict(torch.load(os.path.join(CK, "model.pt")))
m64.eval().double()
sae = sae_mod.SAE(w["d_model"], 64)
sae.load_state_dict(torch.load(os.path.join(CK, "sae.pt")))
Wd = sae.W_dec.detach().numpy().astype(np.float64).T
Wd = Wd / np.linalg.norm(Wd, axis=1, keepdims=True)
pick = np.random.default_rng(11).choice(Wd.shape[0], size=4, replace=False)
U = np.zeros((4, w["seq_len"], w["d_model"]))
U[np.arange(4), w["seq_len"] - 1, :] = Wd[pick]
prompts, _ = task.enumerate_prompts(limit=256, rng=np.random.default_rng(0),
                                    safe_only=True)
with torch.no_grad():
    trace = m64.residual_trace(torch.from_numpy(prompts))
X0 = trace[0].numpy().astype(np.float64)          # (256, T, D)
iu, isf = task.margin_readout()
K = 4


# --------------------------------------------------------------- shared pieces

def margin_bound(a_lo, a_hi, x_nom, chunk=128):
    out = []
    for i in range(0, a_lo.shape[0], chunk):
        z = vf.alpha_to_zono(a_lo[i:i + chunk], a_hi[i:i + chunk], U) + x_nom[None]
        zL = vf._blocks_from(z, w, 0, w["n_layers"])
        out.append(bd.unsafe_margin_upper(bd.readout_logits(zL, w), iu, isf))
    return np.concatenate(out)


def margin_bound_box(a_lo, a_hi, lo0, hi0, chunk=64):
    """Same, but the nominal stream is itself a BOX [lo0, hi0]."""
    out = []
    c0 = 0.5 * (lo0 + hi0); r0 = 0.5 * (hi0 - lo0)
    for i in range(0, a_lo.shape[0], chunk):
        z = vf.alpha_to_zono(a_lo[i:i + chunk], a_hi[i:i + chunk], U)
        B = z.B
        # append the hull's own generators (one per stream coordinate)
        n = r0.size
        G_extra = np.zeros((B, n) + r0.shape)
        flat = r0.ravel()
        for j in range(n):
            G_extra.reshape(B, n, -1)[:, j, j] = flat[j]
        Z = bd.HZono(z.c + c0[None], np.concatenate([z.G, G_extra], axis=1), z.E)
        zL = vf._blocks_from(Z, w, 0, w["n_layers"])
        out.append(bd.unsafe_margin_upper(bd.readout_logits(zL, w), iu, isf))
    return np.concatenate(out)


def relu_scores(a_lo, a_hi, x_nom):
    """Per-alpha-coordinate unstable-ReLU mass at layer 0. (B, K)."""
    z0 = vf.alpha_to_zono(a_lo, a_hi, U) + x_nom[None]
    bw = w["blocks"][0]
    zc = z0.promote_E().compact(128)
    y1 = bd.layernorm(zc, bw["ln1_g"], bw["ln1_b"], w["ln_eps"]).promote_E_topk(48)
    a1, _ = bd.attention(y1, bw["WQ"], bw["WK"], bw["WV"], bw["WO"], w["n_heads"])
    z1 = zc + a1
    y2 = bd.layernorm(z1, bw["ln2_g"], bw["ln2_b"], w["ln_eps"]).promote_E_topk(48)
    pre = y2.linear(bw["fc_in_W"], bw["fc_in_b"])
    lo, hi = pre.bounds()
    unstable = (lo < 0) & (hi > 0)                       # (B,T,d_mlp)
    G = np.abs(pre.G[:, :K])                             # (B,K,T,d_mlp)
    return (G * unstable[:, None]).reshape(G.shape[0], K, -1).sum(-1)


# 2**12 = 4096, so 13 iterations is all a max_boxes=4096 budget can use. The
# default of 20 lets a FAILING radius keep re-evaluating a full frontier of
# boxes for eight more rounds that cannot change the verdict, which dominated
# the runtime of this sweep (~1000s per failing radius against ~250s for a
# succeeding one).
def bab(eval_fn, rho, rule, max_boxes, max_iters=13, min_width=1e-5,
        score_fn=None):
    a_lo = np.full((1, K), -rho); a_hi = np.full((1, K), rho)
    touched = 0
    for it in range(max_iters):
        ub = eval_fn(a_lo, a_hi)
        ub = np.where(np.isfinite(ub), ub, np.inf)
        keep = ub >= 0.0
        a_lo, a_hi = a_lo[keep], a_hi[keep]
        touched += int(keep.sum())
        if a_lo.shape[0] == 0:
            return True, {"proved": True, "iterations": it + 1,
                          "boxes_touched": touched, "rule": rule}
        wd = a_hi - a_lo
        if wd.max(axis=1).min() < min_width or a_lo.shape[0] * 2 > max_boxes:
            return False, {"proved": False, "reason": "max_boxes",
                           "worst": float(ub[np.isfinite(ub)].max()),
                           "boxes_touched": touched, "rule": rule}
        if rule == "widest":
            dim = wd.argmax(axis=1)
        else:
            sc = score_fn(a_lo, a_hi)
            # never split a coordinate already at the width floor
            sc = np.where(wd > min_width, sc, -np.inf)
            dim = sc.argmax(axis=1)
        r = np.arange(a_lo.shape[0])
        mid = 0.5 * (a_lo[r, dim] + a_hi[r, dim])
        l1, h1 = a_lo.copy(), a_hi.copy(); h1[r, dim] = mid
        l2, h2 = a_lo.copy(), a_hi.copy(); l2[r, dim] = mid
        a_lo = np.concatenate([l1, l2]); a_hi = np.concatenate([h1, h2])
    return False, {"proved": False, "reason": "max_iters",
                   "boxes_touched": touched, "rule": rule}


def max_radius(eval_fn, rule, grid, score_fn=None, max_boxes=None):
    best, rows = None, {}
    for rho in grid:
        ok, st = bab(eval_fn, rho, rule, max_boxes or args.max_boxes,
                     score_fn=score_fn)
        rows[str(rho)] = {"proved": bool(ok), **{k: v for k, v in st.items()
                                                 if k != "rule"}}
        if ok:
            best = rho
        else:
            break
    return best, rows


# 0.06 is omitted: baseline.json already establishes the prover fails there, and
# max_radius stops at the first failure anyway.
GRID = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
rep = {"config": {"n_prompts": args.n_prompts, "k": K,
                  "feature_ids": pick.tolist(), "grid": GRID,
                  "max_boxes": args.max_boxes}}

# ------------------------------------------------- experiment 3: branching rule
if args.skip_branching:
    src = args.branching_from or args.out
    rep["branching"] = json.load(open(src))["branching"]
    stamp(f"experiment 3: carried over from {src} "
          f"(box budget {rep['branching'].get('max_boxes', 'see that file')})")
else:
    stamp("experiment 3: branching rule (anchor prompt 0)")
    x0 = X0[0]
    res_b = {}
    for rule in ("widest", "relu_guided"):
        t = time.time()
        best, rows = max_radius(lambda l, h: margin_bound(l, h, x0), rule, GRID,
                                score_fn=lambda l, h: relu_scores(l, h, x0))
        res_b[rule] = {"max_certified_rho": best, "per_rho": rows,
                       "seconds": time.time() - t}
        stamp(f"  {rule:12s} max rho = {best}  ({time.time()-t:.0f}s)")
    res_b["max_boxes"] = args.max_boxes
    rep["branching"] = res_b
    bw_ = res_b["widest"]["max_certified_rho"]
    br_ = res_b["relu_guided"]["max_certified_rho"]
    rep["branching"]["verdict"] = {
        "improved": bool(br_ is not None and bw_ is not None and br_ > bw_),
        "radius_ratio": (br_ / bw_) if (br_ and bw_) else None,
        "gap_before": (10.0 / bw_) if bw_ else None,
        "gap_after": (10.0 / br_) if br_ else None,
    }

# --------------------------------------- experiment 1: anchor -> distribution
stamp(f"experiment 1: {args.n_prompts}-prompt distribution")
per_prompt = {}
for p in range(args.n_prompts):
    xp = X0[p]
    best, rows = max_radius(lambda l, h: margin_bound(l, h, xp), "widest", GRID)
    per_prompt[str(p)] = {"max_certified_rho": best,
                          "tokens": prompts[p].tolist()}
    stamp(f"  prompt {p}: max rho = {best}")
radii = [v["max_certified_rho"] for v in per_prompt.values()]
worst = min([r for r in radii if r is not None], default=None)

stamp("  hull over the same prompts")
sub = X0[:args.n_prompts]
lo0, hi0 = sub.min(axis=0), sub.max(axis=0)
hull_best, hull_rows = max_radius(lambda l, h: margin_bound_box(l, h, lo0, hi0),
                                  "widest", GRID)
stamp(f"  hull: max rho = {hull_best}")

rep["distribution"] = {
    "per_prompt": per_prompt,
    "worst_case_over_prompts": worst,
    "hull": {"max_certified_rho": hull_best, "per_rho": hull_rows,
             "hull_width_mean": float((hi0 - lo0).mean()),
             "hull_width_max": float((hi0 - lo0).max()),
             "extra_generators": int(lo0.size)},
    "conservation_tax": {
        "per_prompt_worst": worst,
        "hull": hull_best,
        "ratio": (worst / hull_best) if (worst and hull_best) else None,
        "note": ("ratio = how much radius the single-proof hull gives up "
                 "against certifying each anchor separately. null hull means "
                 "the hull certified nothing on the grid."),
    },
}
rep["runtime_sec"] = time.time() - t0
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(rep, open(args.out, "w"), indent=2)
stamp(f"wrote {args.out}")
print(json.dumps({"branching": rep["branching"]["verdict"],
                  "tax": rep["distribution"]["conservation_tax"]}, indent=2))
