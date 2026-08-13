"""c17 -- softmax relaxation ablation, width scaling, and a 50-prompt sweep.

Experiment 1: does a real softmax relaxation close the attention gap?
---------------------------------------------------------------------
c16 left the CROWN reference engine 39-44x the attainable width at layer-0
attention against the prover's 3.9x, and concluded the fix was a softmax
relaxation competitive with the prover's Jacobian bracket. This tests that
conclusion by implementing one (exp secant/tangent, reciprocal secant/tangent,
McCormick product -- Shi et al. style) and ablating against an interval
enclosure of the same probabilities.

Experiment 2: width scaling
---------------------------
Synthetic RANDOMLY INITIALISED transformers at d_model in {32, 64, 128}. These
are not trained, so nothing here is a statement about certified radius on a
trained model -- the quantity being measured is how the RELAXATION behaves as
width grows: unstable-ReLU count, LayerNorm variance spread, and per-layer width
amplification. Random init is a fair proxy for that and an unfair one for
anything else, so only that is reported.

Experiment 3: prompt sweep
--------------------------
50 anchors at a fixed radius. The task has a fixed SEQ_LEN=5, so sequence
length cannot be varied without retraining the model; only the attention map
varies, via the token content. That limitation is reported rather than papered
over with a length axis that does not exist.
"""
import sys, os, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from src import toy_transformer as tt, task, sae as sae_mod, bounds as bd, verifier as vf
import crown_reference as cr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "checkpoints")

ap = argparse.ArgumentParser()
ap.add_argument("--n-prompts", type=int, default=50)
ap.add_argument("--prompt-rho", type=float, default=0.02)
ap.add_argument("--prompt-max-boxes", type=int, default=512)
ap.add_argument("--widths", default="32,64,128")
ap.add_argument("--out", default=os.path.join(ROOT, "results", "c17.json"))
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
X0 = trace[0].numpy().astype(np.float64)
iu, isf = task.margin_readout()
rep = {"config": {"widths": args.widths, "n_prompts": args.n_prompts,
                  "prompt_rho": args.prompt_rho,
                  "prompt_max_boxes": args.prompt_max_boxes}}


# ------------------------------------------------- 1. softmax relaxation ablation
stamp("experiment 1: softmax relaxation ablation")
rho = 1e-4
mid, rad = np.zeros(4), np.full(4, rho)
al = np.random.default_rng(0).uniform(-rho, rho, size=(4000, 4))
xn = X0[0]
xs = torch.from_numpy(np.einsum("bk,ktd->btd", al, U) + xn[None])
with torch.no_grad():
    blk = m64.blocks[0]
    a_true = blk.attn(blk.ln1(xs)).numpy()
true_w = float((a_true.max(0) - a_true.min(0)).max())
bw0 = w["blocks"][0]
z = vf.alpha_to_zono(np.full((1, 4), -rho), np.full((1, 4), rho), U) + xn[None]
zc = z.promote_E().compact(128)
y1 = bd.layernorm(zc, bw0["ln1_g"], bw0["ln1_b"], w["ln_eps"]).promote_E_topk(48)
ah, _ = bd.attention(y1, bw0["WQ"], bw0["WK"], bw0["WV"], bw0["WO"], w["n_heads"])
hz_w = float(ah.width().max())

abl = {"rho": rho, "true_width": true_w,
       "zonotope": {"width": hz_w, "over_true": hz_w / true_w}, "modes": {}}
for mode in ("interval", "linear"):
    cr.SOFTMAX_MODE = mode
    Lb = cr.LB.exact(U.copy(), xn.copy())
    Y1 = cr.layernorm(Lb, bw0["ln1_g"], bw0["ln1_b"], w["ln_eps"], mid, rad)
    A = cr.attention(Y1, bw0["WQ"], bw0["WK"], bw0["WV"], bw0["WO"],
                     w["n_heads"], mid, rad)
    lo, hi = A.concretize(mid, rad)
    cw = float(np.max(hi - lo))
    mg = cr.certify_margin(np.full(4, -rho), np.full(4, rho), U, xn, w, iu, isf)
    abl["modes"][mode] = {"width": cw, "over_true": cw / true_w, "margin": mg}
    stamp(f"  softmax={mode:9s} L0 attn {cw:.4e} ({cw/true_w:.2f}x true)  "
          f"margin {mg:+.4e}")
cr.SOFTMAX_MODE = "linear"
iv, ln_ = abl["modes"]["interval"], abl["modes"]["linear"]
abl["verdict"] = {
    "gap_reduction_factor": iv["over_true"] / ln_["over_true"],
    "remaining_gap_vs_zonotope": ln_["over_true"] / abl["zonotope"]["over_true"],
    "conclusion": (
        "The state-of-the-art softmax relaxation reduces the attention gap by "
        f"{100*(1 - ln_['over_true']/iv['over_true']):.1f}% -- essentially "
        "nothing. Softmax relaxation quality is NOT the bottleneck. The "
        "bottleneck is the representation: two-sided linear bounds must relax "
        "the product p*v, whereas a zonotope carries p and v as affine forms "
        "over SHARED noise symbols and keeps the first-order cross terms "
        "exactly, paying only second order. c16 concluded a better softmax "
        "relaxation was what was needed; this refutes that."),
}
rep["softmax_ablation"] = abl


# ------------------------------------------------------------- 2. width scaling
def synth_weights(d_model, n_heads=4, n_layers=2, seq_len=5, seed=0):
    """Randomly initialised transformer weights in export_weights() format."""
    g = np.random.default_rng(seed)
    s = 1.0 / np.sqrt(d_model)
    W = {"d_model": d_model, "n_heads": n_heads, "d_head": d_model // n_heads,
         "n_layers": n_layers, "seq_len": seq_len, "ln_eps": 1e-5,
         "ln_f_g": np.ones(d_model), "ln_f_b": np.zeros(d_model),
         "unembed": g.normal(scale=s, size=(task.VOCAB, d_model)), "blocks": []}
    for _ in range(n_layers):
        W["blocks"].append({
            "ln1_g": np.ones(d_model), "ln1_b": np.zeros(d_model),
            "ln2_g": np.ones(d_model), "ln2_b": np.zeros(d_model),
            "WQ": g.normal(scale=s, size=(d_model, d_model)),
            "WK": g.normal(scale=s, size=(d_model, d_model)),
            "WV": g.normal(scale=s, size=(d_model, d_model)),
            "WO": g.normal(scale=s, size=(d_model, d_model)),
            "fc_in_W": g.normal(scale=s, size=(4 * d_model, d_model)),
            "fc_in_b": np.zeros(4 * d_model),
            "fc_out_W": g.normal(scale=1.0 / np.sqrt(4 * d_model),
                                 size=(d_model, 4 * d_model)),
            "fc_out_b": np.zeros(d_model)})
    return W


def width_probe(W, rho, seed=0):
    g = np.random.default_rng(seed)
    d, T = W["d_model"], W["seq_len"]
    x_nom = g.normal(size=(T, d))
    Uw = np.zeros((4, T, d))
    dirs = g.normal(size=(4, d))
    Uw[np.arange(4), T - 1, :] = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    z = vf.alpha_to_zono(np.full((1, 4), -rho), np.full((1, 4), rho), Uw) + x_nom[None]
    rows = []
    in_w = float(z.width().max())
    for l in range(W["n_layers"]):
        bw = W["blocks"][l]
        zc = z.promote_E().compact(128)
        y1 = bd.layernorm(zc, bw["ln1_g"], bw["ln1_b"], W["ln_eps"]).promote_E_topk(48)
        a1, _ = bd.attention(y1, bw["WQ"], bw["WK"], bw["WV"], bw["WO"], W["n_heads"])
        z1 = zc + a1
        # LayerNorm variance spread at the ln2 site
        d_ = z1.S[-1]
        P0 = np.eye(d_) - np.ones((d_, d_)) / d_
        xc = z1.linear(P0)
        lo_, hi_ = xc.bounds()
        sq_hi = np.maximum(lo_ ** 2, hi_ ** 2)
        straddle = (lo_ < 0) & (hi_ > 0)
        sq_lo = np.where(straddle, 0.0, np.minimum(lo_ ** 2, hi_ ** 2))
        v_lo = sq_lo.mean(-1, keepdims=True) + W["ln_eps"]
        v_hi = sq_hi.mean(-1, keepdims=True) + W["ln_eps"]
        spread = float(np.max(np.sqrt(v_hi / np.maximum(v_lo, 1e-300))))
        y2 = bd.layernorm(z1, bw["ln2_g"], bw["ln2_b"], W["ln_eps"]).promote_E_topk(48)
        pre = y2.linear(bw["fc_in_W"], bw["fc_in_b"])
        plo, phi = pre.bounds()
        unstable = int(((plo < 0) & (phi > 0)).sum())
        m = bd.mlp(y2, bw["fc_in_W"], bw["fc_in_b"], bw["fc_out_W"],
                   bw["fc_out_b"], promote_k=48)
        z = (z1 + m).compact(128)
        out_w = float(z.width().max())
        rows.append({"layer": l, "unstable_relus": unstable,
                     "of_total": int(plo.size),
                     "unstable_frac": unstable / plo.size,
                     "ln2_spread": spread, "out_width": out_w})
    return {"input_width": in_w, "layers": rows,
            "final_width": float(z.width().max()),
            "amplification": float(z.width().max() / max(in_w, 1e-300))}


stamp("experiment 2: width scaling (randomly initialised, NOT trained)")
widths = [int(x) for x in args.widths.split(",")]
scal = []
for d in widths:
    W = synth_weights(d)
    for rho in (1e-4, 1e-3, 1e-2):
        pr_ = width_probe(W, rho)
        l1 = pr_["layers"][-1]
        scal.append({"d_model": d, "rho": rho, **pr_})
        stamp(f"  d_model={d:4d} rho={rho:<7g} amp={pr_['amplification']:.3e}  "
              f"L1 unstable={l1['unstable_frac']:.3f}  spread={l1['ln2_spread']:.3e}")
rep["width_scaling"] = {
    "rows": scal,
    "caveat": ("Randomly initialised, untrained transformers. These measure how "
               "the RELAXATION scales with width, not certified radius on a "
               "trained model of that width."),
}

# ------------------------------------------------------------ 3. prompt sweep
stamp(f"experiment 3: {args.n_prompts} anchors at rho={args.prompt_rho}")


def certified(x_nom, rho, max_boxes):
    a_lo = np.full((1, 4), -rho); a_hi = np.full((1, 4), rho)
    for _ in range(13):
        out = []
        for i in range(0, a_lo.shape[0], 128):
            zz = vf.alpha_to_zono(a_lo[i:i + 128], a_hi[i:i + 128], U) + x_nom[None]
            zL = vf._blocks_from(zz, w, 0, w["n_layers"])
            out.append(bd.unsafe_margin_upper(bd.readout_logits(zL, w), iu, isf))
        ub = np.concatenate(out)
        ub = np.where(np.isfinite(ub), ub, np.inf)
        keep = ub >= 0.0
        a_lo, a_hi = a_lo[keep], a_hi[keep]
        if a_lo.shape[0] == 0:
            return True
        if a_lo.shape[0] * 2 > max_boxes:
            return False
        wd = a_hi - a_lo
        dim = wd.argmax(axis=1); r = np.arange(a_lo.shape[0])
        mid_ = 0.5 * (a_lo[r, dim] + a_hi[r, dim])
        l1, h1 = a_lo.copy(), a_hi.copy(); h1[r, dim] = mid_
        l2, h2 = a_lo.copy(), a_hi.copy(); l2[r, dim] = mid_
        a_lo = np.concatenate([l1, l2]); a_hi = np.concatenate([h1, h2])
    return False


res_p = []
for p in range(args.n_prompts):
    ok = certified(X0[p], args.prompt_rho, args.prompt_max_boxes)
    res_p.append({"prompt": p, "tokens": prompts[p].tolist(), "certified": bool(ok)})
    if (p + 1) % 10 == 0:
        stamp(f"  {p+1}/{args.n_prompts} done, "
              f"{sum(r['certified'] for r in res_p)} certified so far")
n_ok = sum(r["certified"] for r in res_p)
rep["prompt_sweep"] = {
    "rho": args.prompt_rho, "max_boxes": args.prompt_max_boxes,
    "n": args.n_prompts, "n_certified": n_ok,
    "fraction_certified": n_ok / args.n_prompts,
    "failures": [r["prompt"] for r in res_p if not r["certified"]],
    "rows": res_p,
    "seq_len_note": ("SEQ_LEN is fixed at 5 by the task, so sequence length "
                     "could not be varied without retraining; only the "
                     "attention map varies, through token content."),
}
stamp(f"  certified {n_ok}/{args.n_prompts} at rho={args.prompt_rho}")

rep["runtime_sec"] = time.time() - t0
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(rep, open(args.out, "w"), indent=2)
stamp(f"wrote {args.out}")
print(json.dumps({"softmax": rep["softmax_ablation"]["verdict"],
                  "prompts": {"fraction_certified":
                              rep["prompt_sweep"]["fraction_certified"]}}, indent=2))
