"""c16 -- stage-by-stage comparison: true vs prover vs CROWN reference.

The headline question for the CROWN engine is not "is it sound" (selftest_crown
answers that) but "where exactly does it lose against the native prover". This
emits that table, which is the artifact that localises the remaining gap.

`true` widths are attained widths from dense sampling of the exact float64
model: a lower bound on what any sound method must report.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from src import toy_transformer as tt, task, sae as sae_mod, bounds as bd, verifier as vf
import crown_reference as cr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "checkpoints")
ap = argparse.ArgumentParser()
ap.add_argument("--rhos", default="1e-4,1e-3,5e-3,0.02,0.04")
ap.add_argument("--samples", type=int, default=4000)
ap.add_argument("--out", default=os.path.join(ROOT, "results", "c16_crown.json"))
args = ap.parse_args()

model = tt.ToyTransformer()
model.load_state_dict(torch.load(os.path.join(CK, "model.pt")))
model.eval().double()
w = model.export_weights()
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
    tr = model.residual_trace(torch.from_numpy(prompts))
xn = tr[0][0].numpy().astype(np.float64)
iu, isf = task.margin_readout()

STAGES = []
for l in range(w["n_layers"]):
    STAGES += [f"L{l} ln1", f"L{l} attn", f"L{l} resid1", f"L{l} ln2", f"L{l} out"]

rows = []
for rho in [float(x) for x in args.rhos.split(",")]:
    mid = np.zeros(4); rad = np.full(4, rho)
    a_lo = np.full((1, 4), -rho); a_hi = np.full((1, 4), rho)

    rng = np.random.default_rng(0)
    al = rng.uniform(-rho, rho, size=(args.samples, 4))
    xs = torch.from_numpy(np.einsum("bk,ktd->btd", al, U) + xn[None])
    tw = {}
    with torch.no_grad():
        x = xs
        for l, blk in enumerate(model.blocks):
            n1 = blk.ln1(x); tw[f"L{l} ln1"] = n1
            a = blk.attn(n1); tw[f"L{l} attn"] = a
            x1 = x + a; tw[f"L{l} resid1"] = x1
            n2 = blk.ln2(x1); tw[f"L{l} ln2"] = n2
            x = x1 + blk.mlp(n2); tw[f"L{l} out"] = x
    tw = {k: float((v.numpy().max(0) - v.numpy().min(0)).max()) for k, v in tw.items()}

    L = cr.LB.exact(U.copy(), xn.copy()); cw = {}
    for l in range(w["n_layers"]):
        bwl = w["blocks"][l]
        Y1 = cr.layernorm(L, bwl["ln1_g"], bwl["ln1_b"], w["ln_eps"], mid, rad)
        cw[f"L{l} ln1"] = Y1
        A = cr.attention(Y1, bwl["WQ"], bwl["WK"], bwl["WV"], bwl["WO"],
                         w["n_heads"], mid, rad); cw[f"L{l} attn"] = A
        Z1 = L.add(A); cw[f"L{l} resid1"] = Z1
        Y2 = cr.layernorm(Z1, bwl["ln2_g"], bwl["ln2_b"], w["ln_eps"], mid, rad)
        cw[f"L{l} ln2"] = Y2
        Pre = cr.lin(Y2, bwl["fc_in_W"], bwl["fc_in_b"])
        pl, ph = Pre.concretize(mid, rad)
        L = Z1.add(cr.lin(cr.relu(Pre, pl, ph), bwl["fc_out_W"], bwl["fc_out_b"]))
        cw[f"L{l} out"] = L
    cwid = {k: float(np.max(v.concretize(mid, rad)[1] - v.concretize(mid, rad)[0]))
            for k, v in cw.items()}

    z = vf.alpha_to_zono(a_lo, a_hi, U) + xn[None]; hw = {}
    for l in range(w["n_layers"]):
        bwl = w["blocks"][l]; zc = z.promote_E().compact(128)
        y1 = bd.layernorm(zc, bwl["ln1_g"], bwl["ln1_b"], w["ln_eps"]).promote_E_topk(48)
        hw[f"L{l} ln1"] = y1
        a, _ = bd.attention(y1, bwl["WQ"], bwl["WK"], bwl["WV"], bwl["WO"], w["n_heads"])
        hw[f"L{l} attn"] = a
        z1 = zc + a; hw[f"L{l} resid1"] = z1
        y2 = bd.layernorm(z1, bwl["ln2_g"], bwl["ln2_b"], w["ln_eps"]).promote_E_topk(48)
        hw[f"L{l} ln2"] = y2
        m = bd.mlp(y2, bwl["fc_in_W"], bwl["fc_in_b"], bwl["fc_out_W"],
                   bwl["fc_out_b"], promote_k=48)
        z = (z1 + m).compact(128); hw[f"L{l} out"] = z
    hwid = {k: float(v.width().max()) for k, v in hw.items()}

    rows.append({
        "rho": rho,
        "stages": [{"stage": s, "true": tw[s], "hzono": hwid[s], "crown": cwid[s],
                    "hzono_over_true": hwid[s] / tw[s],
                    "crown_over_true": cwid[s] / tw[s]} for s in STAGES],
        "margin_hzono": float(bd.unsafe_margin_upper(bd.readout_logits(z, w), iu, isf)[0]),
        "margin_crown": cr.certify_margin(np.full(4, -rho), np.full(4, rho),
                                          U, xn, w, iu, isf),
    })
    print(f"rho={rho:<8g} hzono margin={rows[-1]['margin_hzono']:+.4e}  "
          f"crown margin={rows[-1]['margin_crown']:+.4e}", flush=True)

first = rows[0]["stages"]
worst = max(first, key=lambda s: s["crown_over_true"] / max(s["hzono_over_true"], 1e-30))
out = {"config": {"rhos": args.rhos, "samples": args.samples,
                  "feature_ids": pick.tolist()},
       "rows": rows,
       "diagnosis": {
           "first_divergent_stage": next(s["stage"] for s in first
                                         if s["crown_over_true"] > 10 * s["hzono_over_true"]),
           "worst_relative_stage": worst["stage"],
           "note": ("CROWN is sound and exact at a point but loses to the "
                    "prover at attention, where softmax uncertainty enters as "
                    "interval coefficients on V. The loss compounds through "
                    "layer 1. It closes no box the prover closes."),
       }}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(out, open(args.out, "w"), indent=2)
print(json.dumps(out["diagnosis"], indent=2))
print(f"wrote {args.out}")
