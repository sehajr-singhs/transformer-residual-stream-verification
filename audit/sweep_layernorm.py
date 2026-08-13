"""c14 part 2 -- map the LayerNorm variance-bracket cliff.

`src/bounds.py:layernorm` brackets the variance two incomparable ways and keeps
the better of each end:

  coordinate-wise   var_lo_cw = mean_d(min u_d^2) + eps   -- collapses to eps as
                    soon as every coordinate interval straddles zero, giving
                    1/sqrt(var_lo) ~ 1/sqrt(1e-5) ~ 316
  norm-based        var_lo_nm = max(||c|| - R, 0)^2 / d + eps  -- uses the
                    correlation the zonotope carries, and dies when the
                    perturbation radius R reaches the centre norm ||c||

The cliff is the radius at which the norm bracket stops being the active one and
the coordinate-wise floor takes over. Past it, s_hi/s_lo jumps by orders of
magnitude, that spread lands in E, and every downstream dense map multiplies it
by an l1 row norm.

This sweep records, on a fine rho grid and per LayerNorm site:
  - which bracket is active at each end
  - the bracket ratio s_hi/s_lo, which is what actually feeds E
  - unstable ReLU counts per layer
  - the resulting end-to-end margin bound

so the mechanism behind the 250x conservativeness gap is a measured curve rather
than an assertion.

NOTE ON SCOPE. The O(L) property claimed in ARCHITECTURE.md is a statement about
COST (each layer's obligation is discharged independently over a shared box), not
about tightness, and the compositional growth route certified no gamma at any
radius in the baseline. So there is no radius at which an O(L) tightness claim
"collapses" -- it was never established. What this sweep localises is the
bracket cliff that drives the monolithic bound, which is the certificate that
actually closes.
"""
import sys, os, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import toy_transformer as tt, task, sae as sae_mod, bounds as bd, verifier as vf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "checkpoints")
_ABS = np.abs

ap = argparse.ArgumentParser()
ap.add_argument("--n-rho", type=int, default=40)
ap.add_argument("--rho-lo", type=float, default=1e-4)
ap.add_argument("--rho-hi", type=float, default=0.25)
ap.add_argument("--out", default=os.path.join(ROOT, "results", "c14_ln_sweep.json"))
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
Wdec = sae.W_dec.detach().numpy().astype(np.float64).T
Wdec = Wdec / np.linalg.norm(Wdec, axis=1, keepdims=True)
pick = np.random.default_rng(11).choice(Wdec.shape[0], size=4, replace=False)
U = np.zeros((4, w["seq_len"], w["d_model"]))
U[np.arange(4), w["seq_len"] - 1, :] = Wdec[pick]
prompts, _ = task.enumerate_prompts(limit=256, rng=np.random.default_rng(0),
                                    safe_only=True)
with torch.no_grad():
    tr = m64.residual_trace(torch.from_numpy(prompts))
x_nom_0 = tr[0][0].numpy().astype(np.float64)
iu, isf = task.margin_readout()


def bracket_report(z, eps):
    """Recompute both variance brackets for a form, exactly as bounds.py does."""
    d = z.S[-1]
    P0 = np.eye(d) - np.ones((d, d)) / d
    xc = z.linear(P0)
    lo, hi = xc.bounds()
    sq_hi = np.maximum(lo ** 2, hi ** 2)
    straddle = (lo < 0) & (hi > 0)
    sq_lo = np.where(straddle, 0.0, np.minimum(lo ** 2, hi ** 2))
    var_lo_cw = sq_lo.mean(axis=-1, keepdims=True) + eps
    var_hi_cw = sq_hi.mean(axis=-1, keepdims=True) + eps
    cn = np.linalg.norm(xc.c, axis=-1, keepdims=True)
    tot_r = _ABS(xc.G).sum(axis=1) + xc.E
    R2 = np.minimum(np.linalg.norm(tot_r, axis=-1, keepdims=True),
                    np.linalg.norm(xc.G, axis=-1).sum(axis=1)[..., None]
                    + np.linalg.norm(xc.E, axis=-1, keepdims=True))
    var_lo_nm = np.maximum(cn - R2, 0.0) ** 2 / d + eps
    var_hi_nm = (cn + R2) ** 2 / d + eps
    var_lo = np.maximum(var_lo_cw, var_lo_nm)
    var_hi = np.minimum(var_hi_cw, var_hi_nm)
    s_hi = 1.0 / np.sqrt(var_lo)
    s_lo = 1.0 / np.sqrt(var_hi)
    return {
        "straddle_frac": float(straddle.mean()),
        "norm_bracket_active_lo": float((var_lo_nm >= var_lo_cw).mean()),
        "cn_over_R": float(np.median(cn / np.maximum(R2, 1e-300))),
        "var_lo": float(np.median(var_lo)),
        "var_hi": float(np.median(var_hi)),
        "spread_ratio": float(np.median(s_hi / np.maximum(s_lo, 1e-300))),
        "spread_ratio_max": float(np.max(s_hi / np.maximum(s_lo, 1e-300))),
        "var_lo_at_eps_floor": float(np.mean(var_lo <= 1.0000001e-5)),
        "norm_bracket_dead": bool(np.all(var_lo_nm <= var_lo_cw)),
    }


def probe(rho):
    a_lo = np.full((1, 4), -rho); a_hi = np.full((1, 4), rho)
    z0 = vf.alpha_to_zono(a_lo, a_hi, U) + x_nom_0[None]
    row = {"rho": rho, "sites": []}
    z = z0
    for l in range(w["n_layers"]):
        bw = w["blocks"][l]
        zc = z.promote_E().compact(128)
        row["sites"].append({"layer": l, "site": "ln1", **bracket_report(zc, w["ln_eps"])})
        y1 = bd.layernorm(zc, bw["ln1_g"], bw["ln1_b"], w["ln_eps"]).promote_E_topk(48)
        a1, _ = bd.attention(y1, bw["WQ"], bw["WK"], bw["WV"], bw["WO"], w["n_heads"])
        z1 = zc + a1
        row["sites"].append({"layer": l, "site": "ln2", **bracket_report(z1, w["ln_eps"])})
        y2 = bd.layernorm(z1, bw["ln2_g"], bw["ln2_b"], w["ln_eps"]).promote_E_topk(48)
        pre = y2.linear(bw["fc_in_W"], bw["fc_in_b"])
        plo, phi = pre.bounds()
        unstable = int(((plo < 0) & (phi > 0)).sum())
        m = bd.mlp(y2, bw["fc_in_W"], bw["fc_in_b"], bw["fc_out_W"], bw["fc_out_b"],
                   promote_k=48)
        z = (z1 + m).compact(128)
        lo_, hi_ = z.bounds()
        row["sites"][-1]["unstable_relus"] = unstable
        row["sites"][-1]["of_total"] = int(plo.size)
        row["sites"][-1]["out_width"] = float((hi_ - lo_).mean())
    zlog = bd.readout_logits(z, w)
    row["margin_bound"] = float(bd.unsafe_margin_upper(zlog, iu, isf)[0])
    row["final_stream_width"] = float(z.width().mean())
    return row


rhos = np.geomspace(args.rho_lo, args.rho_hi, args.n_rho)
rows = []
for r in rhos:
    rows.append(probe(float(r)))
    s = rows[-1]
    ln2_l1 = [x for x in s["sites"] if x["layer"] == 1 and x["site"] == "ln2"][0]
    stamp(f"rho={r:.5f}  margin={s['margin_bound']:+.4e}  "
          f"L1.ln2 unstable={ln2_l1['unstable_relus']:3d}/{ln2_l1['of_total']}  "
          f"spread_med={ln2_l1['spread_ratio']:.4f}  "
          f"spread_max={ln2_l1['spread_ratio_max']:.4e}  "
          f"eps_floor={ln2_l1['var_lo_at_eps_floor']:.2f}")

# locate the cliff: first radius where the norm bracket stops carrying the floor
cliff = {}
for site in ("ln1", "ln2"):
    for l in range(w["n_layers"]):
        key = f"layer{l}_{site}"
        dead_at = None
        for row in rows:
            s = [x for x in row["sites"] if x["layer"] == l and x["site"] == site][0]
            if s["norm_bracket_dead"] and dead_at is None:
                dead_at = row["rho"]
        cliff[key] = dead_at
first_pos = next((r["rho"] for r in rows if r["margin_bound"] >= 0), None)

out = {"config": {"n_rho": args.n_rho, "rho_lo": args.rho_lo, "rho_hi": args.rho_hi,
                  "feature_ids": pick.tolist()},
       "rows": rows,
       "cliff": {"norm_bracket_dies_at_rho": cliff,
                 "margin_bound_first_nonnegative_at_rho": first_pos},
       "scope_note": ("O(L) in ARCHITECTURE.md is a cost claim, not a tightness "
                      "claim, and the compositional growth route certified no "
                      "gamma at any radius; what is localised here is the "
                      "bracket cliff driving the monolithic bound."),
       "runtime_sec": time.time() - t0}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(out, open(args.out, "w"), indent=2)
stamp(f"wrote {args.out}")
print(json.dumps(out["cliff"], indent=2))
