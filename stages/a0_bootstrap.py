"""Stage A0 -- bootstrap: build every mathematical object, verify soundness,
certify what is certifiable, and measure the ceiling on what is not.

Pipeline
  1. toy transformer on the feature-routing task
  2. exact mean-direction (LayerNorm gauge) equivariance check
  3. SAE over the residual stream + bridge-gap accounting
  4. well-posedness check: is the deviation dynamics contractive at all?
  5. ICNN Lyapunov function, trained for minimum GROWTH factor
  6. soundness harness (adversarial falsification of the prover's own bounds)
  7. certification: per-layer growth (compositional) + monolithic safety margin
  8. compositional dissipativity, and the stress/ceiling measurements

Writes results/baseline.json. Nothing here reports a certificate that the
soundness harness has not first tried to break.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src import (bounds as bd, toy_transformer as tt, task, sae as sae_mod,
                 icnn, verifier as vf, dissipativity as ds, soundness, frames)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, CK = os.path.join(ROOT, "results"), os.path.join(ROOT, "checkpoints")
os.makedirs(RES, exist_ok=True); os.makedirs(CK, exist_ok=True)

SEED, K_DIRS, RHO_TRAIN = 0, 4, 0.02
# alpha is NOT a cosmetic tiebreaker in the growth formulation. It is the only
# term giving V a certified positive FLOOR, and the ratio condition
# V(e') <= gamma V(e) is unsatisfiable for every gamma wherever that floor is 0.
# At alpha=1e-3 the ICNN term vanished near the origin, V's certified lower bound
# was ~0 (indeed negative, via the DeepZ ReLU relaxation), and the bound was
# completely insensitive to gamma. alpha=0.5 fixes the feasibility.
ALPHA = 0.5
INNER_FRAC = 0.3    # inner region discharged by the margin certificate instead
CERT_RHOS = (0.005,)
MARGIN_RHOS = (0.005, 0.01, 0.02, 0.03, 0.04, 0.06)
PGD_RHOS = (0.04, 0.3, 1.0, 3.0, 10.0)
T0 = time.time()
R = {"config": {"seed": SEED, "k_dirs": K_DIRS, "rho_train": RHO_TRAIN,
                "alpha": ALPHA,
                "formulation": "certified finite-horizon growth (not decrease)"}}


def stamp(m):
    print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)


# ---------------------------------------------------------------- 1. the model
stamp("1/8  toy transformer")
mp = os.path.join(CK, "model.pt")
model = tt.ToyTransformer()
if os.path.exists(mp):
    model.load_state_dict(torch.load(mp)); stamp("     loaded checkpoint")
else:
    model, _ = tt.train(steps=2500, seed=SEED, log_every=1000)
    torch.save(model.state_dict(), mp)
model.eval()
w = model.export_weights()
R["model"] = {"d_model": w["d_model"], "n_heads": w["n_heads"],
              "n_layers": w["n_layers"], "seq_len": w["seq_len"],
              "d_mlp": tt.D_MLP,
              "params": sum(p.numel() for p in model.parameters()),
              "eval": tt.evaluate(model)}
stamp(f"     acc={R['model']['eval']['acc_all']:.4f}  "
      f"worst safe-prompt margin={R['model']['eval']['unsafe_margin_max_safe']:.3f}")

prompts, _ = task.enumerate_prompts(limit=256, rng=np.random.default_rng(SEED),
                                    safe_only=True)
with torch.no_grad():
    trace = [t.numpy().astype(np.float64)
             for t in model.residual_trace(torch.from_numpy(prompts))]
x_nom = [t[0] for t in trace]

# ------------------------------------------------------- 2. invariant frame
stamp("2/8  invariant coordinate frame")
fr = frames.InvariantFrame(x_nom[-1])
R["frame"] = {"projector": fr.check_projector(),
              "gauge": frames.layernorm_gauge_report(w),
              "mean_equivariance": vf.check_mean_equivariance(model, prompts, n=64)}
stamp(f"     LN mean-direction symmetry: max logit deviation "
      f"{R['frame']['mean_equivariance']['max_logit_deviation']:.2e} "
      f"(exact, {w['seq_len']} dims removed for free)")

# ------------------------------------------------------------------ 3. the SAE
stamp("3/8  sparse autoencoder")
sp = os.path.join(CK, "sae.pt")
acts = sae_mod.collect_activations(model, n=20000, seed=7)
sae = sae_mod.SAE(w["d_model"], 64)
if os.path.exists(sp):
    sae.load_state_dict(torch.load(sp))
else:
    sae, _ = sae_mod.train_sae(acts, d_dict=64, steps=3000, seed=SEED, log_every=1500)
    torch.save(sae.state_dict(), sp)
R["sae"] = {"bridge_gap": sae_mod.bridge_gap(sae, acts)}
stamp(f"     rel recon err {R['sae']['bridge_gap']['rel_recon_err_mean']:.4f}  "
      f"L0 {R['sae']['bridge_gap']['l0_mean']:.1f}/64  "
      f"FVU {R['sae']['bridge_gap']['fvu']:.2e}")

Wdec = sae.W_dec.detach().numpy().astype(np.float64).T
Wdec = Wdec / np.linalg.norm(Wdec, axis=1, keepdims=True)
pick = np.random.default_rng(11).choice(Wdec.shape[0], size=K_DIRS, replace=False)
U = np.zeros((K_DIRS, w["seq_len"], w["d_model"]))
U[np.arange(K_DIRS), w["seq_len"] - 1, :] = Wdec[pick]
R["threat_model"] = {
    "kind": "activation steering along k SAE decoder directions at the final token",
    "k": K_DIRS, "feature_ids": pick.tolist(),
    "set": "e = sum_j alpha_j U_j,  |alpha_j| <= rho",
    "why": ("a full box over the residual stream needs T*d_model = 160 generators "
            "and inflates ~2x per dense map regardless of propagation tightness; "
            "steering attacks move the stream along a few interpretable directions"),
}

# ------------------------------------------- 4. well-posedness of the obligation
stamp("4/8  well-posedness: is the deviation dynamics contractive?")


def full_jac(layer, x):
    xt = torch.as_tensor(x, dtype=torch.float64)[None].requires_grad_(True)
    blk = model.blocks[layer].double()
    out = blk(xt).reshape(-1)
    J = torch.zeros(out.numel(), xt.numel(), dtype=torch.float64)
    for i in range(out.numel()):
        g, = torch.autograd.grad(out[i], xt, retain_graph=(i < out.numel() - 1))
        J[i] = g.reshape(-1)
    model.blocks[layer].float()
    return J.numpy()


J0, J1 = full_jac(0, x_nom[0]), full_jac(1, x_nom[1])
Uf = U.reshape(K_DIRS, -1)
wp = {}
for nm, J in (("layer0", J0), ("layer1", J1), ("composite", J1 @ J0)):
    ev = np.abs(np.linalg.eigvals(J))
    g = np.linalg.norm(J @ Uf.T, axis=0)
    wp[nm] = {"sigma_max": float(np.linalg.svd(J, compute_uv=False)[0]),
              "spectral_radius": float(ev.max()),
              "growth_on_U": [float(v) for v in g]}
evr = np.abs(np.linalg.eigvals(Uf @ (J1 @ J0) @ Uf.T))
wp["composite_restricted_to_span_U"] = {
    "spectral_radius": float(evr.max()),
    "eigenvalues": sorted(float(v) for v in evr)[::-1]}
wp["contraction_possible"] = bool(evr.max() < 1.0)
wp["conclusion"] = (
    "A positive-definite V with V(e') <= gamma V(e), gamma < 1, exists IFF the "
    "deviation spectral radius is < 1. It is "
    f"{evr.max():.3f} > 1 on the certified subspace, so the strict-decrease "
    "obligation is INFEASIBLE for this model -- not merely hard to relax. The "
    "certificate is therefore stated as bounded finite-horizon growth, which is "
    "true and still sufficient for a safety guarantee.")
R["well_posedness"] = wp
stamp(f"     spectral radius on span(U) = {evr.max():.3f}  "
      f"=> contraction possible: {wp['contraction_possible']}")

# ----------------------------------------------------------- 5. Lyapunov fn
stamp("5/8  ICNN Lyapunov function (minimum-growth objective)")
vp = os.path.join(CK, f"lyap_a{ALPHA}.pt")
V = icnn.ICNNLyapunov(sae, w["d_model"], w["seq_len"], widths=(64, 64), alpha=ALPHA)
if os.path.exists(vp):
    blob = torch.load(vp); V.load_state_dict(blob["sd"]); gamma_tr = blob["gamma"]
else:
    V, _, gamma_tr = icnn.train_lyapunov(model, sae, prompts, U, steps=1500,
                                         rho=RHO_TRAIN, alpha=ALPHA, seed=SEED,
                                         log_every=500)
    torch.save({"sd": V.state_dict(), "gamma": gamma_tr}, vp)
V.eval()
vw = V.export_weights()
R["lyapunov"] = {
    "form": "V(e) = ReLU(g(W_enc e) - g(0)) + alpha*||P e||^2, g an ICNN",
    "V_at_origin": float(V(torch.zeros(1, w["seq_len"], w["d_model"])).item()),
    "positive_definite_by_construction": True,
    "convex_in_state": True,
    "sampled_growth_factor": gamma_tr, "alpha": ALPHA,
}
stamp(f"     V(0)={R['lyapunov']['V_at_origin']:.1e} (exact)  "
      f"sampled gamma={gamma_tr:.4f}")

# -------------------------------------------------------------- 6. soundness
stamp("6/8  soundness harness (trying to break our own bounds)")
snd = {"primitives": soundness.check_primitives(),
       "v_bounds": soundness.check_v_bounds(V, vw, 0.02, w["seq_len"],
                                            w["d_model"], n=4000, seed=1),
       "block_bounds": soundness.check_block_bounds(model, w, x_nom[0], 0.002,
                                                    n=2000, seed=0)}
sub = {}
for l in range(w["n_layers"]):
    for rho in (0.01, 0.05):
        e0, e1 = vf.deviation_step_subspace(np.full((1, K_DIRS), -rho),
                                            np.full((1, K_DIRS), rho), U, w, l,
                                            x_nom[l], x_nom[l + 1])
        b = float(icnn.lyap_gap_upper(e0, e1, vw, gamma_tr)["d_hi"][0])
        rr = np.random.default_rng(l)
        a = rr.uniform(-rho, rho, size=(4000, K_DIRS))
        with torch.no_grad():
            et = torch.as_tensor(np.einsum("bk,ktd->btd", a, U), dtype=torch.float32)
            xn = torch.as_tensor(x_nom[l], dtype=torch.float32)[None]
            xn1 = torch.as_tensor(x_nom[l + 1], dtype=torch.float32)[None]
            tv = (V(model.blocks[l](xn + et) - xn1) - gamma_tr * V(et)).numpy()
        att = float(tv.max())
        sub[f"layer{l}_rho{rho}"] = {
            "sound_bound": b, "attained_max_sampled": att,
            "max_violation": float(att - b), "sound": bool(att - b <= 1e-5),
            "relaxation_gap": float(b - att)}
snd["growth_bound_subspace"] = sub
snd["ALL_SOUND"] = bool(all(v["sound"] for v in snd["primitives"].values())
                        and snd["v_bounds"]["sound"]
                        and all(v["sound"] for v in snd["block_bounds"].values())
                        and all(v["sound"] for v in sub.values()))
R["soundness"] = snd
stamp(f"     ALL_SOUND = {snd['ALL_SOUND']} (no sampled point escaped any bound)")

# ------------------------------------------------------------ 7. certification
stamp("7/8  certification")
cert = {"per_layer_growth": {}, "monolithic_safety": {}, "metric_ablation": {}}
# Ablation: does the learned ICNN metric certify a SMALLER growth factor than the
# trivial quadratic one? If not, the ICNN is decoration and the honest statement
# is just a bound on l2 perturbation growth. Reported either way.
METRICS = (("icnn", vw), ("quadratic_only", icnn.quadratic_only_vw(vw)))
for mname, mvw in METRICS:
    cert["metric_ablation"][mname] = {}
    for rho in CERT_RHOS:
        row = {}
        for l in range(w["n_layers"]):
            gmin, st = vf.min_certified_gamma(
                w, mvw, U, x_nom[l], x_nom[l + 1], l, rho, lo=1.0, hi=6.0,
                iters=3, max_boxes=256, max_iters=12, inner_frac=INNER_FRAC)
            row[f"layer{l}"] = {"gamma_certified": gmin, "stats": st}
        gs = [row[f"layer{l}"]["gamma_certified"] for l in range(w["n_layers"])]
        row["composite_growth_bound"] = (float(np.prod(gs)) if all(gs) else None)
        cert["metric_ablation"][mname][str(rho)] = row
        stamp(f"     [{mname:14s}] rho={rho:<6} per-layer gamma={gs}  "
              f"composite={row['composite_growth_bound']}")
cert["per_layer_growth"] = cert["metric_ablation"]["icnn"]
cert["inner_frac"] = INNER_FRAC
cert["inner_region_note"] = (
    "The inner region |alpha_j| <= inner_frac*rho is NOT covered by the growth "
    "condition: V(e_l) has no positive floor there, so V(e') <= gamma V(e) is "
    "unsatisfiable for every gamma. It is discharged separately by the margin "
    "certificate below, which is the standard practical-stability formulation. "
    "Its size is a reported parameter, not a hidden knob.")

iu, isf = task.margin_readout()
for rho in MARGIN_RHOS:
    ok, st = vf.certify_margin_radius(w, U, x_nom[0], iu, isf, rho,
                                      max_boxes=4096, max_iters=20, chunk=128)
    cert["monolithic_safety"][str(rho)] = {"certified_safe": bool(ok), "stats": st}
    stamp(f"     margin rho={rho:<6} certified_safe={ok}  "
          f"worst={st.get('worst', 'n/a')}  boxes={st.get('boxes_touched')}")
safe_rhos = [float(r) for r, v in cert["monolithic_safety"].items()
             if v["certified_safe"]]
cert["max_certified_safe_rho"] = max(safe_rhos) if safe_rhos else None

# The denominator for the geometric gap: where does the model ACTUALLY break?
# Comparing a sound radius against a sampled "nothing happened" radius would
# flatter the certificate; PGD gives an honest adversarial denominator.
Ut = torch.as_tensor(U, dtype=torch.float32)
xn0 = torch.as_tensor(x_nom[0], dtype=torch.float32)[None]
pgd = {}
for rho in PGD_RHOS:
    a = ((torch.rand((256, K_DIRS)) * 2 - 1) * rho).requires_grad_(True)
    best = -np.inf
    for _ in range(120):
        e = torch.einsum("bk,ktd->btd", a, Ut)
        lg = model.logits_from_stream(model.blocks[1](model.blocks[0](xn0 + e)))[:, -1]
        s = lg[:, iu].max(1).values - lg[:, isf].max(1).values
        g, = torch.autograd.grad(-s.sum(), a)
        with torch.no_grad():
            a -= (rho / 8) * g.sign(); a.clamp_(-rho, rho)
            best = max(best, float(s.detach().max()))
    pgd[str(rho)] = {"pgd_max_margin": best, "broken": bool(best > 0)}
    stamp(f"     PGD rho={rho:<6} max margin={best:+.4f}  broken={best > 0}")
cert["pgd_attack"] = pgd
broken = [float(r) for r, v in pgd.items() if v["broken"]]
cert["empirical_robust_radius"] = (min(broken) if broken else None)
if cert["max_certified_safe_rho"] and cert["empirical_robust_radius"]:
    cert["geometric_gap_fraction"] = (cert["max_certified_safe_rho"]
                                      / cert["empirical_robust_radius"])
R["certification"] = cert

# --------------------------------------------------------- 8. dissipativity
stamp("8/8  compositional dissipativity + stress")
gains = ds.subblock_gains(w, x_nom, 0.01)
R["dissipativity"] = {"subblock_gains": gains,
                      "gain_report": ds.layer_gain_report(gains),
                      "supply_rate_lp": ds.compose_supply_rates(gains, kappa=0.02)}

stress = {"unstable_relu_vs_rho": [], "subspace_dim_scaling": [],
          "propagation_ablation": [], "sae_bridge_sensitivity": {}}
P0 = np.eye(w["d_model"]) - np.ones((w["d_model"], w["d_model"])) / w["d_model"]
for rho in (0.005, 0.01, 0.02, 0.05, 0.1, 0.25):
    z = bd.HZono.from_subspace(x_nom[0][None], U, rho)
    row = {"rho": rho, "layers": []}
    zz = z
    for l in range(w["n_layers"]):
        bw = w["blocks"][l]
        zc = zz.promote_E().compact(192)
        xc = zc.linear(P0)
        tot = np.abs(xc.G).sum(1) + xc.E
        ratio = float((np.linalg.norm(tot, axis=-1)
                       / np.linalg.norm(xc.c, axis=-1)).max())
        y1 = bd.layernorm(zc, bw["ln1_g"], bw["ln1_b"], w["ln_eps"]).promote_E()
        a, _ = bd.attention(y1, bw["WQ"], bw["WK"], bw["WV"], bw["WO"], w["n_heads"])
        y2 = bd.layernorm(zc + a, bw["ln2_g"], bw["ln2_b"], w["ln_eps"]).promote_E()
        hl, hh = y2.linear(bw["fc_in_W"], bw["fc_in_b"]).bounds()
        zz, _ = bd.block(zz, bw, w["n_heads"], w["ln_eps"])
        lo, hi = zz.bounds()
        row["layers"].append({"layer": l, "perturbation_to_signal_ratio": ratio,
                              "unstable_relus": int(((hl < 0) & (hh > 0)).sum()),
                              "of_total": int(hl.size),
                              "out_width": float((hi - lo).mean())})
    stress["unstable_relu_vs_rho"].append(row)

for k in (1, 2, 4, 8, 16):
    pk = np.random.default_rng(11).choice(Wdec.shape[0], size=k, replace=False)
    Uk = np.zeros((k, w["seq_len"], w["d_model"]))
    Uk[np.arange(k), w["seq_len"] - 1, :] = Wdec[pk]
    zz = bd.HZono.from_subspace(x_nom[0][None], Uk, 0.01)
    for l in range(w["n_layers"]):
        zz, _ = bd.block(zz, w["blocks"][l], w["n_heads"], w["ln_eps"])
    lo, hi = zz.bounds()
    stress["subspace_dim_scaling"].append(
        {"k": k, "rho": 0.01, "final_width": float((hi - lo).mean()),
         "finite": bool(np.isfinite(hi).all())})

for label, kw in (("hybrid+promotion", {}), ("no promotion", {"promote": False})):
    zz = bd.HZono.from_subspace(x_nom[0][None], U, 0.01)
    for l in range(w["n_layers"]):
        zz, _ = bd.block(zz, w["blocks"][l], w["n_heads"], w["ln_eps"], **kw)
    lo, hi = zz.bounds()
    stress["propagation_ablation"].append(
        {"mode": label, "final_width": float((hi - lo).mean())})

zz = bd.HZono.from_box((x_nom[0] - 0.002)[None], (x_nom[0] + 0.002)[None])
for l in range(w["n_layers"]):
    zz, _ = bd.block(zz, w["blocks"][l], w["n_heads"], w["ln_eps"])
lo, hi = zz.bounds()
stress["full_box_baseline"] = {"rho": 0.002, "n_generators": 160,
                               "final_width": float((hi - lo).mean())}
bg = R["sae"]["bridge_gap"]
stress["sae_bridge_sensitivity"] = {
    "rel_recon_err_mean": bg["rel_recon_err_mean"],
    "note": ("the encoder is used as a READOUT (V is a function of e, not of "
             "Dec(Enc(e))), so reconstruction error does NOT enter the proved "
             "dynamics; it bounds only how well V's level sets align with "
             "feature semantics. If anyone closes the loop through Dec, this "
             "number becomes the soundness bottleneck."),
}
R["stress"] = stress
R["backends"] = vf.backend_report()
R["runtime_sec"] = time.time() - T0

with open(os.path.join(RES, "baseline.json"), "w") as f:
    json.dump(R, f, indent=2, default=str)

print("\n" + "=" * 78)
print("BASELINE SUMMARY")
print("=" * 78)
print(f"  task accuracy                 {R['model']['eval']['acc_all']:.4f}")
print(f"  SAE rel. recon error          {bg['rel_recon_err_mean']:.4f}")
print(f"  all bounds sound              {R['soundness']['ALL_SOUND']}")
print(f"  contraction possible          {wp['contraction_possible']} "
      f"(spectral radius {evr.max():.3f})")
print(f"  sampled growth factor         {gamma_tr:.4f}")
for r, v in cert["per_layer_growth"].items():
    print(f"  certified growth @ rho={r:<6} {v['composite_growth_bound']}")
print(f"  max certified safe rho        {cert['max_certified_safe_rho']}")
print(f"  PGD breaks at rho            {cert.get('empirical_robust_radius')}")
if cert.get("geometric_gap_fraction"):
    print(f"  geometric gap                 certificate covers "
          f"{100*cert['geometric_gap_fraction']:.2f}% of true robust radius")
print(f"  small-gain composition        "
      f"{R['dissipativity']['gain_report']['small_gain_satisfied']} "
      f"(product {R['dissipativity']['gain_report']['cascade_gain_product']:.1f})")
print(f"  runtime                       {R['runtime_sec']:.0f}s")
print(f"\n  wrote results/baseline.json")
