"""Diagnose the V-degeneracy and find a workable (alpha, rho, budget) regime.

Symptom: the sound bound on V(e') - gamma V(e) was insensitive to gamma, which
can only happen if the lower bound on V(e) is ~0. With alpha tiny, ReLU(g - g0)
vanishes on a neighbourhood of the origin and the quadratic floor is negligible,
so V(e) has no certified positive lower bound and the RATIO condition
V(e') <= gamma V(e) is unsatisfiable for every gamma. alpha is not a cosmetic
tiebreaker for this formulation -- it is what makes the obligation feasible.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import toy_transformer as tt, task, sae as sae_mod, icnn, verifier as vf, bounds as bd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "checkpoints")
model = tt.ToyTransformer(); model.load_state_dict(torch.load(os.path.join(CK, "model.pt")))
model.eval(); w = model.export_weights()
sae = sae_mod.SAE(w["d_model"], 64); sae.load_state_dict(torch.load(os.path.join(CK, "sae.pt")))
prompts, _ = task.enumerate_prompts(limit=256, rng=np.random.default_rng(0), safe_only=True)
with torch.no_grad():
    tr = [t.numpy().astype(np.float64) for t in model.residual_trace(torch.from_numpy(prompts))]
x_nom = [t[0] for t in tr]
Wdec = sae.W_dec.detach().numpy().astype(np.float64).T
Wdec = Wdec / np.linalg.norm(Wdec, axis=1, keepdims=True)
pick = np.random.default_rng(11).choice(Wdec.shape[0], size=4, replace=False)
U = np.zeros((4, w["seq_len"], w["d_model"]))
U[np.arange(4), w["seq_len"] - 1, :] = Wdec[pick]

# --- confirm the degeneracy on the existing V
blob = torch.load(os.path.join(CK, "lyap.pt"))
V0 = icnn.ICNNLyapunov(sae, w["d_model"], w["seq_len"], widths=(64, 64), alpha=1e-3)
V0.load_state_dict(blob["sd"]); V0.eval(); vw0 = V0.export_weights()
e0, e1 = vf.deviation_step_subspace(np.full((1, 4), -0.05), np.full((1, 4), 0.05),
                                    U, w, 0, x_nom[0], x_nom[1])
d = icnn.lyap_gap_upper(e0, e1, vw0, 1.4)
print("alpha=1e-3 (current V):")
print(f"  V(e_l)  in [{d['v0_lo'][0]:.4e}, {d['v0_hi'][0]:.4e}]   <-- lower bound ~0")
print(f"  V(e_l+1) in [{d['v1_lo'][0]:.4e}, {d['v1_hi'][0]:.4e}]")
print(f"  => bound on V' - gamma V = {d['d_hi'][0]:.4e}; gamma cannot help\n")

# --- retrain with a meaningful quadratic floor
for ALPHA in (0.5,):
    print(f"retraining V with alpha={ALPHA} ...")
    V, hist, gam = icnn.train_lyapunov(model, sae, prompts, U, steps=1200,
                                       rho=0.02, alpha=ALPHA, seed=0,
                                       log_every=400, widths=(64, 64))
    V.eval(); vw = V.export_weights()
    torch.save({"sd": V.state_dict(), "gamma": gam, "alpha": ALPHA},
               os.path.join(CK, f"lyap_a{ALPHA}.pt"))
    print(f"  sampled gamma = {gam:.4f}")
    for rho in (0.002, 0.005, 0.01, 0.02):
        e0, e1 = vf.deviation_step_subspace(np.full((1, 4), -rho), np.full((1, 4), rho),
                                            U, w, 0, x_nom[0], x_nom[1])
        d = icnn.lyap_gap_upper(e0, e1, vw, gam)
        rr = np.random.default_rng(0)
        a = rr.uniform(-rho, rho, size=(3000, 4))
        with torch.no_grad():
            et = torch.as_tensor(np.einsum("bk,ktd->btd", a, U), dtype=torch.float32)
            xn = torch.as_tensor(x_nom[0], dtype=torch.float32)[None]
            xn1 = torch.as_tensor(x_nom[1], dtype=torch.float32)[None]
            tv = (V(model.blocks[0](xn + et) - xn1) - gam * V(et)).numpy()
        print(f"    rho={rho:<6} V0=[{d['v0_lo'][0]:.3e},{d['v0_hi'][0]:.3e}] "
              f"V1_hi={d['v1_hi'][0]:.3e}  bound={d['d_hi'][0]:+.3e}  "
              f"sampled_max={tv.max():+.3e}")

    print("\n  BaB probe (budget 256 boxes, 12 iters):")
    for rho in (0.002, 0.005, 0.01):
        for gtry in (gam, gam * 1.5, 3.0):
            t = time.time()
            ok, st = vf.certify_growth(w, vw, U, x_nom[0], x_nom[1], 0, rho, gtry,
                                       max_boxes=256, max_iters=12, chunk=32)
            print(f"    rho={rho:<6} gamma={gtry:<6.3f} proved={ok}  "
                  f"{time.time()-t:5.1f}s  worst={st.get('worst', 0):+.3e} "
                  f"reason={st.get('reason', '-')}")
            if ok:
                break
