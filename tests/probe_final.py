"""Decisive probe: does the corrected formulation certify anything, and does the
ICNN beat a trivial quadratic metric?

Changes under test
  1. clamped floor on V(e_l)  -- V >= quad >= 0 beats the DeepZ concretization
  2. a MEANINGFUL inner exclusion. With inner_frac tiny, boxes adjacent to the
     origin have V(e_l) floor ~ 0 while V(e_{l+1}) carries an O(width^2)
     propagation error, so the ratio condition can never close there. Excluding
     |alpha_j| <= inner_frac*rho gives a genuine positive floor
     V >= alpha * (inner_frac*rho)^2 * sigma_min(U)^2, and the inner region is
     discharged separately by the margin certificate (practical stability).
  3. ICNN vs quadratic-only V.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import toy_transformer as tt, task, sae as sae_mod, icnn, verifier as vf

OUT = open(os.path.join(os.path.dirname(__file__), "probe_final.txt"), "w")


def say(s):
    print(s, flush=True); OUT.write(s + "\n"); OUT.flush()


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

blob = torch.load(os.path.join(CK, "lyap_a0.5.pt"))
V = icnn.ICNNLyapunov(sae, w["d_model"], w["seq_len"], widths=(64, 64), alpha=0.5)
V.load_state_dict(blob["sd"]); V.eval()
vw_icnn = V.export_weights()
vw_quad = icnn.quadratic_only_vw(vw_icnn)
GAM = blob["gamma"]
say(f"sampled gamma (ICNN V, alpha=0.5): {GAM:.4f}")

say("\n=== effect of the clamped floor (rho=0.005, layer 0) ===")
e0, e1 = vf.deviation_step_subspace(np.full((1, 4), -0.005), np.full((1, 4), 0.005),
                                    U, w, 0, x_nom[0], x_nom[1])
for nm, vw in (("ICNN", vw_icnn), ("quadratic-only", vw_quad)):
    d = icnn.lyap_gap_upper(e0, e1, vw, 1.5)
    say(f"  {nm:15s} V0=[{d['v0_lo'][0]:+.3e},{d['v0_hi'][0]:+.3e}] "
        f"floor={d['v0_floor'][0]:.3e}  V1_hi={d['v1_hi'][0]:.3e}")
    say(f"  {'':15s} correlated={d['d_hi_correlated'][0]:+.3e}  "
        f"clamped={d['d_hi_clamped'][0]:+.3e}  used={d['d_hi'][0]:+.3e}")

say("\n=== BaB with a meaningful inner exclusion ===")
say(f"{'metric':<16}{'rho':>7}{'inner':>7}{'gamma':>8}{'proved':>8}{'worst':>12}"
    f"{'boxes':>8}{'sec':>7}")
results = {}
for nm, vw in (("quadratic-only", vw_quad), ("ICNN", vw_icnn)):
    for rho in (0.005, 0.02):
        for inner in (0.3, 0.5):
            for gtry in (1.5, 2.5, 4.0):
                t = time.time()
                ok, st = vf.certify_growth(w, vw, U, x_nom[0], x_nom[1], 0, rho, gtry,
                                           max_boxes=512, max_iters=14, chunk=32,
                                           inner_frac=inner)
                say(f"{nm:<16}{rho:>7}{inner:>7}{gtry:>8.2f}{str(ok):>8}"
                    f"{st.get('worst', 0):>12.3e}{st.get('boxes_touched', 0):>8}"
                    f"{time.time()-t:>7.1f}")
                results[(nm, rho, inner, gtry)] = ok
                if ok:
                    break

say("\n=== monolithic safety margin (the safety-relevant obligation) ===")
iu, isf = task.margin_readout()
for rho in (0.002, 0.005, 0.01, 0.02, 0.05):
    t = time.time()
    ok, st = vf.certify_margin_radius(w, U, x_nom[0], iu, isf, rho,
                                      max_boxes=1024, max_iters=16)
    say(f"  rho={rho:<7} certified_safe={str(ok):<6} worst={st.get('worst', 0):+.4e}  "
        f"boxes={st.get('boxes_touched', 0):<6} {time.time()-t:.1f}s")
OUT.close()
