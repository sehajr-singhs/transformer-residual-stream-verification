"""Find the regime where the certificate is actually provable.

Sweeps perturbation subspace dim k and radius rho with FIXED directions, and
reports the two quantities that decide feasibility:
  - unstable (crossing) ReLU count: the source of spurious degrees of freedom
  - LayerNorm scale spread at layer 1: the quantity whose collapse detonates
    the whole propagation
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import bounds as bd, toy_transformer as tt, task, sae as sae_mod

CKPT = os.path.join(os.path.dirname(__file__), "_model.pt")
SAECK = os.path.join(os.path.dirname(__file__), "_sae.pt")
model = tt.ToyTransformer(); model.load_state_dict(torch.load(CKPT))
w = model.export_weights()
rng = np.random.default_rng(0)
toks, _ = task.enumerate_prompts(limit=8, rng=rng, safe_only=True)
with torch.no_grad():
    tr = [t.numpy().astype(np.float64) for t in model.residual_trace(torch.from_numpy(toks))]
x_nom = tr[0][0]
T, D = x_nom.shape

acts = sae_mod.collect_activations(model, n=20000, seed=7)
if os.path.exists(SAECK):
    sae = sae_mod.SAE(D, 64); sae.load_state_dict(torch.load(SAECK))
else:
    sae, _ = sae_mod.train_sae(acts, d_dict=64, steps=3000, log_every=3000, verbose=False)
    torch.save(sae.state_dict(), SAECK)
Wdec = sae.W_dec.detach().numpy().astype(np.float64).T
Wdec = Wdec / np.linalg.norm(Wdec, axis=1, keepdims=True)
FIX = np.random.default_rng(11).choice(Wdec.shape[0], size=32, replace=False)


def dirs(k, last_only=True):
    U = np.zeros((k, T, D))
    for j in range(k):
        if last_only:
            U[j, T - 1, :] = Wdec[FIX[j]]
        else:
            U[j, :, :] = Wdec[FIX[j]][None] / np.sqrt(T)
    return U


def probe(z0):
    """Propagate, reporting crossings and LN spread per layer."""
    info = []
    zz = z0
    for l in range(w["n_layers"]):
        bw = w["blocks"][l]
        zc = zz.promote_E().compact(192)
        # ln1 scale spread
        P0 = np.eye(D) - np.ones((D, D)) / D
        xc = zc.linear(P0)
        tot = np.abs(xc.G).sum(1) + xc.E
        cn = np.linalg.norm(xc.c, axis=-1); R2 = np.linalg.norm(tot, axis=-1)
        y1 = bd.layernorm(zc, bw["ln1_g"], bw["ln1_b"], w["ln_eps"]).promote_E()
        a, _ = bd.attention(y1, bw["WQ"], bw["WK"], bw["WV"], bw["WO"], w["n_heads"])
        z1 = zc + a
        y2 = bd.layernorm(z1, bw["ln2_g"], bw["ln2_b"], w["ln_eps"]).promote_E()
        h = y2.linear(bw["fc_in_W"], bw["fc_in_b"])
        hl, hh = h.bounds()
        cross = int(((hl < 0) & (hh > 0)).sum())
        zz, _ = bd.block(zz, bw, w["n_heads"], w["ln_eps"])
        lo, hi = zz.bounds()
        info.append({"layer": l, "R2_over_cn": float((R2 / cn).max()),
                     "crossing": cross, "of": int(hl.size),
                     "out_width": float((hi - lo).mean())})
    return info


print(f"{'k':>4}{'rho':>8}{'L0 R2/cn':>11}{'L0 cross':>10}{'L0 wid':>11}"
      f"{'L1 R2/cn':>11}{'L1 cross':>10}{'L1 wid':>11}")
for k in (2, 4, 8, 16):
    for rho in (0.005, 0.02, 0.05, 0.1, 0.25):
        U = dirs(k)
        z = bd.HZono.from_subspace(x_nom[None], U, rho)
        try:
            inf = probe(z)
            print(f"{k:>4}{rho:>8}{inf[0]['R2_over_cn']:>11.3f}{inf[0]['crossing']:>10}"
                  f"{inf[0]['out_width']:>11.3e}{inf[1]['R2_over_cn']:>11.3f}"
                  f"{inf[1]['crossing']:>10}{inf[1]['out_width']:>11.3e}")
        except Exception as e:
            print(f"{k:>4}{rho:>8}   FAILED {e}")

print("\nfull 160-D box for comparison")
for rho in (0.002, 0.005, 0.02):
    z = bd.HZono.from_box((x_nom - rho)[None], (x_nom + rho)[None])
    inf = probe(z)
    print(f"{'box':>4}{rho:>8}{inf[0]['R2_over_cn']:>11.3f}{inf[0]['crossing']:>10}"
          f"{inf[0]['out_width']:>11.3e}{inf[1]['R2_over_cn']:>11.3f}"
          f"{inf[1]['crossing']:>10}{inf[1]['out_width']:>11.3e}")
