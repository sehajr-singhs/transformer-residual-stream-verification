"""Wrapping factor vs perturbation-subspace dimension k.

The central quantitative question of the architecture: does confining the
threat model to k SAE feature directions actually buy tractable bounds?
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import bounds as bd, toy_transformer as tt, task, sae as sae_mod

CKPT = os.path.join(os.path.dirname(__file__), "_model.pt")
model = tt.ToyTransformer(); model.load_state_dict(torch.load(CKPT))
w = model.export_weights()
rng = np.random.default_rng(0)
toks, _ = task.enumerate_prompts(limit=8, rng=rng, safe_only=True)
with torch.no_grad():
    tr = [t.numpy().astype(np.float64) for t in model.residual_trace(torch.from_numpy(toks))]
x_nom = tr[0][0]
T, D = x_nom.shape

print("training SAE for feature directions...")
acts = sae_mod.collect_activations(model, n=20000, seed=7)
sae, _ = sae_mod.train_sae(acts, d_dict=64, steps=3000, log_every=3000, verbose=False)
print("  bridge:", json.dumps(sae_mod.bridge_gap(sae, acts), indent=None))
Wdec = sae.W_dec.detach().numpy().astype(np.float64).T      # (d_dict, d_model)
Wdec = Wdec / np.linalg.norm(Wdec, axis=1, keepdims=True)

rr = np.random.default_rng(5)


def feature_dirs(k, pos="all"):
    """k perturbation directions built from SAE decoder atoms."""
    picks = rr.choice(Wdec.shape[0], size=k, replace=False)
    U = np.zeros((k, T, D))
    for j, p in enumerate(picks):
        if pos == "all":
            U[j, :, :] = Wdec[p][None, :] / np.sqrt(T)
        else:
            U[j, pos, :] = Wdec[p]
    return U


print(f"\n{'set':<26}{'k':>5}{'rho':>8}{'L0 bound':>12}{'L1 bound':>12}"
      f"{'L0 true':>11}{'L1 true':>11}{'L1 wrap':>11}  sound")
for label, mk in (("SAE feats (last pos)", lambda k: feature_dirs(k, pos=T - 1)),
                  ("SAE feats (all pos)", lambda k: feature_dirs(k, pos="all"))):
    for k in (4, 8, 16, 32):
        for rho in (0.05, 0.2):
            U = mk(k)
            z = bd.HZono.from_subspace(x_nom[None], U, rho)
            zz = z; wid = []
            for l in range(w["n_layers"]):
                zz, _ = bd.block(zz, w["blocks"][l], w["n_heads"], w["ln_eps"])
                lo, hi = zz.bounds(); wid.append(float((hi - lo).mean()))
            a = rr.uniform(-rho, rho, size=(4000, k))
            xs = x_nom[None] + np.einsum("nk,ktd->ntd", a, U)
            with torch.no_grad():
                t1 = model.blocks[0](torch.as_tensor(xs, dtype=torch.float32)).numpy()
                t2 = model.blocks[1](torch.as_tensor(t1)).numpy()
            tw = [float((t1.max(0) - t1.min(0)).mean()), float((t2.max(0) - t2.min(0)).mean())]
            lo1, hi1 = zz.bounds()
            snd = bool((t2 >= lo1[0] - 1e-6).all() and (t2 <= hi1[0] + 1e-6).all())
            print(f"{label:<26}{k:>5}{rho:>8}{wid[0]:>12.3e}{wid[1]:>12.3e}"
                  f"{tw[0]:>11.3e}{tw[1]:>11.3e}{wid[1]/tw[1]:>11.2f}  {snd}")

print("\nbaseline: full 160-D box")
for rho in (0.005, 0.02):
    z = bd.HZono.from_box((x_nom - rho)[None], (x_nom + rho)[None])
    zz = z; wid = []
    for l in range(w["n_layers"]):
        zz, _ = bd.block(zz, w["blocks"][l], w["n_heads"], w["ln_eps"])
        lo, hi = zz.bounds(); wid.append(float((hi - lo).mean()))
    print(f"  rho={rho}: L0={wid[0]:.3e} L1={wid[1]:.3e}")
