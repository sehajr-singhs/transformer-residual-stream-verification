"""Validate the CROWN reference engine: exactness, soundness, containment.

An independent engine that is merely loose is still useful as a witness; an
independent engine that is WRONG is worse than none. These checks run before any
claim is made from it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from src import toy_transformer as tt, task, sae as sae_mod
import crown_reference as cr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "checkpoints")
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
x_nom = tr[0][0].numpy().astype(np.float64)
iu, isf = task.margin_readout()
U_t, xn_t = torch.from_numpy(U), torch.from_numpy(x_nom)


def true_margin(al):
    a = torch.as_tensor(al, dtype=torch.float64)
    x = xn_t[None] + torch.einsum("bk,ktd->btd", a, U_t)
    with torch.no_grad():
        for blk in model.blocks:
            x = blk(x)
        L = model.logits_from_stream(x)[:, -1]
    return task.unsafe_margin(L.numpy())


# --- 1. exact at a point
rng = np.random.default_rng(4)
err = []
for _ in range(10):
    a = rng.uniform(-0.04, 0.04, size=4)
    err.append(abs(cr.certify_margin(a, a, U, x_nom, w, iu, isf) - true_margin(a[None])[0]))
print(f"degenerate box: max |bound - true| = {max(err):.3e}")
assert max(err) < 1e-6, "CROWN margin not exact at a point"

# --- 2. soundness of the margin bound on real boxes
worst = -np.inf
for rho in (1e-4, 1e-3, 5e-3, 0.01, 0.04):
    for _ in range(4):
        c = rng.uniform(-rho, rho, size=4)
        r = rng.uniform(0.1, 1.0) * rho
        a_lo, a_hi = c - r, c + r
        b = cr.certify_margin(a_lo, a_hi, U, x_nom, w, iu, isf)
        pts = a_lo + rng.random((3000, 4)) * (a_hi - a_lo)
        worst = max(worst, float((true_margin(pts) - b).max()))
print(f"margin soundness : max (true - bound) = {worst:+.3e}")
assert worst <= 1e-9, "CROWN margin bound is UNSOUND"

# --- 3. containment of intermediate activations
viol = -np.inf
for rho in (1e-4, 1e-3, 5e-3):
    a_lo, a_hi = np.full(4, -rho), np.full(4, rho)
    mid, rad = np.zeros(4), np.full(4, rho)
    L = cr.LB.exact(U.copy(), x_nom.copy())
    al = rng.uniform(-rho, rho, size=(2000, 4))
    xs = torch.from_numpy(np.einsum("bk,ktd->btd", al, U) + x_nom[None])
    with torch.no_grad():
        x = xs
        for l in range(w["n_layers"]):
            L = cr.block(L, w["blocks"][l], w["n_heads"], w["ln_eps"], mid, rad)
            x = model.blocks[l](x)
            lo, hi = L.concretize(mid, rad)
            xn = x.numpy()
            viol = max(viol, float(np.max(np.maximum(lo[None] - xn, xn - hi[None]))))
print(f"stream containment: max violation    = {viol:+.3e}")
assert viol <= 1e-9, "CROWN intermediate bounds do not contain the true stream"
print("\nCROWN reference engine validated -- exact at a point, sound on boxes")
