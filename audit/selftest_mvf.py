"""Validate the mean-value engine before trusting it as a witness.

Three checks, in increasing strength:
  1. at a degenerate box the margin bound is the exact margin;
  2. at a degenerate box the interval Jacobian collapses to torch's autograd
     Jacobian d(logits)/d(alpha) -- this is the check that would catch a wrong
     derivative rule in LayerNorm, softmax or attention;
  3. on non-degenerate boxes the enclosure actually contains the autograd
     Jacobian sampled at many interior points.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from src import toy_transformer as tt, task, sae as sae_mod
import mvf_ref as mvf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "checkpoints")
model = tt.ToyTransformer()
model.load_state_dict(torch.load(os.path.join(CK, "model.pt")))
model.eval().double()
w = model.export_weights()
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
    tr = model.residual_trace(torch.from_numpy(prompts))
x_nom = tr[0][0].numpy().astype(np.float64)
iu, isf = task.margin_readout()
U_t = torch.from_numpy(U); xn_t = torch.from_numpy(x_nom)


def torch_logits(alpha):
    a = torch.tensor(np.asarray(alpha, dtype=np.float64), dtype=torch.float64,
                     requires_grad=True)
    x = xn_t + torch.einsum("k,ktd->td", a, U_t)
    for blk in model.blocks:
        x = blk(x[None])[0]
    return a, model.logits_from_stream(x[None])[0, -1]


def torch_jac(alpha):
    a, L = torch_logits(alpha)
    J = torch.zeros(len(alpha), L.numel(), dtype=torch.float64)
    for i in range(L.numel()):
        g, = torch.autograd.grad(L[i], a, retain_graph=(i < L.numel() - 1))
        J[:, i] = g
    return J.detach().numpy()


# --- 1 + 2: degenerate box
rng = np.random.default_rng(3)
err_m, err_J = [], []
for _ in range(12):
    a = rng.uniform(-0.04, 0.04, size=4)
    _, Lt = torch_logits(a)
    true_m = task.unsafe_margin(Lt.detach().numpy())
    b = mvf.certify_margin(a, a, U, x_nom, w, iu, isf)
    err_m.append(abs(b - true_m))
    _, (dlo, dhi) = mvf.jacobian_enclosure(a, a, U, x_nom, w)
    Jm = 0.5 * (dlo[:, -1, :] + dhi[:, -1, :])
    err_J.append(np.abs(Jm - torch_jac(a)).max())
print(f"degenerate box: max |margin - true|      = {max(err_m):.3e}")
print(f"degenerate box: max |J_interval - J_ad|  = {max(err_J):.3e}")
assert max(err_m) < 1e-9, "mean-value margin not exact at a point"
assert max(err_J) < 1e-8, "interval Jacobian disagrees with autograd at a point"

# --- 3: containment on real boxes
worst = -np.inf
for rho in (0.005, 0.01, 0.02, 0.04):
    for _ in range(6):
        c = rng.uniform(-rho, rho, size=4)
        r = rng.uniform(0.1, 1.0) * rho
        a_lo, a_hi = c - r, c + r
        _, (dlo, dhi) = mvf.jacobian_enclosure(a_lo, a_hi, U, x_nom, w)
        Jl, Jh = dlo[:, -1, :], dhi[:, -1, :]
        for _ in range(40):
            p = rng.uniform(a_lo, a_hi)
            J = torch_jac(p)
            worst = max(worst, float(np.max(np.maximum(Jl - J, J - Jh))))
print(f"non-degenerate: max autograd-J outside enclosure = {worst:+.3e}")
assert worst <= 1e-9, "interval Jacobian does not contain the true Jacobian"
print("\nmean-value engine validated -- usable as an independent witness")
