"""Is the layer-to-layer deviation dynamics contractive at all?

If the Jacobian of e -> Block_l(x* + e) - x*_{l+1} has singular values above 1
in the directions we certify over, then NO positive-definite V can satisfy
V(e') <= (1-kappa) V(e) there, and the decrease formulation is asking for a
theorem that is false. This is a five-minute check that determines whether the
whole obligation is well posed, so it belongs before any verifier tuning.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import toy_transformer as tt, task, sae as sae_mod

CKPT = os.path.join(os.path.dirname(__file__), "_model.pt")
model = tt.ToyTransformer(); model.load_state_dict(torch.load(CKPT)); model.eval()
w = model.export_weights()
T, D = w["seq_len"], w["d_model"]
rng = np.random.default_rng(0)
toks, _ = task.enumerate_prompts(limit=32, rng=rng, safe_only=True)
with torch.no_grad():
    tr = model.residual_trace(torch.from_numpy(toks))


def jac(layer, x):
    """Full (T*D, T*D) Jacobian of block `layer` at stream x (T,D)."""
    xt = torch.as_tensor(x, dtype=torch.float64)[None].requires_grad_(True)
    blk = model.blocks[layer].double()
    out = blk(xt).reshape(-1)
    J = torch.zeros(out.numel(), xt.numel(), dtype=torch.float64)
    for i in range(out.numel()):
        g, = torch.autograd.grad(out[i], xt, retain_graph=(i < out.numel() - 1))
        J[i] = g.reshape(-1)
    model.blocks[layer].float()
    return J.numpy()


print("Per-layer deviation Jacobian, averaged over 8 prompt classes\n")
sv_all = {0: [], 1: []}
comp = []
for p in range(8):
    x0 = tr[0][p].detach().numpy().astype(np.float64)
    x1 = tr[1][p].detach().numpy().astype(np.float64)
    J0 = jac(0, x0); J1 = jac(1, x1)
    sv_all[0].append(np.linalg.svd(J0, compute_uv=False))
    sv_all[1].append(np.linalg.svd(J1, compute_uv=False))
    comp.append(np.linalg.svd(J1 @ J0, compute_uv=False))

for l in (0, 1):
    S = np.array(sv_all[l])
    print(f"  layer {l}: sigma_max={S[:,0].mean():7.3f}   sigma_min={S[:,-1].mean():7.4f}   "
          f"frac(sigma>1)={float((S>1).mean()):.3f}")
C = np.array(comp)
print(f"  2-layer composite: sigma_max={C[:,0].mean():7.3f}  frac(sigma>1)={float((C>1).mean()):.3f}")

print("\nRestricted to the SAE feature subspace we certify over:")
SAECK = os.path.join(os.path.dirname(__file__), "_sae.pt")
sae = sae_mod.SAE(D, 64); sae.load_state_dict(torch.load(SAECK))
Wdec = sae.W_dec.detach().numpy().astype(np.float64).T
Wdec = Wdec / np.linalg.norm(Wdec, axis=1, keepdims=True)
pick = np.random.default_rng(11).choice(Wdec.shape[0], size=4, replace=False)
U = np.zeros((4, T, D)); U[np.arange(4), T - 1, :] = Wdec[pick]
Uf = U.reshape(4, -1)

x0 = tr[0][0].detach().numpy().astype(np.float64)
x1 = tr[1][0].detach().numpy().astype(np.float64)
J0, J1 = jac(0, x0), jac(1, x1)
for nm, J in (("layer0", J0), ("layer1", J1), ("composite", J1 @ J0)):
    M = J @ Uf.T                      # image of the k unit directions
    g = np.linalg.norm(M, axis=0)
    print(f"  {nm:10s} ||J u_j||/||u_j|| = {np.round(g, 3)}   max={g.max():.3f}")

print("\nSpectral radius -- decides whether ANY metric can show contraction:")
for nm, J in (("layer0", J0), ("layer1", J1), ("composite", J1 @ J0)):
    ev = np.abs(np.linalg.eigvals(J))
    print(f"  {nm:10s} rho(J)={ev.max():7.3f}   #|lambda|>1 = {int((ev > 1).sum())}/{len(ev)}")
# restricted to the certified subspace: project the composite onto span(U)
Pu = Uf.T @ np.linalg.pinv(Uf.T)
Jr = Uf @ (J1 @ J0) @ Uf.T
ev = np.abs(np.linalg.eigvals(Jr))
print(f"  composite|span(U): rho={ev.max():.3f}  eigs={np.round(np.sort(ev)[::-1], 3)}")
print("  => a contraction metric exists iff rho < 1; growth certificate needed otherwise")

print("\nGrowth of the actual perturbation along layers (empirical, PGD-free):")
x2 = tr[2][0].detach().numpy().astype(np.float64)
for rho in (0.01, 0.05, 0.2):
    a = np.random.default_rng(3).uniform(-rho, rho, size=(2000, 4))
    e0 = np.einsum("bk,ktd->btd", a, U)
    with torch.no_grad():
        xt = torch.as_tensor(x0 + e0, dtype=torch.float32)
        s1 = model.blocks[0](xt)
        y1 = s1.numpy() - x1
        y2 = model.blocks[1](s1).numpy() - x2
    n0 = np.linalg.norm(e0.reshape(len(e0), -1), axis=1)
    n1 = np.linalg.norm(y1.reshape(len(y1), -1), axis=1)
    n2 = np.linalg.norm(y2.reshape(len(y2), -1), axis=1)
    print(f"  rho={rho:<6} ||e0||={n0.mean():.4f} -> ||e1||={n1.mean():.4f} "
          f"({(n1/n0).mean():.3f}x) -> ||e2||={n2.mean():.4f} ({(n2/n1).mean():.3f}x)")
