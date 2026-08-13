"""Locate the source of bound blow-up, sub-step by sub-step."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import bounds as bd, toy_transformer as tt, task

torch.manual_seed(0)
model = tt.ToyTransformer()
w = model.export_weights()
rng = np.random.default_rng(0)
toks, _ = task.sample_batch(4, rng)
with torch.no_grad():
    tr = [t.numpy().astype(np.float64) for t in model.residual_trace(torch.from_numpy(toks))]
x_nom = tr[0][0]
rho = 0.02
z = bd.HZono.from_box((x_nom - rho)[None], (x_nom + rho)[None])
bw = w["blocks"][0]


def show(tag, zz):
    lo, hi = zz.bounds()
    print(f"  {tag:22s} width_mean={float((hi-lo).mean()):12.4e}  "
          f"|G|={float(np.abs(zz.G).sum(1).mean()):10.4e}  E={float(zz.E.mean()):10.4e}")


show("input", z)
y = bd.layernorm(z, bw["ln1_g"], bw["ln1_b"], w["ln_eps"])
show("ln1", y)

d = 32
P0 = np.eye(d) - np.ones((d, d)) / d
xc = z.linear(P0)
lo, hi = xc.bounds()
cn = np.linalg.norm(xc.c, axis=-1, keepdims=True)
R2 = np.linalg.norm(xc.G, axis=-1).sum(axis=1)[..., None] + np.linalg.norm(xc.E, axis=-1, keepdims=True)
print("   centre norm per pos:", cn.ravel())
print("   R2 per pos         :", R2.ravel())
var_lo = np.maximum(cn - R2, 0) ** 2 / d + 1e-5
var_hi = (cn + R2) ** 2 / d + 1e-5
print("   1/sqrt(var) bracket:", (1 / np.sqrt(var_hi)).ravel(), "..", (1 / np.sqrt(var_lo)).ravel())

q = y.linear(bw["WQ"]); k = y.linear(bw["WK"]); v = y.linear(bw["WV"])
show("q", q); show("k", k); show("v", v)

B, T, D = y.c.shape
H, dh = w["n_heads"], D // w["n_heads"]
qc = q.c.reshape(B, T, H, dh); kc = k.c.reshape(B, T, H, dh)
qG = q.G.reshape(B, -1, T, H, dh); kG = k.G.reshape(B, -1, T, H, dh)
qr = q.radius().reshape(B, T, H, dh); kr = k.radius().reshape(B, T, H, dh)
scale = 1 / np.sqrt(dh)
sc_c = np.einsum("bthd,buhd->bhtu", qc, kc) * scale
sc_G = (np.einsum("bmthd,buhd->bhmtu", qG, kc) + np.einsum("bthd,bmuhd->bhmtu", qc, kG)) * scale
sc_E = np.einsum("bthd,buhd->bhtu", qr, kr) * scale
sc_r = np.abs(sc_G).sum(axis=2) + sc_E
print(f"   scores: |c|max={np.abs(sc_c).max():.4e}  radius_mean={sc_r.mean():.4e}  radius_max={sc_r.max():.4e}")
print(f"           from G={np.abs(sc_G).sum(2).mean():.4e}  from E(2nd order)={sc_E.mean():.4e}")

mask = np.broadcast_to(bd.causal_mask(T)[None, None], sc_c.shape)
a_lo, a_hi = bd.softmax_interval(sc_c - sc_r, sc_c + sc_r, mask)
print(f"   attn weight spread mean={float((a_hi-a_lo).mean()):.4e} max={float((a_hi-a_lo).max()):.4e}")

out, _ = bd.attention(y, bw["WQ"], bw["WK"], bw["WV"], bw["WO"], H)
show("attn out", out)
z1 = z + out
show("x + attn", z1)
y2 = bd.layernorm(z1, bw["ln2_g"], bw["ln2_b"], w["ln_eps"])
show("ln2", y2)
h = y2.linear(bw["fc_in_W"], bw["fc_in_b"])
show("fc_in", h)
hr = bd.relu(h)
show("relu", hr)
mo = hr.linear(bw["fc_out_W"], bw["fc_out_b"])
show("mlp out", mo)
show("block out", z1 + mo)
