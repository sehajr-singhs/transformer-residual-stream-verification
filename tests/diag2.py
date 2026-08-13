"""Sub-step blow-up trace on a TRAINED model."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import bounds as bd, toy_transformer as tt, task

CKPT = os.path.join(os.path.dirname(__file__), "_model.pt")
model = tt.ToyTransformer()
if os.path.exists(CKPT):
    model.load_state_dict(torch.load(CKPT))
    print("loaded checkpoint")
else:
    model, _ = tt.train(steps=2500, log_every=2500, verbose=False)
    torch.save(model.state_dict(), CKPT)
    print("trained + saved")
w = model.export_weights()
rng = np.random.default_rng(0)
toks, _ = task.enumerate_prompts(limit=8, rng=rng, safe_only=True)
with torch.no_grad():
    tr = [t.numpy().astype(np.float64) for t in model.residual_trace(torch.from_numpy(toks))]
x_nom = tr[0][0]

for nm in ("WQ", "WK", "WV", "WO", "fc_in_W", "fc_out_W"):
    a = w["blocks"][0][nm]
    print(f"  |{nm}|: fro={np.linalg.norm(a):8.2f} spec={np.linalg.svd(a, compute_uv=False)[0]:8.2f} "
          f"l1row_max={np.abs(a).sum(1).max():8.2f}")
print("  ln1_g:", np.abs(w["blocks"][0]["ln1_g"]).max(), " ln2_g:", np.abs(w["blocks"][0]["ln2_g"]).max())

rho = 0.005
z = bd.HZono.from_box((x_nom - rho)[None], (x_nom + rho)[None])
bw = w["blocks"][0]


def show(tag, zz):
    lo, hi = zz.bounds()
    print(f"  {tag:22s} width={float((hi-lo).mean()):12.4e}  |G|={float(np.abs(zz.G).sum(1).mean()):11.4e}  "
          f"E={float(zz.E.mean()):11.4e}  m={zz.m}")


show("input", z)
zin = z.promote_E().compact(192)
y = bd.layernorm(zin, bw["ln1_g"], bw["ln1_b"], w["ln_eps"])
show("ln1 (raw)", y)
y = y.promote_E()
show("ln1 (promoted)", y)
B, T, D = y.c.shape
H, dh = w["n_heads"], D // w["n_heads"]
q = y.linear(bw["WQ"]); k = y.linear(bw["WK"]); v = y.linear(bw["WV"])
show("q", q); show("v", v)
qc = q.c.reshape(B, T, H, dh); kc = k.c.reshape(B, T, H, dh)
qG = q.G.reshape(B, -1, T, H, dh); kG = k.G.reshape(B, -1, T, H, dh)
qr = q.radius().reshape(B, T, H, dh); kr = k.radius().reshape(B, T, H, dh)
scale = 1 / np.sqrt(dh)
sc_c = np.einsum("bthd,buhd->bhtu", qc, kc) * scale
sc_G = (np.einsum("bmthd,buhd->bhmtu", qG, kc) + np.einsum("bthd,bmuhd->bhmtu", qc, kG)) * scale
sc_E = np.einsum("bthd,buhd->bhtu", qr, kr) * scale
sc_r = np.abs(sc_G).sum(axis=2) + sc_E
print(f"  scores |c|max={np.abs(sc_c).max():.3e}  r_mean={sc_r.mean():.3e}  r_max={sc_r.max():.3e}"
      f"   (G part {np.abs(sc_G).sum(2).mean():.3e}, 2nd-order E {sc_E.mean():.3e})")
mask = np.broadcast_to(bd.causal_mask(T)[None, None], sc_c.shape)
a_lo, a_hi = bd.softmax_interval(sc_c - sc_r, sc_c + sc_r, mask)
print(f"  attn spread mean={float((a_hi-a_lo).mean()):.3e} max={float((a_hi-a_lo).max()):.3e}")
out, _ = bd.attention(y, bw["WQ"], bw["WK"], bw["WV"], bw["WO"], H)
show("attn out", out)
z1 = zin + out
show("x+attn", z1)
y2 = bd.layernorm(z1, bw["ln2_g"], bw["ln2_b"], w["ln_eps"])
show("ln2 (raw)", y2)
y2 = y2.promote_E()
show("ln2 (promoted)", y2)
h = y2.linear(bw["fc_in_W"], bw["fc_in_b"]); show("fc_in", h)
hr = bd.relu(h); show("relu", hr)
lo_h, hi_h = h.bounds()
print(f"  hidden neurons: stable={int(((lo_h>=0)|(hi_h<=0)).sum())}/{lo_h.size} crossing={int((((lo_h<0)&(hi_h>0))).sum())}")
hp = hr.promote_E_topk(128)
mo = hp.linear(bw["fc_out_W"], bw["fc_out_b"]); show("mlp out", mo)
show("block out", (z1 + mo).compact(192))

print("\n  full block via bd.block:")
zz, _ = bd.block(z, bw, H, w["ln_eps"])
show("  block0 out", zz)
zz2, _ = bd.block(zz, w["blocks"][1], H, w["ln_eps"])
show("  block1 out", zz2)
with torch.no_grad():
    rr = np.random.default_rng(1)
    xs = x_nom + rr.uniform(-rho, rho, size=(4000,) + x_nom.shape)
    t1 = model.blocks[0](torch.as_tensor(xs, dtype=torch.float64).float()).numpy()
    t2 = model.blocks[1](torch.as_tensor(t1)).numpy()
print(f"    true attained widths: L0={float((t1.max(0)-t1.min(0)).mean()):.4e} "
      f"L1={float((t2.max(0)-t2.min(0)).mean()):.4e}")
