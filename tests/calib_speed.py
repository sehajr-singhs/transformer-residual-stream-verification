"""Speed/tightness operating point for BaB.

Promotion buys tightness but grows the generator count m from k=4 to ~600, and
cost is roughly linear in m through the MLP's (T, d_mlp) einsums. Branch-and-bound
converts THROUGHPUT into tightness too, so the right operating point is whichever
maximizes bound quality per unit time -- not whichever gives the best single-pass
bound. Measured here rather than assumed.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import toy_transformer as tt, task, sae as sae_mod, verifier as vf, bounds as bd

OUT = open(os.path.join(os.path.dirname(__file__), "calib_speed.txt"), "w")
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
iu, isf = task.margin_readout()

CONFIGS = [
    ("no-promote (m=4)", {"promote": False}),
    ("promote k=16", {"m_max": 16, "promote_k": 8}),
    ("promote k=48", {"m_max": 48, "promote_k": 24}),
    ("promote full", {}),
]

say(f"{'config':<22}{'rho':>8}{'m_out':>7}{'margin bound':>15}{'ms/box':>9}")
for nm, kw in CONFIGS:
    for rho in (0.002, 0.005):
        B = 64
        a_lo = np.full((B, 4), -rho); a_hi = np.full((B, 4), rho)
        z = vf.alpha_to_zono(a_lo, a_hi, U) + x_nom[0][None]
        t = time.time()
        zL = z
        for l in range(w["n_layers"]):
            zL, _ = bd.block(zL, w["blocks"][l], w["n_heads"], w["ln_eps"], **kw)
        s = bd.unsafe_margin_upper(bd.readout_logits(zL, w), iu, isf)
        dt = (time.time() - t) / B * 1000
        say(f"{nm:<22}{rho:>8}{zL.m:>7}{float(s[0]):>15.4e}{dt:>9.2f}")

say("\nBaB with the cheap configuration (no promotion), generous box budget:")
say(f"{'rho':>8}{'proved':>8}{'worst':>13}{'boxes':>8}{'reason':>12}{'sec':>7}")
import src.bounds as _bd
_orig = _bd.block


def cheap_block(z, bw, n_heads, ln_eps=1e-5, **kw):
    return _orig(z, bw, n_heads, ln_eps, promote=False)


_bd.block = cheap_block
for rho in (0.001, 0.002, 0.003, 0.005, 0.01):
    t = time.time()
    ok, st = vf.certify_margin_radius(w, U, x_nom[0], iu, isf, rho,
                                      max_boxes=20000, max_iters=30, chunk=256)
    say(f"{rho:>8}{str(ok):>8}{st.get('worst', 0):>13.4e}"
        f"{st.get('boxes_touched', 0):>8}{st.get('reason', 'proved'):>12}"
        f"{time.time()-t:>7.1f}")
_bd.block = _orig
OUT.close()
