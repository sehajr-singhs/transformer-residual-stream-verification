"""Calibrate BaB cost before launching a long certification run."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import toy_transformer as tt, task, sae as sae_mod, icnn, verifier as vf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "checkpoints")
model = tt.ToyTransformer(); model.load_state_dict(torch.load(os.path.join(CK, "model.pt")))
model.eval()
w = model.export_weights()
sae = sae_mod.SAE(w["d_model"], 64)
sae.load_state_dict(torch.load(os.path.join(CK, "sae.pt")))
prompts, _ = task.enumerate_prompts(limit=256, rng=np.random.default_rng(0), safe_only=True)
with torch.no_grad():
    tr = [t.numpy().astype(np.float64) for t in model.residual_trace(torch.from_numpy(prompts))]
x_nom = [t[0] for t in tr]

Wdec = sae.W_dec.detach().numpy().astype(np.float64).T
Wdec = Wdec / np.linalg.norm(Wdec, axis=1, keepdims=True)
pick = np.random.default_rng(11).choice(Wdec.shape[0], size=4, replace=False)
U = np.zeros((4, w["seq_len"], w["d_model"]))
U[np.arange(4), w["seq_len"] - 1, :] = Wdec[pick]

blob = torch.load(os.path.join(CK, "lyap.pt"))
V = icnn.ICNNLyapunov(sae, w["d_model"], w["seq_len"], widths=(64, 64), alpha=1e-3)
V.load_state_dict(blob["sd"]); V.eval()
vw = V.export_weights()
print("trained gamma:", blob["gamma"])

# cost of a single batched evaluation
for B in (32, 128):
    a_lo = np.full((B, 4), -0.05); a_hi = np.full((B, 4), 0.05)
    t = time.time()
    ub = vf._eval_alpha(a_lo, a_hi, U, w, vw, 0, x_nom[0], x_nom[1], 1.4, 32)
    dt = time.time() - t
    print(f"  eval {B:4d} boxes: {dt:6.2f}s  ({1000*dt/B:.1f} ms/box)  "
          f"worst={ub.max():+.3e}")

print("\nsingle certify_growth calls:")
for gamma in (1.4, 2.0, 4.0):
    t = time.time()
    ok, st = vf.certify_growth(w, vw, U, x_nom[0], x_nom[1], 0, 0.05, gamma,
                               max_boxes=512, max_iters=14, chunk=32)
    print(f"  gamma={gamma:<5} proved={ok}  {time.time()-t:6.1f}s  "
          f"{json.dumps({k: v for k, v in st.items() if k != 'stats'})}")
