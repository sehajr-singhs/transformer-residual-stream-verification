"""Fast shape + soundness smoke test. Run before anything expensive."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src import bounds as bd, toy_transformer as tt, icnn, sae as sae_mod, soundness, task

t0 = time.time()
print("[1] primitives")
prim = soundness.check_primitives()
print(json.dumps(prim, indent=2))
assert all(v["sound"] for v in prim.values()), "PRIMITIVE UNSOUND"

print("[2] untrained model block propagation")
torch.manual_seed(0)
model = tt.ToyTransformer()
w = model.export_weights()
rng = np.random.default_rng(0)
toks, _ = task.sample_batch(4, rng)
with torch.no_grad():
    tr = [t.numpy().astype(np.float64) for t in model.residual_trace(torch.from_numpy(toks))]
x_nom = [t[0] for t in tr]
print("   nominal stream shapes:", [t.shape for t in x_nom])

for rho in (0.02, 0.1):
    rep = soundness.check_block_bounds(model, w, x_nom[0], rho, n=800, seed=0)
    print(f"   rho={rho}: " + json.dumps(rep))
    assert all(v["sound"] for v in rep.values()), f"BLOCK BOUND UNSOUND at rho={rho}"

print("[3] V bounds (untrained SAE + V)")
acts = sae_mod.collect_activations(model, n=2000, seed=1)
sae, _ = sae_mod.train_sae(acts, d_dict=32, steps=200, log_every=1000, verbose=False)
V = icnn.ICNNLyapunov(sae, w["d_model"], w["seq_len"], widths=(24, 24), alpha=1e-3)
vw = V.export_weights()
print("   V(0) =", float(V(torch.zeros(1, w["seq_len"], w["d_model"])).item()))
vb = soundness.check_v_bounds(V, vw, 0.05, w["seq_len"], w["d_model"], n=1500, seed=1)
print("   " + json.dumps(vb))
assert vb["sound"], "V BOUND UNSOUND"

print("[4] decrease bound")
db = soundness.check_decrease_bound(model, V, w, vw, x_nom[0], x_nom[1], 0, 0.05,
                                    kappa=0.05, n=800, seed=2, pgd_steps=30)
print("   " + json.dumps(db))
assert db["sound"], "DECREASE BOUND UNSOUND"

print("[5] margin bound")
iu, isf = task.margin_readout()
from src import verifier as vf
m = vf.certify_safety_margin(w, x_nom[0], 0.05, iu, isf)
print("   sound margin upper bound:", m)
print("   backends:", vf.backend_report())
print("   mean-direction equivariance:", vf.check_mean_equivariance(model, toks, n=4))

print(f"\nSMOKE OK in {time.time()-t0:.1f}s")
