"""At rho=0 the propagation is exact, so the sound margin bound MUST equal the
true margin. That invariant would have caught the min/max bug in
unsafe_margin_upper immediately; it is pinned here so it cannot come back.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import toy_transformer as tt, task, bounds as bd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = tt.ToyTransformer()
model.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "model.pt")))
model.eval(); w = model.export_weights()
prompts, _ = task.enumerate_prompts(limit=16, rng=np.random.default_rng(0), safe_only=True)
iu, isf = task.margin_readout()
with torch.no_grad():
    tr = model.residual_trace(torch.from_numpy(prompts))
    x0 = tr[0].numpy().astype(np.float64)
    true = task.unsafe_margin(model.logits_from_stream(tr[-1])[:, -1].numpy())

z = bd.HZono.point(x0)
zL = z
for l in range(w["n_layers"]):
    zL, _ = bd.block(zL, w["blocks"][l], w["n_heads"], w["ln_eps"])
bound = bd.unsafe_margin_upper(bd.readout_logits(zL, w), iu, isf)
err = np.abs(bound - true).max()
print(f"  max |bound - true| at rho=0 over {len(true)} prompts: {err:.3e}")
print(f"  sample: bound={bound[:3]}  true={true[:3]}")
assert err < 1e-4, f"margin bound is not exact at rho=0 (err={err:.4e})"
assert (bound <= true + 1e-9).all() or err < 1e-4, "bound must be >= true (sound)"
print("margin exactness at rho=0 OK")
