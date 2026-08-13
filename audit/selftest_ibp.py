"""Validate the reference engine against torch before trusting it as a witness.

At rho=0 every interval is degenerate, so the IBP bound must reproduce the true
margin to floating-point. If this fails, the reference engine is mis-transcribed
and nothing it says about the prover means anything.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import toy_transformer as tt, task
import ibp_ref as ref

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = tt.ToyTransformer()
model.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", "model.pt")))
model.eval()
w = model.export_weights()
iu, isf = task.margin_readout()

prompts, _ = task.enumerate_prompts(limit=32, rng=np.random.default_rng(0),
                                    safe_only=True)
# The exported weights are float64 but the checkpoint was trained in float32.
# Compare against a float64 torch forward so the residual measured here is the
# reference engine's own error, not float32 rounding in the oracle.
model64 = model.double()
with torch.no_grad():
    tr = model64.residual_trace(torch.from_numpy(prompts))
    x0 = tr[0].numpy().astype(np.float64)
    true_final = tr[-1].numpy().astype(np.float64)
    true_logits = model64.logits_from_stream(tr[-1]).numpy().astype(np.float64)
true_margin = task.unsafe_margin(true_logits[:, -1])

err_stream, err_logit, err_margin = [], [], []
for i in range(len(prompts)):
    lo = hi = x0[i]
    flo, fhi, _ = ref.propagate(lo.copy(), hi.copy(), w)
    L_lo, L_hi = ref.readout(flo, fhi, w)
    m = ref.margin_upper(L_lo, L_hi, iu, isf)
    err_stream.append(np.abs(0.5 * (flo + fhi) - true_final[i]).max())
    err_logit.append(np.abs(0.5 * (L_lo + L_hi) - true_logits[i]).max())
    err_margin.append(abs(m - true_margin[i]))

print(f"prompts checked            : {len(prompts)}")
print(f"max |stream_mid - torch|   : {max(err_stream):.3e}")
print(f"max |logit_mid  - torch|   : {max(err_logit):.3e}")
print(f"max |margin_ibp - torch|   : {max(err_margin):.3e}")
assert max(err_stream) < 1e-6, "reference stream disagrees with torch"
assert max(err_logit) < 1e-6, "reference logits disagree with torch"
assert max(err_margin) < 1e-6, "reference margin disagrees with torch"
print("\nreference engine matches torch at rho=0 -- usable as an independent witness")
