"""The safety-relevant obligation, run on its own.

The margin condition is ADDITIVE (prove s <= 0 where the nominal s is -9.56),
not a RATIO like V(e')/V(e). Additive slack is exactly what branch-and-bound is
good at closing, so this is where a real certificate is most likely to land.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import toy_transformer as tt, task, sae as sae_mod, verifier as vf, bounds as bd

OUT = open(os.path.join(os.path.dirname(__file__), "probe_margin.txt"), "w")
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

say(f"nominal unsafe-logit margin at anchor: "
    f"{float(task.unsafe_margin(model(torch.from_numpy(prompts))[0,-1].detach().numpy())):.4f}")
say("\nsingle-pass sound margin bound (no branching):")
for rho in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25):
    z = bd.HZono.from_subspace(x_nom[0][None], U, rho)
    zL = z
    for l in range(w["n_layers"]):
        zL, _ = bd.block(zL, w["blocks"][l], w["n_heads"], w["ln_eps"])
    s = bd.unsafe_margin_upper(bd.readout_logits(zL, w), iu, isf)
    a = np.random.default_rng(0).uniform(-rho, rho, size=(4000, 4))
    with torch.no_grad():
        xs = torch.as_tensor(x_nom[0][None] + np.einsum("bk,ktd->btd", a, U),
                             dtype=torch.float32)
        lg = model.logits_from_stream(model.blocks[1](model.blocks[0](xs)))
        true = task.unsafe_margin(lg[:, -1].numpy()).max()
    say(f"  rho={rho:<6} bound={float(s[0]):+.4e}  sampled_max={true:+.4f}  "
        f"slack={float(s[0])-true:+.4e}  proved={bool(s[0] < 0)}")

say("\nwith branch-and-bound:")
for rho in (0.001, 0.002, 0.003, 0.005, 0.008):
    t = time.time()
    ok, st = vf.certify_margin_radius(w, U, x_nom[0], iu, isf, rho,
                                      max_boxes=8192, max_iters=24)
    say(f"  rho={rho:<6} certified_safe={str(ok):<6} worst={st.get('worst',0):+.4e} "
        f"boxes={st.get('boxes_touched',0):<6} reason={st.get('reason','proved')} "
        f"{time.time()-t:.0f}s")
OUT.close()
