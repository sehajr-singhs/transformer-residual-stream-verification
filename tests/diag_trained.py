"""Wrapping factor on a TRAINED model, across rho and propagation modes."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import bounds as bd, toy_transformer as tt, task, soundness

t0 = time.time()
model, hist = tt.train(steps=2500, log_every=1000, verbose=True)
print("eval:", json.dumps(tt.evaluate(model), indent=2))
w = model.export_weights()
rng = np.random.default_rng(0)
toks, _ = task.enumerate_prompts(limit=8, rng=rng, safe_only=True)
with torch.no_grad():
    tr = [t.numpy().astype(np.float64) for t in model.residual_trace(torch.from_numpy(toks))]
x_nom = [t[0] for t in tr]
print("stream norms per layer:", [float(np.linalg.norm(t, axis=-1).mean()) for t in x_nom])

print("\n rho     mode        L0 wrap    L1 wrap    L0 width    L1 width   sound")
for rho in (0.005, 0.02, 0.05, 0.15):
    for mode, kw in (("promote", {}), ("naive", {"promote": False})):
        z = bd.HZono.from_box((x_nom[0] - rho)[None], (x_nom[0] + rho)[None])
        widths = []
        zz = z
        for l in range(w["n_layers"]):
            zz, _ = bd.block(zz, w["blocks"][l], w["n_heads"], w["ln_eps"], **kw)
            lo, hi = zz.bounds()
            widths.append(float((hi - lo).mean()))
        rep = soundness.check_block_bounds(model, w, x_nom[0], rho, n=600, seed=0)
        att = [rep[f"layer{l}"]["mean_attained_width"] for l in range(w["n_layers"])]
        snd = all(rep[f"layer{l}"]["sound"] for l in range(w["n_layers"]))
        print(f" {rho:<6} {mode:<10} {widths[0]/att[0]:9.2f}  {widths[1]/att[1]:9.2f}  "
              f"{widths[0]:9.3e}  {widths[1]:9.3e}   {snd}")

print(f"\n({time.time()-t0:.0f}s)")
np.save(os.path.join(os.path.dirname(__file__), "_nom.npy"), np.array(x_nom))
