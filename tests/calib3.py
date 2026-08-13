"""How fast does the margin bound actually fall with radius and with splitting?"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import toy_transformer as tt, task, sae as sae_mod, verifier as vf, bounds as bd
OUT = open(os.path.join(os.path.dirname(__file__), "calib3.txt"), "w")
def say(s):
    print(s, flush=True); OUT.write(s + "\n"); OUT.flush()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "checkpoints")
model = tt.ToyTransformer(); model.load_state_dict(torch.load(os.path.join(CK,"model.pt")))
model.eval(); w = model.export_weights()
sae = sae_mod.SAE(w["d_model"],64); sae.load_state_dict(torch.load(os.path.join(CK,"sae.pt")))
prompts,_ = task.enumerate_prompts(limit=256, rng=np.random.default_rng(0), safe_only=True)
with torch.no_grad():
    tr=[t.numpy().astype(np.float64) for t in model.residual_trace(torch.from_numpy(prompts))]
x_nom=[t[0] for t in tr]
Wdec = sae.W_dec.detach().numpy().astype(np.float64).T
Wdec = Wdec/np.linalg.norm(Wdec,axis=1,keepdims=True)
pick = np.random.default_rng(11).choice(Wdec.shape[0],size=4,replace=False)
U=np.zeros((4,w["seq_len"],w["d_model"])); U[np.arange(4),w["seq_len"]-1,:]=Wdec[pick]
iu,isf = task.margin_readout()

say("single-pass margin bound vs radius (ln_k=48 default config):")
for rho in (5e-3,7e-3,8e-3,9e-3,1e-2,1.2e-2,1.5e-2):
    z = bd.HZono.from_subspace(x_nom[0][None], U, rho); zL=z
    for l in range(w["n_layers"]):
        zL,_ = bd.block(zL, w["blocks"][l], w["n_heads"], w["ln_eps"])
    s = float(bd.unsafe_margin_upper(bd.readout_logits(zL,w), iu, isf)[0])
    say(f"  rho={rho:<8.1e} bound={s:+.4e}  proved={s<0}")

say("\nBaB (capped budgets so it cannot run away):")
say(f"{'rho':>9}{'proved':>8}{'worst':>13}{'boxes':>8}{'reason':>11}{'sec':>7}")
for rho in (0.008, 0.01, 0.015, 0.02, 0.03):
    t=time.time()
    ok,st = vf.certify_margin_radius(w,U,x_nom[0],iu,isf,rho,
                                     max_boxes=512,max_iters=12,chunk=64)
    say(f"{rho:>9.1e}{str(ok):>8}{st.get('worst',0):>13.4e}"
        f"{st.get('boxes_touched',0):>8}{st.get('reason','proved'):>11}{time.time()-t:>7.1f}")
OUT.close()
