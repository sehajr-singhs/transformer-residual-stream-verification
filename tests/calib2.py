"""Pick the operating point: bound quality per second across ln_promote_k."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import toy_transformer as tt, task, sae as sae_mod, verifier as vf, bounds as bd
OUT = open(os.path.join(os.path.dirname(__file__), "calib2.txt"), "w")
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
U[np.arange(4), w["seq_len"]-1, :] = Wdec[pick]
iu, isf = task.margin_readout()
say(f"{'ln_k':>6}{'mlp_k':>7}{'m_out':>7}{'bound@.002':>13}{'ms/box':>9}")
for lnk, mlpk, mmax in ((0,0,8),(8,8,24),(16,16,48),(24,24,64),(48,48,128),(None,128,192)):
    B=64
    z = vf.alpha_to_zono(np.full((B,4),-0.002), np.full((B,4),0.002), U) + x_nom[0][None]
    kw = dict(m_max=mmax, promote_k=mlpk, ln_promote_k=lnk)
    t=time.time(); zL=z
    for l in range(w["n_layers"]):
        zL,_ = bd.block(zL, w["blocks"][l], w["n_heads"], w["ln_eps"], **kw)
    s = bd.unsafe_margin_upper(bd.readout_logits(zL,w), iu, isf)
    say(f"{str(lnk):>6}{mlpk:>7}{zL.m:>7}{float(s[0]):>13.4e}{(time.time()-t)/B*1000:>9.2f}")
OUT.close()
