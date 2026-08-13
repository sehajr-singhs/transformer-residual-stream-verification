"""Certified radius (sound BaB) vs actual breaking radius (PGD). Their ratio is
the geometric gap: the fraction of genuine robustness the prover can capture."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import toy_transformer as tt, task, sae as sae_mod, verifier as vf, bounds as bd
OUT = open(os.path.join(os.path.dirname(__file__), "calib4.txt"), "w")
def say(s):
    print(s, flush=True); OUT.write(s+"\n"); OUT.flush()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT,"checkpoints")
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
Ut = torch.as_tensor(U, dtype=torch.float32)
iu,isf = task.margin_readout()

say("BaB with a larger budget:")
say(f"{'rho':>8}{'proved':>8}{'worst':>12}{'boxes':>8}{'reason':>11}{'sec':>7}")
for rho in (0.025, 0.03, 0.04, 0.06):
    t=time.time()
    ok,st = vf.certify_margin_radius(w,U,x_nom[0],iu,isf,rho,
                                     max_boxes=8192,max_iters=20,chunk=128)
    say(f"{rho:>8}{str(ok):>8}{st.get('worst',0):>12.4e}{st.get('boxes_touched',0):>8}"
        f"{st.get('reason','proved'):>11}{time.time()-t:>7.1f}")

say("\nPGD attack on the margin -- where does the model ACTUALLY break?")
xn = torch.as_tensor(x_nom[0], dtype=torch.float32)[None]
for rho in (0.02, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0):
    a = ((torch.rand((512,4))*2-1)*rho).requires_grad_(True)
    best = -1e9
    for _ in range(250):
        e = torch.einsum("bk,ktd->btd", a, Ut)
        lg = model.logits_from_stream(model.blocks[1](model.blocks[0](xn+e)))[:, -1]
        s = lg[:, iu].max(1).values - lg[:, isf].max(1).values
        g, = torch.autograd.grad(-s.sum(), a)
        with torch.no_grad():
            a -= (rho/10)*g.sign(); a.clamp_(-rho, rho)
            best = max(best, float(s.detach().max()))
    say(f"  rho={rho:<6} PGD max margin={best:+.4f}  broken={best>0}")
OUT.close()
