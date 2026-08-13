import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from src import toy_transformer as tt, task, sae as sae_mod
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
xn = torch.as_tensor(x_nom[0], dtype=torch.float32)[None]
out=[]
for rho in (0.04, 0.1, 0.3, 1.0, 3.0, 10.0):
    a = ((torch.rand((256,4))*2-1)*rho).requires_grad_(True)
    best=-1e9
    for _ in range(120):
        e = torch.einsum("bk,ktd->btd", a, Ut)
        lg = model.logits_from_stream(model.blocks[1](model.blocks[0](xn+e)))[:,-1]
        s = lg[:,iu].max(1).values - lg[:,isf].max(1).values
        g, = torch.autograd.grad(-s.sum(), a)
        with torch.no_grad():
            a -= (rho/8)*g.sign(); a.clamp_(-rho,rho)
            best=max(best,float(s.detach().max()))
    line=f"  rho={rho:<6} PGD max margin={best:+.4f}  broken={best>0}"
    print(line, flush=True); out.append(line)
open(os.path.join(os.path.dirname(__file__),"calib5.txt"),"w").write("\n".join(out))
