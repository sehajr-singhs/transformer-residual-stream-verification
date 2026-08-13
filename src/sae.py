"""Sparse autoencoder over the residual stream, plus the honest accounting of
the gap it introduces.

The architectural bet is that the dynamics should be certified in the SAE
feature frame rather than in raw activation coordinates. That bet buys two
things and costs one, and the cost is the part that decides whether this work
survives review:

  BUYS  (a) sparsity -> most encoder pre-activations are provably negative over
            a verification box, so those features are ELIMINATED from the
            branch-and-bound rather than branched on. `support_certificate`
            turns that into a number: the effective dimension.
        (b) an interpretable frame in which the safety specification can be
            written down as a condition on named features.

  COSTS     reconstruction error. Enc/Dec is not the identity. Any statement
            proved about feature-space dynamics transfers to activation-space
            dynamics only up to eps = ||x - Dec(Enc(x))||. We therefore never
            verify the SAE round trip as if it were exact. Instead the encoder
            is used as a READOUT (features are a function of x, x is the state),
            so no reconstruction error enters the dynamics at all, and the SAE
            error only affects how well V's level sets align with feature
            semantics. `bridge_gap` measures it either way, because the moment
            anyone proposes closing the loop through Dec, that number becomes
            the soundness bottleneck.
"""
from . import task  # noqa: F401

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import bounds as bd


class SAE(nn.Module):
    def __init__(self, d_model, d_dict, l1=3e-3):
        super().__init__()
        self.d_model, self.d_dict, self.l1 = d_model, d_dict, l1
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.randn(d_dict, d_model) / np.sqrt(d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_dict))
        self.W_dec = nn.Parameter(torch.randn(d_model, d_dict) / np.sqrt(d_dict))

    def normalize_decoder(self):
        with torch.no_grad():
            self.W_dec.data /= self.W_dec.data.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def encode(self, x):
        return F.relu((x - self.b_dec) @ self.W_enc.T + self.b_enc)

    def decode(self, f):
        return f @ self.W_dec.T + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f

    def loss(self, x):
        xh, f = self(x)
        recon = ((xh - x) ** 2).sum(-1).mean()
        l1 = f.abs().sum(-1).mean()
        return recon + self.l1 * l1, recon.detach(), l1.detach()

    @torch.no_grad()
    def export_weights(self):
        return {
            "W_enc": self.W_enc.detach().numpy().astype(np.float64),
            "b_enc": self.b_enc.detach().numpy().astype(np.float64),
            "W_dec": self.W_dec.detach().numpy().astype(np.float64),
            "b_dec": self.b_dec.detach().numpy().astype(np.float64),
            "d_dict": self.d_dict,
            "d_model": self.d_model,
        }


@torch.no_grad()
def collect_activations(model, n=20000, seed=7, layers="all"):
    """Residual-stream activations across all layers and positions."""
    rng = np.random.default_rng(seed)
    toks, _ = task.sample_batch(n, rng)
    acts = []
    for i in range(0, n, 2048):
        tr = model.residual_trace(torch.from_numpy(toks[i:i + 2048]))
        sel = tr if layers == "all" else [tr[l] for l in layers]
        acts.append(torch.cat([t.reshape(-1, t.shape[-1]) for t in sel], 0))
    return torch.cat(acts, 0)


def train_sae(acts, d_dict=64, steps=4000, batch=1024, lr=1e-3, l1=3e-3,
              seed=0, log_every=1000, verbose=True):
    torch.manual_seed(seed)
    sae = SAE(acts.shape[-1], d_dict, l1=l1)
    sae.normalize_decoder()
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    N = acts.shape[0]
    g = torch.Generator().manual_seed(seed)
    hist = []
    for step in range(steps):
        idx = torch.randint(0, N, (batch,), generator=g)
        loss, recon, l1v = sae.loss(acts[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        sae.normalize_decoder()
        if step % log_every == 0 or step == steps - 1:
            hist.append({"step": step, "loss": float(loss), "recon": float(recon),
                         "l1": float(l1v)})
            if verbose:
                print(f"    step {step:5d}  loss {float(loss):.4f}  "
                      f"recon {float(recon):.4f}  l1 {float(l1v):.3f}")
    return sae, hist


@torch.no_grad()
def bridge_gap(sae, acts, n=8192):
    """Quantify the SAE bridge: reconstruction error and sparsity.

    `rel_recon_err` is the number any reviewer will ask for. It is reported, not
    buried, and it is the coefficient on every feature-space claim that gets
    pushed back to activation space.
    """
    x = acts[:n]
    xh, f = sae(x)
    err = (xh - x).norm(dim=-1)
    nx = x.norm(dim=-1).clamp_min(1e-9)
    l0 = (f > 1e-6).float().sum(-1)
    return {
        "abs_recon_err_mean": float(err.mean()),
        "abs_recon_err_p99": float(err.quantile(0.99)),
        "rel_recon_err_mean": float((err / nx).mean()),
        "rel_recon_err_p99": float((err / nx).quantile(0.99)),
        "fvu": float(((xh - x) ** 2).sum() / ((x - x.mean(0)) ** 2).sum()),
        "l0_mean": float(l0.mean()),
        "l0_p99": float(l0.quantile(0.99)),
        "d_dict": int(sae.d_dict),
        "dead_features": int((f.max(0).values <= 1e-6).sum()),
    }


# ------------------------------------------------------------------ verification


def feature_displacement(z, sw, anchor):
    """Sound bounds on  df(x) = Enc(x) - Enc(x*)  for a hybrid zonotope x.

    Enc is affine-then-ReLU, so this is exact up to the ReLU relaxation, and the
    ReLU relaxation is ZERO on every feature that is provably on or provably off
    over the box. That is the sparsity dividend, made rigorous.
    """
    pre = z.linear(sw["W_enc"], sw["b_enc"] - sw["W_enc"] @ sw["b_dec"])
    f = bd.relu(pre)
    f_star = np.maximum((anchor - sw["b_dec"]) @ sw["W_enc"].T + sw["b_enc"], 0.0)
    return f - f_star, pre


def support_certificate(pre):
    """Partition features into provably-off / provably-on / undecided.

    Returns per-box counts. `undecided` is the EFFECTIVE DIMENSION of the
    verification problem: provably-off features contribute exactly zero and
    provably-on features are locally affine, so neither needs branching. This is
    the concrete mechanism by which the curse of dimensionality is dodged, and
    it either shows up in the numbers or the architectural bet is wrong.
    """
    lo, hi = pre.bounds()
    off = hi <= 0.0
    on = lo >= 0.0
    und = ~(off | on)
    axes = tuple(range(1, lo.ndim))
    return {
        "n_off": off.sum(axis=axes),
        "n_on": on.sum(axis=axes),
        "n_undecided": und.sum(axis=axes),
        "total": int(np.prod(lo.shape[1:])),
    }
