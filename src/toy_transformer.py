"""Toy pre-LN transformer viewed as a discrete-time nonlinear dynamical system.

    x_{l+1} = x_l + Attn_l(LN(x_l)) + MLP_l(LN(x_l + Attn_l(LN(x_l))))

The residual stream x_l in R^{SEQ_LEN x d_model} is the STATE and the layer
index l is TIME. Pre-LN is used deliberately: it keeps the residual stream an
unmodified additive accumulator, so the "+ x_l" identity path is exact and the
dynamical-system reading is literal rather than a metaphor.

Two deliberate architectural choices, both of which cost realism and buy
verifiability, and both of which are reported as such:

1. ReLU MLP rather than GELU. ReLU has an exact 3-piece convex relaxation
   (DeepZ) with zero relaxation error on stable neurons. GELU needs a tanh-style
   relaxation that leaks error on every neuron. Swapping back is supported.
2. No dropout / no biases inside attention. Keeps the block map deterministic
   and its Jacobian structure clean.

The numpy export (`export_weights`) is the single source of truth consumed by
the sound verifier -- the verifier never touches torch, so autograd can never
silently change what is being proved.
"""
from . import task  # noqa: F401  (env pinning happens in package __init__)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 32
N_HEADS = 4
D_HEAD = D_MODEL // N_HEADS
N_LAYERS = 2
D_MLP = 4 * D_MODEL
LN_EPS = 1e-5


class Attention(nn.Module):
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS):
        super().__init__()
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, return_pattern=False):
        B, T, D = x.shape
        H, dh = self.n_heads, self.d_head
        q = self.W_Q(x).view(B, T, H, dh).transpose(1, 2)
        k = self.W_K(x).view(B, T, H, dh).transpose(1, 2)
        v = self.W_V(x).view(B, T, H, dh).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / np.sqrt(dh)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        scores = scores.masked_fill(mask, -1e9)
        patt = scores.softmax(dim=-1)
        z = (patt @ v).transpose(1, 2).reshape(B, T, D)
        out = self.W_O(z)
        return (out, patt) if return_pattern else out


class MLP(nn.Module):
    def __init__(self, d_model=D_MODEL, d_mlp=D_MLP):
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_mlp)
        self.fc_out = nn.Linear(d_mlp, d_model)

    def forward(self, x):
        return self.fc_out(F.relu(self.fc_in(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.attn = Attention()
        self.ln2 = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ToyTransformer(nn.Module):
    def __init__(self, vocab=task.VOCAB, seq_len=task.SEQ_LEN, n_layers=N_LAYERS):
        super().__init__()
        self.embed = nn.Embedding(vocab, D_MODEL)
        self.pos = nn.Parameter(torch.randn(seq_len, D_MODEL) * 0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(D_MODEL, eps=LN_EPS)
        self.unembed = nn.Linear(D_MODEL, vocab, bias=False)
        self.n_layers = n_layers

    def residual_trace(self, toks):
        """List of length n_layers+1 of residual streams (B, T, D)."""
        x = self.embed(toks) + self.pos[None, : toks.shape[1]]
        trace = [x]
        for blk in self.blocks:
            x = blk(x)
            trace.append(x)
        return trace

    def forward(self, toks):
        x = self.residual_trace(toks)[-1]
        return self.unembed(self.ln_f(x))

    def logits_from_stream(self, x_final):
        """Readout applied to an arbitrary final-layer residual stream."""
        return self.unembed(self.ln_f(x_final))

    # ---------------------------------------------------------------- export

    @torch.no_grad()
    def export_weights(self):
        """Plain-numpy dict consumed by the sound verifier."""
        w = {
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "d_head": D_HEAD,
            "n_layers": self.n_layers,
            "seq_len": self.pos.shape[0],
            "ln_eps": LN_EPS,
            "embed": self.embed.weight.detach().numpy().astype(np.float64),
            "pos": self.pos.detach().numpy().astype(np.float64),
            "ln_f_g": self.ln_f.weight.detach().numpy().astype(np.float64),
            "ln_f_b": self.ln_f.bias.detach().numpy().astype(np.float64),
            "unembed": self.unembed.weight.detach().numpy().astype(np.float64),
            "blocks": [],
        }
        for blk in self.blocks:
            w["blocks"].append({
                "ln1_g": blk.ln1.weight.detach().numpy().astype(np.float64),
                "ln1_b": blk.ln1.bias.detach().numpy().astype(np.float64),
                "ln2_g": blk.ln2.weight.detach().numpy().astype(np.float64),
                "ln2_b": blk.ln2.bias.detach().numpy().astype(np.float64),
                "WQ": blk.attn.W_Q.weight.detach().numpy().astype(np.float64),
                "WK": blk.attn.W_K.weight.detach().numpy().astype(np.float64),
                "WV": blk.attn.W_V.weight.detach().numpy().astype(np.float64),
                "WO": blk.attn.W_O.weight.detach().numpy().astype(np.float64),
                "fc_in_W": blk.mlp.fc_in.weight.detach().numpy().astype(np.float64),
                "fc_in_b": blk.mlp.fc_in.bias.detach().numpy().astype(np.float64),
                "fc_out_W": blk.mlp.fc_out.weight.detach().numpy().astype(np.float64),
                "fc_out_b": blk.mlp.fc_out.bias.detach().numpy().astype(np.float64),
            })
        return w


def train(steps=3000, batch=256, lr=3e-3, seed=0, log_every=500, verbose=True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ToyTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    hist = []
    for step in range(steps):
        toks, tgt = task.sample_batch(batch, rng)
        toks_t = torch.from_numpy(toks)
        tgt_t = torch.from_numpy(tgt)
        logits = model(toks_t)[:, -1, :]
        loss = F.cross_entropy(logits, tgt_t)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % log_every == 0 or step == steps - 1:
            acc = (logits.argmax(-1) == tgt_t).float().mean().item()
            hist.append({"step": step, "loss": loss.item(), "acc": acc})
            if verbose:
                print(f"    step {step:5d}  loss {loss.item():.4f}  acc {acc:.4f}")
    return model, hist


@torch.no_grad()
def evaluate(model, n=4096, seed=123):
    rng = np.random.default_rng(seed)
    toks, tgt = task.sample_batch(n, rng)
    logits = model(torch.from_numpy(toks))[:, -1, :]
    acc = (logits.argmax(-1).numpy() == tgt).mean()
    margin = task.unsafe_margin(logits.numpy())
    toks_s, tgt_s = task.sample_batch(n, rng, safe_only=True)
    logits_s = model(torch.from_numpy(toks_s))[:, -1, :]
    acc_s = (logits_s.argmax(-1).numpy() == tgt_s).mean()
    margin_s = task.unsafe_margin(logits_s.numpy())
    return {
        "acc_all": float(acc),
        "acc_safe_prompts": float(acc_s),
        "unsafe_margin_mean_safe": float(margin_s.mean()),
        "unsafe_margin_max_safe": float(margin_s.max()),
        "unsafe_margin_frac_negative_safe": float((margin_s < 0).mean()),
        "unsafe_margin_mean_all": float(margin.mean()),
    }
