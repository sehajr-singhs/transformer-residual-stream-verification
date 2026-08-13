"""c26 -- swappable architecture primitives for the c25 certified-training grid.

Exists as a separate module because `c24_scaling.py` is imported by every c25
run and must stay byte-identical: its training fingerprint d22561792a22e50a
keys the c24 cell cache, and editing it would invalidate 15 banked cells.

WHAT CAN BE CERTIFIED, AND WHAT CANNOT
--------------------------------------
The post-hoc certificate comes from the NumPy hybrid-zonotope prover, which
implements ReLU and LayerNorm (`src/bounds.py`) plus fixnorm and a tanh secant
(`audit/c18_variants.py`). It does NOT implement GeLU, SiLU, a gated MLP, or a
floored normaliser. Those variants can be TRAINED with a sound interval bound
(`src/torch_bounds.py` has relaxations for all of them, validated by sampling)
but they cannot yet be certified.

`certifiable()` is the guard. A variant outside the prover's repertoire must be
recorded as uncertified with a reason -- never silently handed to the prover,
which would either crash or, worse, quietly certify the wrong function.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-5

NORMS = ("standard", "fixnorm", "capnorm")
ACTS = ("relu", "gelu", "silu")
MLPS = ("standard", "swiglu")

# (norm, act, mlp) combinations the NumPy prover can actually verify.
PROVER_NORMS = ("standard", "fixnorm")
PROVER_ACTS = ("relu",)
PROVER_MLPS = ("standard",)


def certifiable(norm, act, mlp):
    """-> (bool, reason). Reason is None when certification is supported."""
    bad = []
    if norm not in PROVER_NORMS:
        bad.append(f"norm={norm}")
    if act not in PROVER_ACTS:
        bad.append(f"act={act}")
    if mlp not in PROVER_MLPS:
        bad.append(f"mlp={mlp}")
    if bad:
        return False, ("no sound relaxation in src/bounds.py for "
                       + ", ".join(bad))
    return True, None


class Norm(nn.Module):
    """standard | fixnorm | capnorm.

    capnorm floors the RMS denominator, which caps the gain at 1/floor and
    makes the map globally Lipschitz with constant max|g|/floor. It is the
    intermediate point between LayerNorm (unbounded gain as var -> 0, the thing
    that makes the bracket explode) and fixnorm (gain frozen at a constant).
    """

    def __init__(self, d, mode, floor=0.5):
        super().__init__()
        self.mode, self.floor = mode, floor
        self.g = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))
        self.register_buffer("scale", torch.ones(1))
        self.calibrating = False

    def forward(self, x):
        xc = x - x.mean(-1, keepdim=True)
        rms = torch.sqrt(xc.pow(2).mean(-1, keepdim=True) + EPS)
        if self.calibrating:
            with torch.no_grad():
                self.scale.fill_(float(rms.mean()))
        if self.mode == "standard":
            s = rms
        elif self.mode == "fixnorm":
            s = self.scale
        else:
            s = rms.clamp_min(self.floor)
        return xc / s * self.g + self.b


def act_fn(name):
    return {"relu": F.relu, "gelu": F.gelu,
            "silu": F.silu}[name]


class Block(nn.Module):
    def __init__(self, d, n_heads, norm, act, mlp, floor, seq_len):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads
        self.act, self.mlp_kind = act, mlp
        self.n1 = Norm(d, norm, floor)
        self.n2 = Norm(d, norm, floor)
        self.WQ = nn.Linear(d, d, bias=False)
        self.WK = nn.Linear(d, d, bias=False)
        self.WV = nn.Linear(d, d, bias=False)
        self.WO = nn.Linear(d, d, bias=False)
        if mlp == "swiglu":
            # 4d/1.5 keeps the parameter count comparable to the 4d ReLU MLP,
            # since SwiGLU carries a third projection.
            hdim = int(round(4 * d * 2 / 3))
            self.fc_gate = nn.Linear(d, hdim)
            self.fc_up = nn.Linear(d, hdim)
            self.fc_down = nn.Linear(hdim, d)
        else:
            self.fc_in = nn.Linear(d, 4 * d)
            self.fc_out = nn.Linear(4 * d, d)

    def attn(self, x):
        B, T, D = x.shape
        q = self.WQ(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.WK(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.WV(x).view(B, T, self.h, self.dh).transpose(1, 2)
        sc = (q @ k.transpose(-1, -2)) / np.sqrt(self.dh)
        # Causal mask, load-bearing. True here means DISALLOWED, matching
        # src/torch_bounds.softmax_rows. src/bounds.causal_mask uses the
        # OPPOSITE polarity; swapping them is a silent wrong-answer bug.
        m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        sc = sc.masked_fill(m, -1e9).softmax(-1)
        return self.WO((sc @ v).transpose(1, 2).reshape(B, T, D))

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        n = self.n2(x)
        if self.mlp_kind == "swiglu":
            h = act_fn(self.act)(self.fc_gate(n)) * self.fc_up(n)
            return x + self.fc_down(h)
        return x + self.fc_out(act_fn(self.act)(self.fc_in(n)))


class ArchLM(nn.Module):
    def __init__(self, d, vocab, seq_len, n_heads=4, n_layers=2,
                 norm="standard", act="relu", mlp="standard", floor=0.5):
        super().__init__()
        self.n_layers, self.seq_len = n_layers, seq_len
        self.norm, self.act, self.mlp_kind = norm, act, mlp
        self.variant = f"{norm}-{act}-{mlp}"
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(seq_len, d) * 0.02)
        self.blocks = nn.ModuleList(
            [Block(d, n_heads, norm, act, mlp, floor, seq_len)
             for _ in range(n_layers)])
        self.nf = Norm(d, norm, floor)
        self.unembed = nn.Linear(d, vocab, bias=False)

    def embed_stream(self, toks):
        return self.embed(toks) + self.pos[None, :toks.shape[1]]

    def forward(self, toks):
        x = self.embed_stream(toks)
        for b in self.blocks:
            x = b(x)
        return self.unembed(self.nf(x))

    def set_calibrating(self, f):
        for m in self.modules():
            if isinstance(m, Norm):
                m.calibrating = f
