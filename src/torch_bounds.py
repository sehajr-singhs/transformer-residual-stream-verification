"""c25 -- differentiable interval bound propagation (IBP) in PyTorch.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This is a tensorised, batched, autograd-friendly transcription of
`audit/ibp_ref.py`. It computes PLAIN INTERVAL BOUNDS. It is deliberately NOT a
port of `src/bounds.py`.

That distinction is load-bearing and is the reason this file exists at all:

  * `src/bounds.py` is a HYBRID ZONOTOPE. It carries correlation in `G` and
    confines relaxation error to `E`. It is the prover that produces every
    certified radius this project reports.
  * IBP puts everything in the interval. It is strictly, and usually
    catastrophically, looser -- bounds.py's own docstring notes that plain
    interval arithmetic "dies to the wrapping effect within one transformer
    block".

So a bound computed here must NEVER be reported in the same column as a
certified radius from the zonotope prover. They are different numbers about the
same object. What IBP buys, and the only thing it buys, is that it is cheap and
differentiable, which is what makes certified TRAINING (Gowal et al. 2018,
Zhang et al. 2020, Shi et al. 2021) possible at all.

`src/bounds.py` is untouched by c25. Every result since c14 rests on that file
being unmodified, and nothing here imports it.

VALIDATION TARGET
-----------------
`audit/ibp_ref.py`, which computes the same mathematical object from the same
torch forward semantics. `audit/selftest_torch_bounds.py` checks agreement.
Validating against `src/bounds.py` instead would be a category error: the two
disagree by orders of magnitude *by construction*, and that disagreement is
correct behaviour.

CONVENTIONS
-----------
Tensors are (B, T, D) unless noted. `mask` is True where attention is
DISALLOWED, matching ibp_ref.softmax_rows and NOT src/bounds.causal_mask (which
uses the opposite polarity -- an easy and silent bug).
"""
import math

import torch

# Outward rounding slack after each nonlinearity. 0.0 matches ibp_ref.PAD, i.e.
# sound modulo float64. Raising it is the wrapping probe described there.
PAD = 0.0


def _out(lo, hi):
    if PAD == 0.0:
        return lo, hi
    return lo - PAD, hi + PAD


# --------------------------------------------------------------- interval ops

def iv_linear(lo, hi, W, b=None):
    """y = x @ W.T + b for x in [lo, hi]. Tightest interval enclosure."""
    c = 0.5 * (lo + hi)
    r = 0.5 * (hi - lo)
    yc = c @ W.transpose(-1, -2)
    yr = r @ W.abs().transpose(-1, -2)
    if b is not None:
        yc = yc + b
    return yc - yr, yc + yr


def iv_mul(alo, ahi, blo, bhi):
    """Elementwise interval product."""
    p = torch.stack([alo * blo, alo * bhi, ahi * blo, ahi * bhi])
    return p.min(dim=0).values, p.max(dim=0).values


def iv_matmul(alo, ahi, blo, bhi):
    """A @ B with BOTH operands intervals.

    Midpoint/radius identity: |AB - Ac Bc| <= |Ac| Br + Ar |Bc| + Ar Br.
    """
    ac, ar = 0.5 * (alo + ahi), 0.5 * (ahi - alo)
    bc, br = 0.5 * (blo + bhi), 0.5 * (bhi - blo)
    c = ac @ bc
    r = ac.abs() @ br + ar @ bc.abs() + ar @ br
    return c - r, c + r


def iv_div(nlo, nhi, dlo, dhi):
    """n / d, denominator interval strictly positive.

    ibp_ref asserts positivity. Here it is clamped instead: an assert on a live
    tensor would fire mid-training on a transient bad step and kill the run,
    and the clamp is sound because the true denominator sqrt(var+eps) >=
    sqrt(eps) > 0 always.
    """
    tiny = torch.finfo(dlo.dtype).tiny
    dlo = dlo.clamp_min(tiny)
    dhi = dhi.clamp_min(tiny)
    cand = torch.stack([nlo / dlo, nlo / dhi, nhi / dlo, nhi / dhi])
    return cand.min(dim=0).values, cand.max(dim=0).values


def iv_square(lo, hi):
    s = torch.stack([lo * lo, hi * hi])
    hi_ = s.max(dim=0).values
    lo_ = torch.where((lo <= 0) & (hi >= 0),
                      torch.zeros_like(lo), s.min(dim=0).values)
    return lo_, hi_


# ------------------------------------------------------------ normalisations

def layernorm(lo, hi, g, b, eps):
    """Standard LayerNorm bound. Transcribed from ibp_ref.layernorm.

    Intersects the interval-arithmetic bound with the STRUCTURAL bound
    |u_d / s| <= sqrt(D), which holds unconditionally because
    ||u/s||^2 = D*var/(var+eps) <= D. The structural term is what keeps this
    engine finite once the interval bound overflows, and it is why a blown-up
    IBP bound stays sound rather than becoming NaN.
    """
    D = lo.shape[-1]
    eye = torch.eye(D, dtype=lo.dtype, device=lo.device)
    M = eye - 1.0 / D
    ulo, uhi = iv_linear(lo, hi, M)
    slo, shi = iv_square(ulo, uhi)
    vlo = slo.mean(dim=-1, keepdim=True)
    vhi = shi.mean(dim=-1, keepdim=True)
    dlo = torch.sqrt(vlo.clamp_min(0.0) + eps)
    dhi = torch.sqrt(vhi.clamp_min(0.0) + eps)
    rlo, rhi = iv_div(ulo, uhi, dlo, dhi)

    cap = math.sqrt(float(D))
    rlo = torch.where(torch.isfinite(rlo), rlo.clamp_min(-cap),
                      torch.full_like(rlo, -cap))
    rhi = torch.where(torch.isfinite(rhi), rhi.clamp_max(cap),
                      torch.full_like(rhi, cap))
    ylo, yhi = iv_mul(rlo, rhi, g, g)
    return _out(ylo + b, yhi + b)


def fixnorm(lo, hi, g, b, scale):
    """c18's fixed-scale norm: y = g * (x - mean(x)) / scale + b, scale CONSTANT.

    Wholly affine, so it introduces zero relaxation error. That is the entire
    point of the variant, and it is why fixnorm certifies where standard does
    not. Implemented as one exact interval-linear map.
    """
    D = lo.shape[-1]
    eye = torch.eye(D, dtype=lo.dtype, device=lo.device)
    M = eye - 1.0 / D
    ulo, uhi = iv_linear(lo, hi, M)
    s = g / scale
    ylo, yhi = iv_mul(ulo, uhi, s, s)
    return _out(ylo + b, yhi + b)


def norm(lo, hi, g, b, eps, variant, scale=None, floor=None):
    if variant == "fixnorm":
        if scale is None:
            raise ValueError("fixnorm needs a calibrated scale")
        return fixnorm(lo, hi, g, b, scale)
    if variant == "capnorm":
        if floor is None:
            raise ValueError("capnorm needs a floor")
        return capnorm(lo, hi, g, b, eps, floor)
    return layernorm(lo, hi, g, b, eps)


# ------------------------------------------------------------------- softmax

def softmax_rows(sclo, schi, mask):
    """Sound elementwise enclosure of softmax over the last axis.

    p_s >= exp(lo_s) / (exp(lo_s) + sum_{j!=s} exp(hi_j))
    p_s <= exp(hi_s) / (exp(hi_s) + sum_{j!=s} exp(lo_j))

    `mask` is True where attention is DISALLOWED. The simplex constraint
    sum_s p_s = 1 is NOT imposed; an interval cannot carry it. This is one of
    the places the engine is loose by design, and one reason the zonotope beats
    it so heavily.
    """
    A = ~mask
    ok = torch.isfinite(sclo) & torch.isfinite(schi) & A
    row_ok = (ok == A).all(dim=-1, keepdim=True)

    neg_inf = torch.full_like(sclo, float("-inf"))
    lo = torch.where(A, sclo, neg_inf)
    hi = torch.where(A, schi, neg_inf)

    shift = torch.where(A, hi, neg_inf).max(dim=-1, keepdim=True).values
    shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))

    zero = torch.zeros_like(sclo)
    elo = torch.where(A, torch.exp(torch.where(A, lo - shift, neg_inf)), zero)
    ehi = torch.where(A, torch.exp(torch.where(A, hi - shift, neg_inf)), zero)
    sum_lo = elo.sum(dim=-1, keepdim=True)
    sum_hi = ehi.sum(dim=-1, keepdim=True)

    den_for_lo = elo + (sum_hi - ehi)
    den_for_hi = ehi + (sum_lo - elo)
    plo = torch.where(den_for_lo > 0,
                      elo / torch.where(den_for_lo > 0, den_for_lo,
                                        torch.ones_like(den_for_lo)), zero)
    phi = torch.where(den_for_hi > 0,
                      ehi / torch.where(den_for_hi > 0, den_for_hi,
                                        torch.ones_like(den_for_hi)),
                      torch.ones_like(ehi))

    plo = torch.where(torch.isfinite(plo), plo, zero)
    phi = torch.where(torch.isfinite(phi), phi, torch.ones_like(phi))
    plo = torch.where(row_ok, plo, zero)
    phi = torch.where(row_ok, phi, torch.ones_like(phi))
    plo = torch.where(A, plo.clamp(0.0, 1.0), zero)
    phi = torch.where(A, phi.clamp(0.0, 1.0), zero)
    return _out(plo, phi)


# ----------------------------------------------------------------- the block

def attention(lo, hi, WQ, WK, WV, WO, n_heads):
    """Multi-head causal attention. Batched form of ibp_ref.attention."""
    B, T, D = lo.shape
    H = n_heads
    dh = D // H
    qlo, qhi = iv_linear(lo, hi, WQ)
    klo, khi = iv_linear(lo, hi, WK)
    vlo, vhi = iv_linear(lo, hi, WV)

    def heads(x):
        return x.view(B, T, H, dh).transpose(1, 2)  # (B, H, T, dh)

    qlo, qhi = heads(qlo), heads(qhi)
    klo, khi = heads(klo), heads(khi)
    vlo, vhi = heads(vlo), heads(vhi)

    scale = 1.0 / math.sqrt(dh)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=lo.device), 1)

    sc_lo, sc_hi = iv_matmul(qlo, qhi,
                             klo.transpose(-1, -2), khi.transpose(-1, -2))
    sc_lo, sc_hi = sc_lo * scale, sc_hi * scale
    p_lo, p_hi = softmax_rows(sc_lo, sc_hi, mask)
    o_lo, o_hi = iv_matmul(p_lo, p_hi, vlo, vhi)

    def merge(x):
        return x.transpose(1, 2).reshape(B, T, D)

    return iv_linear(merge(o_lo), merge(o_hi), WO)


# ------------------------------------------------- c26 alternative primitives
# Smooth activations to ablate against ReLU. Both are NON-MONOTONE: they dip
# below zero and then recover, so the naive [f(lo), f(hi)] endpoint bound is
# UNSOUND on any interval straddling the minimum. Each carries its interior
# minimum explicitly. Constants are the exact stationary points, found by
# solving f'(x)=0 numerically once and hard-coded so the bound is deterministic.

# Located by golden-section search on the (unimodal) dip, then rounded DOWN in
# the last digit. Rounding down is the sound direction: this value is used only
# as a LOWER bound, so erring low can only widen the box, never invalidate it.
GELU_ARGMIN = -0.751791524668        # d/dx gelu = 0
GELU_MIN = -0.169971207481
SILU_ARGMIN = -1.278464540467        # d/dx silu = 0
SILU_MIN = -0.278464542762


def _smooth_iv(lo, hi, f, argmin, fmin):
    """Sound interval image of a function decreasing then increasing.

    Max is at an endpoint. Min is at an endpoint UNLESS the interval contains
    the interior minimiser, in which case it is the interior minimum.
    """
    a, b = f(lo), f(hi)
    out_hi = torch.maximum(a, b)
    out_lo = torch.minimum(a, b)
    inside = (lo <= argmin) & (hi >= argmin)
    out_lo = torch.where(inside, torch.full_like(out_lo, fmin), out_lo)
    return out_lo, out_hi


def gelu_iv(lo, hi):
    return _smooth_iv(lo, hi, lambda t: F_gelu(t), GELU_ARGMIN, GELU_MIN)


def silu_iv(lo, hi):
    return _smooth_iv(lo, hi, lambda t: t * torch.sigmoid(t),
                      SILU_ARGMIN, SILU_MIN)


def F_gelu(t):
    """Exact (erf) GeLU, matching torch.nn.functional.gelu default."""
    return 0.5 * t * (1.0 + torch.erf(t / math.sqrt(2.0)))


ACTS = {"relu": lambda l, h: (l.clamp_min(0.0), h.clamp_min(0.0)),
        "gelu": gelu_iv,
        "silu": silu_iv}


def capnorm(lo, hi, g, b, eps, floor):
    """Lipschitz-capped normaliser: y = g * (x - mean) / max(rms, floor) + b.

    LayerNorm's gain 1/sqrt(var+eps) is unbounded as var -> 0, which is exactly
    what makes its bracket explode (c14/c18). Flooring the denominator caps the
    gain at 1/floor, so the map is globally Lipschitz with constant
    max|g|/floor. Unlike fixnorm it remains data-dependent, so it is the
    intermediate point between LayerNorm and a frozen constant.
    """
    D = lo.shape[-1]
    eye = torch.eye(D, dtype=lo.dtype, device=lo.device)
    M = eye - 1.0 / D
    ulo, uhi = iv_linear(lo, hi, M)
    slo, shi = iv_square(ulo, uhi)
    rms_lo = torch.sqrt(slo.mean(dim=-1, keepdim=True).clamp_min(0.0) + eps)
    rms_hi = torch.sqrt(shi.mean(dim=-1, keepdim=True).clamp_min(0.0) + eps)
    dlo = rms_lo.clamp_min(floor)
    dhi = rms_hi.clamp_min(floor)
    rlo, rhi = iv_div(ulo, uhi, dlo, dhi)
    # The Lipschitz cap is also a sound structural bound: |u_d| <= (hi-lo) sum,
    # and dividing by >= floor cannot amplify beyond 1/floor.
    cap = math.sqrt(float(D))
    rlo = torch.where(torch.isfinite(rlo), rlo.clamp_min(-cap),
                      torch.full_like(rlo, -cap))
    rhi = torch.where(torch.isfinite(rhi), rhi.clamp_max(cap),
                      torch.full_like(rhi, cap))
    ylo, yhi = iv_mul(rlo, rhi, g, g)
    return _out(ylo + b, yhi + b)


def mlp(lo, hi, fc_in_W, fc_in_b, fc_out_W, fc_out_b, act="relu"):
    a_lo, a_hi = iv_linear(lo, hi, fc_in_W, fc_in_b)
    r_lo, r_hi = ACTS[act](a_lo, a_hi)
    return iv_linear(r_lo, r_hi, fc_out_W, fc_out_b)


def mlp_gated(lo, hi, W_gate, b_gate, W_up, b_up, W_down, b_down, act="silu"):
    """SwiGLU-style gated MLP: down( act(x@Wg) * (x@Wu) ).

    The elementwise product of two INTERVALS is where this loses tightness
    relative to a plain MLP -- iv_mul takes the corner extremes and cannot see
    that the two factors are correlated through x. That looseness is intrinsic
    to interval arithmetic, not a bug, and is one more reason the training-time
    bound must not be confused with the zonotope certificate.
    """
    g_lo, g_hi = iv_linear(lo, hi, W_gate, b_gate)
    g_lo, g_hi = ACTS[act](g_lo, g_hi)
    u_lo, u_hi = iv_linear(lo, hi, W_up, b_up)
    h_lo, h_hi = iv_mul(g_lo, g_hi, u_lo, u_hi)
    return iv_linear(h_lo, h_hi, W_down, b_down)


def block(lo, hi, bw, n_heads, eps, variant="standard", act="relu",
          mlp_kind="standard", floor=None):
    """x = x + attn(n1(x));  x = x + mlp(n2(x))."""
    n_lo, n_hi = norm(lo, hi, bw["ln1_g"], bw["ln1_b"], eps, variant,
                      bw.get("scale1"), floor)
    a_lo, a_hi = attention(n_lo, n_hi, bw["WQ"], bw["WK"], bw["WV"], bw["WO"],
                           n_heads)
    lo, hi = lo + a_lo, hi + a_hi
    n_lo, n_hi = norm(lo, hi, bw["ln2_g"], bw["ln2_b"], eps, variant,
                      bw.get("scale2"), floor)
    if mlp_kind == "swiglu":
        m_lo, m_hi = mlp_gated(n_lo, n_hi, bw["W_gate"], bw["b_gate"],
                               bw["W_up"], bw["b_up"], bw["W_down"],
                               bw["b_down"], act)
    else:
        m_lo, m_hi = mlp(n_lo, n_hi, bw["fc_in_W"], bw["fc_in_b"],
                         bw["fc_out_W"], bw["fc_out_b"], act)
    return lo + m_lo, hi + m_hi


def propagate(lo, hi, w, variant="standard", act="relu", mlp_kind="standard",
              floor=None):
    for l in range(w["n_layers"]):
        lo, hi = block(lo, hi, w["blocks"][l], w["n_heads"], w["ln_eps"],
                       variant, act, mlp_kind, floor)
    return lo, hi


def readout(lo, hi, w, variant="standard", floor=None):
    y_lo, y_hi = norm(lo, hi, w["ln_f_g"], w["ln_f_b"], w["ln_eps"], variant,
                      w.get("scale_f"), floor)
    return iv_linear(y_lo, y_hi, w["unembed"])


# ------------------------------------------------------- certified objective

def worst_case_logits(L_lo, L_hi, target):
    """Gowal et al. 2018 elision: the logit vector that maximises every
    competing margin at once.

    For the true class take the LOWER bound, for every other class the UPPER
    bound. Cross-entropy on this vector upper-bounds the worst-case loss over
    the whole input box, so minimising it minimises a sound surrogate.

    L_lo, L_hi: (N, V).  target: (N,) int64.  Returns (N, V).
    """
    z = L_hi.clone()
    idx = torch.arange(L_lo.shape[0], device=L_lo.device)
    z[idx, target] = L_lo[idx, target]
    return z


def certified_margin_lower(L_lo, L_hi, idx_unsafe, idx_safe, pos=-1):
    """Lower bound on (safe logit - unsafe logit), i.e. the safety margin.

    Worst case is the safe logit at its lower bound and the unsafe one at its
    upper bound. POSITIVE means certified safe over the entire input box.

    Sign convention is the mirror of ibp_ref.margin_upper, which bounds
    (unsafe - safe) from ABOVE. Same statement, opposite sign; keep them
    straight when comparing against banked numbers.
    """
    lo = L_lo[:, pos, :] if L_lo.dim() == 3 else L_lo
    hi = L_hi[:, pos, :] if L_hi.dim() == 3 else L_hi
    return lo[:, idx_safe] - hi[:, idx_unsafe]
