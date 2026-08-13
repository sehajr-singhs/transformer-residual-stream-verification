"""Reference interval-bound engine, written from the TORCH forward semantics.

This file deliberately shares no code with `src/bounds.py`. It is transcribed
from `toy_transformer.py`'s `forward` (the ground truth the prover is supposed
to over-approximate), not from the prover, so a mistake in the hybrid-zonotope
mechanics cannot be mirrored here by construction.

It is a plain interval (IBP) engine: strictly weaker than the zonotope, and it
loses the k-dimensional subspace structure of the threat model at the input.
It is used for two things only:

  1. an independent sound bound, wherever it is tight enough to close;
  2. a containment check -- the zonotope's box hull must lie inside this one at
     every layer, since both are sound enclosures of the same set and this one
     makes no structural assumptions. A containment failure is a red flag.

Everything here returns (lo, hi) float64 arrays with outward padding.
"""
import numpy as np

# Outward rounding slack applied after each nonlinearity. Default 0: this engine
# is sound modulo float64, the same standard src/bounds.py holds itself to.
# It is a knob rather than a constant because it doubles as a wrapping probe --
# PAD=1e-12 emerges at the logits as ~4.7e-4, an amplification of 4.7e8 through
# two blocks, which is an independent measurement of the wrapping factor that
# baseline.json reports as 4.6e7 for layer1 alone.
PAD = 0.0


def _out(lo, hi):
    return lo - PAD, hi + PAD


def iv_linear(lo, hi, W, b=None):
    """y = x @ W.T + b for x in [lo, hi]. Tightest interval enclosure."""
    c = 0.5 * (lo + hi)
    r = 0.5 * (hi - lo)
    yc = c @ W.T
    yr = r @ np.abs(W).T
    if b is not None:
        yc = yc + b
    return yc - yr, yc + yr


def iv_mul(alo, ahi, blo, bhi):
    """Elementwise interval product."""
    p = np.stack([alo * blo, alo * bhi, ahi * blo, ahi * bhi])
    return p.min(axis=0), p.max(axis=0)


def iv_matmul(alo, ahi, blo, bhi):
    """Interval matmul  A @ B  with both operands intervals.

    Uses the midpoint/radius identity: |A B - Ac Bc| <= |Ac| Br + Ar |Bc| + Ar Br.
    """
    ac, ar = 0.5 * (alo + ahi), 0.5 * (ahi - alo)
    bc, br = 0.5 * (blo + bhi), 0.5 * (bhi - blo)
    c = ac @ bc
    r = np.abs(ac) @ br + ar @ np.abs(bc) + ar @ br
    return c - r, c + r


def iv_div(nlo, nhi, dlo, dhi):
    """n / d with the denominator interval strictly positive."""
    assert (dlo > 0).all(), "denominator interval must be positive"
    cand = np.stack([nlo / dlo, nlo / dhi, nhi / dlo, nhi / dhi])
    return cand.min(axis=0), cand.max(axis=0)


def iv_square(lo, hi):
    s = np.stack([lo * lo, hi * hi])
    hi_ = s.max(axis=0)
    lo_ = np.where((lo <= 0) & (hi >= 0), 0.0, s.min(axis=0))
    return lo_, hi_


def layernorm(lo, hi, g, b, eps):
    """torch.nn.LayerNorm over the last axis: g*(x-mean)/sqrt(var+eps)+b.

    var is the BIASED variance (divide by D), matching torch.

    Two bounds are computed and intersected:

      (a) the interval-arithmetic bound, tight when the input box is small;

      (b) a STRUCTURAL bound that holds for any input whatsoever. Writing
          u = x - mean(x) and s = sqrt(var + eps), we have
          ||u/s||_2^2 = D*var/(var+eps) <= D, so every coordinate obeys
          |u_d/s| <= sqrt(D) unconditionally.

    (b) is what keeps this engine finite. IBP discards the 4-dimensional
    structure of the threat model at the input, so by layer 1 the interval
    bound (a) can overflow to +-inf; (b) then still returns a valid enclosure
    instead of a NaN. It is also the reason a blown-up IBP bound stays sound
    rather than becoming vacuous garbage.
    """
    D = lo.shape[-1]
    M = np.eye(D) - np.ones((D, D)) / D          # x -> x - mean(x)
    ulo, uhi = iv_linear(lo, hi, M)
    slo, shi = iv_square(ulo, uhi)
    vlo = slo.mean(axis=-1, keepdims=True)
    vhi = shi.mean(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", over="ignore"):
        dlo = np.sqrt(np.maximum(vlo, 0.0) + eps)
        dhi = np.sqrt(np.maximum(vhi, 0.0) + eps)
        rlo, rhi = iv_div(ulo, uhi, dlo, dhi)
    # structural fallback / intersection
    cap = np.sqrt(float(D))
    rlo = np.where(np.isfinite(rlo), np.maximum(rlo, -cap), -cap)
    rhi = np.where(np.isfinite(rhi), np.minimum(rhi, cap), cap)
    ylo, yhi = iv_mul(rlo, rhi, g, g)
    return _out(ylo + b, yhi + b)


def softmax_rows(sclo, schi, mask):
    """Sound elementwise enclosure of softmax over the last axis.

    For each output index s, the ratio is monotone increasing in score_s and
    decreasing in every other allowed score, so the extremes are attained at the
    corners of the score box:

        p_s >= exp(lo_s) / (exp(lo_s) + sum_{j != s} exp(hi_j))
        p_s <= exp(hi_s) / (exp(hi_s) + sum_{j != s} exp(lo_j))

    `mask` is True where attention is DISALLOWED (causal upper triangle).
    The simplex constraint sum_s p_s = 1 is NOT imposed -- an interval cannot
    carry it. That is one of the places this engine is loose by design.
    """
    A = ~mask
    # A row whose scores are not all finite has overflowed the interval
    # arithmetic. Fall back to the trivial enclosure p in [0, 1], which is
    # always sound, rather than producing inf - inf = NaN.
    ok = np.isfinite(sclo) & np.isfinite(schi) & A
    row_ok = (ok == A).all(axis=-1, keepdims=True)

    NEG = -np.inf
    lo = np.where(A, sclo, NEG)
    hi = np.where(A, schi, NEG)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        shift = np.max(np.where(A, hi, -np.inf), axis=-1, keepdims=True)
        shift = np.where(np.isfinite(shift), shift, 0.0)
        elo = np.where(A, np.exp(np.where(A, lo - shift, -np.inf)), 0.0)
        ehi = np.where(A, np.exp(np.where(A, hi - shift, -np.inf)), 0.0)
        sum_lo = elo.sum(axis=-1, keepdims=True)
        sum_hi = ehi.sum(axis=-1, keepdims=True)
        # denominator excluding s, evaluated at the opposite corner
        den_for_lo = elo + (sum_hi - ehi)
        den_for_hi = ehi + (sum_lo - elo)
        plo = np.where(den_for_lo > 0, elo / np.where(den_for_lo > 0, den_for_lo, 1.0), 0.0)
        phi = np.where(den_for_hi > 0, ehi / np.where(den_for_hi > 0, den_for_hi, 1.0), 1.0)

    plo = np.where(np.isfinite(plo), plo, 0.0)
    phi = np.where(np.isfinite(phi), phi, 1.0)
    plo = np.where(row_ok, plo, 0.0)
    phi = np.where(row_ok, phi, 1.0)
    plo = np.where(A, np.clip(plo, 0.0, 1.0), 0.0)
    phi = np.where(A, np.clip(phi, 0.0, 1.0), 0.0)
    return _out(plo, phi)


def attention(lo, hi, WQ, WK, WV, WO, n_heads):
    """Multi-head causal attention, transcribed from Attention.forward."""
    T, D = lo.shape
    H = n_heads
    dh = D // H
    qlo, qhi = iv_linear(lo, hi, WQ)
    klo, khi = iv_linear(lo, hi, WK)
    vlo, vhi = iv_linear(lo, hi, WV)
    scale = 1.0 / np.sqrt(dh)
    mask = np.triu(np.ones((T, T), dtype=bool), 1)
    zlo = np.zeros((T, D))
    zhi = np.zeros((T, D))
    for h in range(H):
        sl = slice(h * dh, (h + 1) * dh)
        qh_lo, qh_hi = qlo[:, sl], qhi[:, sl]
        kh_lo, kh_hi = klo[:, sl], khi[:, sl]
        vh_lo, vh_hi = vlo[:, sl], vhi[:, sl]
        sc_lo, sc_hi = iv_matmul(qh_lo, qh_hi, kh_lo.T, kh_hi.T)
        sc_lo, sc_hi = sc_lo * scale, sc_hi * scale
        p_lo, p_hi = softmax_rows(sc_lo, sc_hi, mask)
        o_lo, o_hi = iv_matmul(p_lo, p_hi, vh_lo, vh_hi)
        zlo[:, sl], zhi[:, sl] = o_lo, o_hi
    return iv_linear(zlo, zhi, WO)


def mlp(lo, hi, fc_in_W, fc_in_b, fc_out_W, fc_out_b):
    a_lo, a_hi = iv_linear(lo, hi, fc_in_W, fc_in_b)
    r_lo, r_hi = np.maximum(a_lo, 0.0), np.maximum(a_hi, 0.0)
    return iv_linear(r_lo, r_hi, fc_out_W, fc_out_b)


def block(lo, hi, bw, n_heads, eps):
    """x = x + attn(ln1(x));  x = x + mlp(ln2(x))."""
    n_lo, n_hi = layernorm(lo, hi, bw["ln1_g"], bw["ln1_b"], eps)
    a_lo, a_hi = attention(n_lo, n_hi, bw["WQ"], bw["WK"], bw["WV"], bw["WO"],
                           n_heads)
    lo, hi = lo + a_lo, hi + a_hi
    n_lo, n_hi = layernorm(lo, hi, bw["ln2_g"], bw["ln2_b"], eps)
    m_lo, m_hi = mlp(n_lo, n_hi, bw["fc_in_W"], bw["fc_in_b"],
                     bw["fc_out_W"], bw["fc_out_b"])
    return lo + m_lo, hi + m_hi


def propagate(lo, hi, w):
    trace = [(lo, hi)]
    for l in range(w["n_layers"]):
        lo, hi = block(lo, hi, w["blocks"][l], w["n_heads"], w["ln_eps"])
        trace.append((lo, hi))
    return lo, hi, trace


def readout(lo, hi, w):
    y_lo, y_hi = layernorm(lo, hi, w["ln_f_g"], w["ln_f_b"], w["ln_eps"])
    return iv_linear(y_lo, y_hi, w["unembed"])


def margin_upper(L_lo, L_hi, idx_unsafe, idx_safe, pos=-1):
    """Sound upper bound on s = max_u L_u - max_j L_j at position `pos`.

    Independently derived: for any fixed safe index j, s <= max_u (L_u - L_j),
    and each pair difference is bounded above by hi(L_u) - lo(L_j). The bound is
    valid for every j, so take the min over j. (Taking a max over j instead is
    the exact bug this audit is meant to be able to catch.)
    """
    hu = L_hi[pos, idx_unsafe]
    lj = L_lo[pos, idx_safe]
    pair = hu[:, None] - lj[None, :]
    return float(pair.max(axis=0).min())


def alpha_box_to_stream(a_lo, a_hi, U, x_nom):
    """e = sum_j alpha_j U_j on the alpha box -> interval box on the stream.

    This is where IBP structurally gives up the threat model: the k-dimensional
    zonotope is replaced by its box hull in T*d_model dimensions.
    """
    mid = 0.5 * (a_lo + a_hi)
    rad = 0.5 * (a_hi - a_lo)
    c = np.einsum("k,ktd->td", mid, U) + x_nom
    r = np.einsum("k,ktd->td", rad, np.abs(U))
    return c - r, c + r


def certify_margin(a_lo, a_hi, U, x_nom, w, idx_unsafe, idx_safe):
    lo, hi = alpha_box_to_stream(a_lo, a_hi, U, x_nom)
    lo, hi, _ = propagate(lo, hi, w)
    L_lo, L_hi = readout(lo, hi, w)
    return margin_upper(L_lo, L_hi, idx_unsafe, idx_safe)
