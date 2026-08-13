"""Independent sound bound #2: mean-value form with interval forward-mode AD.

Why a second engine. `ibp_ref` is independent but replaces the k-dimensional
threat set by its box hull in T*d_model dimensions at the very first step, so it
closes nothing at any radius the prover certifies. An audit that can only say
"my witness is too weak" has not cross-checked anything.

This engine keeps the k-dimensional structure and is still not a zonotope. It
bounds the margin by the mean-value inequality: for the scalar map
g(alpha) = L_u(alpha) - L_j(alpha) on a box with centre a_c and radii r,

    g(alpha) <= g(a_c) + sum_i r_i * max |dg/dalpha_i| over the box

where the derivative enclosure is obtained by pushing (value, derivative)
interval pairs forward through the network -- interval forward-mode automatic
differentiation. ReLU is nonsmooth, so the derivative enclosure at a kink is the
convex hull {0,1}*dx, which makes the inequality a Clarke mean-value inequality;
it remains valid.

Structurally this is a different animal from the prover: the prover propagates
an affine form of the SET and reads off a support function; this propagates a
POINT plus an enclosure of the derivative and applies a first-order remainder
argument. They share the network semantics and nothing else. The centred form
is also quadratically convergent under bisection, which is why it can close
boxes that IBP cannot.

All functions take/return (lo, hi) pairs. Derivative arrays carry a leading axis
of size K (one per alpha coordinate).
"""
import numpy as np

from ibp_ref import (iv_linear, iv_mul, iv_matmul, iv_div, iv_square)


def _lin_vd(v, d, W, b=None):
    """Affine map applied to a (value, derivative) interval pair."""
    vlo, vhi = iv_linear(v[0], v[1], W, b)
    dlo, dhi = iv_linear(d[0], d[1], W, None)     # bias drops out of d/dalpha
    return (vlo, vhi), (dlo, dhi)


def _relu_vd(v, d):
    """ReLU with a Clarke-generalized derivative enclosure at unstable units."""
    vlo, vhi = v
    dlo, dhi = d
    out_v = (np.maximum(vlo, 0.0), np.maximum(vhi, 0.0))
    pos = (vlo > 0)[None]
    neg = (vhi <= 0)[None]
    # unstable: derivative multiplier lies in [0,1], so d passes to hull(0, d)
    ulo = np.minimum(dlo, 0.0)
    uhi = np.maximum(dhi, 0.0)
    nlo = np.where(pos, dlo, np.where(neg, 0.0, ulo))
    nhi = np.where(pos, dhi, np.where(neg, 0.0, uhi))
    return out_v, (nlo, nhi)


def _layernorm_vd(v, d, g, b, eps):
    """LayerNorm and its derivative.

    y = g * u/s + b,  u = x - mean(x),  s = sqrt(var + eps),  var = mean(u^2)
    dy = g * ( du/s - (u/s) * (ds/s) ),   ds = dvar / (2 s),
    dvar = (2/D) * sum_d u_d du_d
    """
    vlo, vhi = v
    dlo, dhi = d
    D = vlo.shape[-1]
    M = np.eye(D) - np.ones((D, D)) / D
    ulo, uhi = iv_linear(vlo, vhi, M)
    dulo, duhi = iv_linear(dlo, dhi, M)

    slo_, shi_ = iv_square(ulo, uhi)
    varlo = slo_.mean(axis=-1, keepdims=True)
    varhi = shi_.mean(axis=-1, keepdims=True)
    s_lo = np.sqrt(np.maximum(varlo, 0.0) + eps)
    s_hi = np.sqrt(np.maximum(varhi, 0.0) + eps)

    # r = u / s, intersected with the unconditional |r| <= sqrt(D)
    with np.errstate(invalid="ignore", over="ignore"):
        rlo, rhi = iv_div(ulo, uhi, s_lo, s_hi)
    cap = np.sqrt(float(D))
    rlo = np.where(np.isfinite(rlo), np.maximum(rlo, -cap), -cap)
    rhi = np.where(np.isfinite(rhi), np.minimum(rhi, cap), cap)

    # dvar = (2/D) sum_d u_d du_d
    plo, phi = iv_mul(ulo[None], uhi[None], dulo, duhi)
    dvarlo = (2.0 / D) * plo.sum(axis=-1, keepdims=True)
    dvarhi = (2.0 / D) * phi.sum(axis=-1, keepdims=True)
    # ds = dvar / (2 s)
    dslo, dshi = iv_div(dvarlo, dvarhi, 2.0 * s_lo, 2.0 * s_hi)
    # du/s
    a_lo, a_hi = iv_div(dulo, duhi, s_lo, s_hi)
    # (u/s) * (ds/s)
    q_lo, q_hi = iv_div(dslo, dshi, s_lo, s_hi)
    c_lo, c_hi = iv_mul(rlo[None], rhi[None], q_lo, q_hi)
    dy_lo, dy_hi = a_lo - c_hi, a_hi - c_lo

    y_lo, y_hi = iv_mul(rlo, rhi, g, g)
    dy_lo, dy_hi = iv_mul(dy_lo, dy_hi, g, g)
    return (y_lo + b, y_hi + b), (dy_lo, dy_hi)


def _softmax_vd(sc, dsc, mask):
    """Softmax and its derivative.

    p = softmax(sc);  dp_s = p_s * ( dsc_s - sum_j p_j dsc_j )
    The p enclosure reuses the monotone corner bound from ibp_ref.
    """
    from ibp_ref import softmax_rows
    p_lo, p_hi = softmax_rows(sc[0], sc[1], mask)
    A = ~mask
    d_lo = np.where(A[None], dsc[0], 0.0)
    d_hi = np.where(A[None], dsc[1], 0.0)
    # w = sum_j p_j dsc_j
    wlo, whi = iv_mul(p_lo[None], p_hi[None], d_lo, d_hi)
    wlo = wlo.sum(axis=-1, keepdims=True)
    whi = whi.sum(axis=-1, keepdims=True)
    inner_lo = d_lo - whi
    inner_hi = d_hi - wlo
    dp_lo, dp_hi = iv_mul(p_lo[None], p_hi[None], inner_lo, inner_hi)
    dp_lo = np.where(A[None], dp_lo, 0.0)
    dp_hi = np.where(A[None], dp_hi, 0.0)
    return (p_lo, p_hi), (dp_lo, dp_hi)


def _attention_vd(v, d, WQ, WK, WV, WO, n_heads):
    T, D = v[0].shape
    H, dh = n_heads, D // n_heads
    q, dq = _lin_vd(v, d, WQ)
    k, dk = _lin_vd(v, d, WK)
    val, dval = _lin_vd(v, d, WV)
    scale = 1.0 / np.sqrt(dh)
    mask = np.triu(np.ones((T, T), dtype=bool), 1)
    K = d[0].shape[0]
    z_lo = np.zeros((T, D)); z_hi = np.zeros((T, D))
    dz_lo = np.zeros((K, T, D)); dz_hi = np.zeros((K, T, D))
    for h in range(H):
        sl = slice(h * dh, (h + 1) * dh)
        qh = (q[0][:, sl], q[1][:, sl]); dqh = (dq[0][:, :, sl], dq[1][:, :, sl])
        kh = (k[0][:, sl], k[1][:, sl]); dkh = (dk[0][:, :, sl], dk[1][:, :, sl])
        vh = (val[0][:, sl], val[1][:, sl]); dvh = (dval[0][:, :, sl], dval[1][:, :, sl])

        sc_lo, sc_hi = iv_matmul(qh[0], qh[1], kh[0].T, kh[1].T)
        sc = (sc_lo * scale, sc_hi * scale)
        # d(q k^T) = dq k^T + q dk^T
        t1 = iv_matmul(dqh[0], dqh[1], kh[0].T[None], kh[1].T[None])
        t2 = iv_matmul(qh[0][None], qh[1][None], np.swapaxes(dkh[0], -1, -2),
                       np.swapaxes(dkh[1], -1, -2))
        dsc = ((t1[0] + t2[0]) * scale, (t1[1] + t2[1]) * scale)

        p, dp = _softmax_vd(sc, dsc, mask)
        o_lo, o_hi = iv_matmul(p[0], p[1], vh[0], vh[1])
        # d(p v) = dp v + p dv
        u1 = iv_matmul(dp[0], dp[1], vh[0][None], vh[1][None])
        u2 = iv_matmul(p[0][None], p[1][None], dvh[0], dvh[1])
        z_lo[:, sl], z_hi[:, sl] = o_lo, o_hi
        dz_lo[:, :, sl] = u1[0] + u2[0]
        dz_hi[:, :, sl] = u1[1] + u2[1]
    return _lin_vd((z_lo, z_hi), (dz_lo, dz_hi), WO)


def _mlp_vd(v, d, fc_in_W, fc_in_b, fc_out_W, fc_out_b):
    a, da = _lin_vd(v, d, fc_in_W, fc_in_b)
    r, dr = _relu_vd(a, da)
    return _lin_vd(r, dr, fc_out_W, fc_out_b)


def _block_vd(v, d, bw, n_heads, eps):
    n, dn = _layernorm_vd(v, d, bw["ln1_g"], bw["ln1_b"], eps)
    a, da = _attention_vd(n, dn, bw["WQ"], bw["WK"], bw["WV"], bw["WO"], n_heads)
    v = (v[0] + a[0], v[1] + a[1]); d = (d[0] + da[0], d[1] + da[1])
    n, dn = _layernorm_vd(v, d, bw["ln2_g"], bw["ln2_b"], eps)
    m, dm = _mlp_vd(n, dn, bw["fc_in_W"], bw["fc_in_b"], bw["fc_out_W"], bw["fc_out_b"])
    return (v[0] + m[0], v[1] + m[1]), (d[0] + dm[0], d[1] + dm[1])


def jacobian_enclosure(a_lo, a_hi, U, x_nom, w):
    """Enclose d(logits)/d(alpha) over the alpha box.

    The input map is exactly linear: x0 = x_nom + sum_k alpha_k U_k, so the seed
    derivative is the constant U itself and carries no interval width.
    """
    K = U.shape[0]
    mid = 0.5 * (a_lo + a_hi); rad = 0.5 * (a_hi - a_lo)
    c = np.einsum("k,ktd->td", mid, U) + x_nom
    r = np.einsum("k,ktd->td", rad, np.abs(U))
    v = (c - r, c + r)
    d = (U.copy(), U.copy())                    # exact, width zero
    for l in range(w["n_layers"]):
        v, d = _block_vd(v, d, w["blocks"][l], w["n_heads"], w["ln_eps"])
    v, d = _layernorm_vd(v, d, w["ln_f_g"], w["ln_f_b"], w["ln_eps"])
    v, d = _lin_vd(v, d, w["unembed"])
    return v, d


def _exact_logits(alpha, U, x_nom, w):
    """Point evaluation of the logits, in pure float64, via degenerate intervals."""
    import ibp_ref as ref
    x = np.einsum("k,ktd->td", alpha, U) + x_nom
    lo, hi, _ = ref.propagate(x.copy(), x.copy(), w)
    L_lo, L_hi = ref.readout(lo, hi, w)
    return 0.5 * (L_lo + L_hi)


def certify_margin(a_lo, a_hi, U, x_nom, w, idx_unsafe, idx_safe, pos=-1):
    """Sound upper bound on the unsafe-logit margin by the mean-value form."""
    mid = 0.5 * (a_lo + a_hi)
    rad = 0.5 * (a_hi - a_lo)
    Lc = _exact_logits(mid, U, x_nom, w)[pos]                  # (VOCAB,)
    _, (dlo, dhi) = jacobian_enclosure(a_lo, a_hi, U, x_nom, w)
    J_lo = dlo[:, pos, :]                                      # (K, VOCAB)
    J_hi = dhi[:, pos, :]

    cu = Lc[idx_unsafe][:, None]                               # (|U|, 1)
    cs = Lc[idx_safe][None, :]                                 # (1, |S|)
    # derivative of the pair difference L_u - L_j
    du_lo = J_lo[:, idx_unsafe][:, :, None]; du_hi = J_hi[:, idx_unsafe][:, :, None]
    dj_lo = J_lo[:, idx_safe][:, None, :];   dj_hi = J_hi[:, idx_safe][:, None, :]
    dp_lo = du_lo - dj_hi
    dp_hi = du_hi - dj_lo
    slack = (rad[:, None, None] * np.maximum(np.abs(dp_lo), np.abs(dp_hi))).sum(axis=0)
    pair_hi = (cu - cs) + slack
    return float(pair_hi.max(axis=0).min())
