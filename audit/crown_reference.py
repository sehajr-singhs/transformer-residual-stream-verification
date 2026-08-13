"""Independent sound bound #3: linear relaxation (CROWN/DeepPoly-style).

c14 established that the two existing reference engines are vacuous because they
DECORRELATE: they replace the 4-dimensional threat set by a box in T*d_model
dimensions at the input and then amplify by 4.7e8 through two blocks. The fix is
not a tighter interval engine, it is an engine that never throws the structure
away.

This module keeps every intermediate quantity as a pair of LINEAR functions of
the steering coefficients alpha:

    A_lo . alpha + b_lo  <=  z  <=  A_hi . alpha + b_hi        (elementwise)

with `A` carrying exactly K=4 coefficients. Concrete bounds are read off only
when a nonlinearity needs them, and the linear form is immediately re-established
afterwards. That is what preserves head-to-head and layer-to-layer correlation.

Independence from `src/bounds.py`. Both engines represent state as an affine
function of the same 4 symbols -- that much is forced by the problem, and c14
showed anything else is useless here. Everything downstream differs:

  - direction: the zonotope propagates a SET forward and reads a support
    function; this propagates a pair of one-sided linear BOUNDS and concretizes
    against the alpha box.
  - ReLU: DeepZ's single-slope form with the offset in an interval remainder E,
    versus CROWN's two-sided envelope with an adaptive lower slope and no
    remainder term at all.
  - LayerNorm: the zonotope multiplies the whole centred form by a scalar
    interval and pushes the spread into E; here the product is relaxed by a
    chord (secant) envelope that stays linear in alpha.
  - attention: the zonotope linearizes softmax about the centre with a Jacobian
    bracket; here the scores use McCormick envelopes and the probabilities enter
    as interval coefficients on a form that is still linear in V.
  - there is no promotion, no compaction, and no generator-alignment invariant,
    which is where two of the engine's real bugs lived.

So an agreeing bound is genuine corroboration, not a mirrored implementation.

Semantics are transcribed from `toy_transformer.py:forward`, never from the
prover.
"""
import numpy as np

# "linear"   -- Shi et al. style softmax relaxation (exp secant/tangent,
#               reciprocal secant/tangent, McCormick product), so the attention
#               probabilities stay linear in alpha.
# "interval" -- ablation: probabilities carry no alpha-dependence.
SOFTMAX_MODE = "linear"


# --------------------------------------------------------------- linear bounds

class LB:
    """z bounded by two affine functions of alpha. A: (K,)+S, b: S."""

    __slots__ = ("A_lo", "b_lo", "A_hi", "b_hi")

    def __init__(self, A_lo, b_lo, A_hi, b_hi):
        self.A_lo, self.b_lo = A_lo, b_lo
        self.A_hi, self.b_hi = A_hi, b_hi

    @staticmethod
    def exact(A, b):
        return LB(A.copy(), b.copy(), A.copy(), b.copy())

    @property
    def K(self):
        return self.A_lo.shape[0]

    def concretize(self, mid, rad):
        """Tightest interval over the alpha box."""
        m = mid.reshape((-1,) + (1,) * (self.A_lo.ndim - 1))
        r = rad.reshape((-1,) + (1,) * (self.A_lo.ndim - 1))
        lo = self.b_lo + (self.A_lo * m).sum(0) - (np.abs(self.A_lo) * r).sum(0)
        hi = self.b_hi + (self.A_hi * m).sum(0) + (np.abs(self.A_hi) * r).sum(0)
        return lo, hi

    def add(self, other):
        return LB(self.A_lo + other.A_lo, self.b_lo + other.b_lo,
                  self.A_hi + other.A_hi, self.b_hi + other.b_hi)


def lin(L, W, b=None):
    """y = z @ W.T + b, exact composition of the affine bounds."""
    Wp, Wn = np.maximum(W, 0.0), np.minimum(W, 0.0)
    A_lo = L.A_lo @ Wp.T + L.A_hi @ Wn.T
    A_hi = L.A_hi @ Wp.T + L.A_lo @ Wn.T
    b_lo = L.b_lo @ Wp.T + L.b_hi @ Wn.T
    b_hi = L.b_hi @ Wp.T + L.b_lo @ Wn.T
    if b is not None:
        b_lo, b_hi = b_lo + b, b_hi + b
    return LB(A_lo, b_lo, A_hi, b_hi)


def scale_shift(L, s, t=0.0):
    """y = s*z + t with s a CONSTANT array (sign-aware)."""
    sp, sn = np.maximum(s, 0.0), np.minimum(s, 0.0)
    return LB(L.A_lo * sp + L.A_hi * sn, L.b_lo * sp + L.b_hi * sn + t,
              L.A_hi * sp + L.A_lo * sn, L.b_hi * sp + L.b_lo * sn + t)


def relu(L, lo, hi):
    """CROWN ReLU envelope.

    crossing unit with pre-activation in [l, u], l<0<u:
        upper:  y <= (u/(u-l)) (x - l)
        lower:  y >= lam * x,  lam in {0,1} chosen adaptively (CROWN picks the
                slope minimising envelope area: lam = 1 iff u >= |l|)
    Both slopes are non-negative, so each composes with the matching one-sided
    bound. Unlike DeepZ there is no interval remainder: the offset stays inside
    the linear form, which is what keeps correlation alive through the MLP.
    """
    pos = lo >= 0.0
    neg = hi <= 0.0
    cross = ~(pos | neg)
    den = np.where(cross, hi - lo, 1.0)
    a_up = np.where(cross, hi / den, np.where(pos, 1.0, 0.0))
    c_up = np.where(cross, -a_up * lo, 0.0)
    a_lo_s = np.where(cross, (hi >= -lo).astype(np.float64),
                      np.where(pos, 1.0, 0.0))
    # lower: y >= a_lo_s * x ; upper: y <= a_up * x + c_up ; both slopes >= 0
    return LB(L.A_lo * a_lo_s, L.b_lo * a_lo_s,
              L.A_hi * a_up, L.b_hi * a_up + c_up)


def mul_interval(L, lo, hi, t_lo, t_hi):
    """y = z * t with t an independent scalar in [t_lo, t_hi] and z in [lo, hi].

    min_t t*z = min(t_lo z, t_hi z) is CONCAVE in z, so its chord over [lo, hi]
    is a valid linear under-estimator. max_t t*z is convex, so its chord is a
    valid over-estimator. Both stay linear in alpha, which is the whole point --
    an interval multiply here would discard the correlation this engine exists
    to keep.
    """
    f_lo_l = np.minimum(t_lo * lo, t_hi * lo)
    f_lo_h = np.minimum(t_lo * hi, t_hi * hi)
    f_hi_l = np.maximum(t_lo * lo, t_hi * lo)
    f_hi_h = np.maximum(t_lo * hi, t_hi * hi)
    w = np.where(hi - lo > 1e-300, hi - lo, 1.0)
    s_lo = np.where(hi - lo > 1e-300, (f_lo_h - f_lo_l) / w, 0.0)
    c_lo = f_lo_l - s_lo * lo
    s_hi = np.where(hi - lo > 1e-300, (f_hi_h - f_hi_l) / w, 0.0)
    c_hi = f_hi_l - s_hi * lo
    lower = scale_shift(L, s_lo, c_lo)
    upper = scale_shift(L, s_hi, c_hi)
    return LB(lower.A_lo, lower.b_lo, upper.A_hi, upper.b_hi)


def mccormick(La, lo_a, hi_a, Lb, lo_b, hi_b):
    """Linear envelopes for the elementwise product of two forms.

    lower: a*b >= lo_a*b + a*lo_b - lo_a*lo_b
    upper: a*b <= hi_a*b + a*lo_b - hi_a*lo_b
    Each is affine in (a, b) and hence in alpha.
    """
    low = scale_shift(La, lo_b).add(scale_shift(Lb, lo_a, -lo_a * lo_b))
    upp = scale_shift(La, lo_b).add(scale_shift(Lb, hi_a, -hi_a * lo_b))
    return LB(low.A_lo, low.b_lo, upp.A_hi, upp.b_hi)


# ------------------------------------------------------------------ primitives

def layernorm(L, g, b, eps, mid, rad):
    """g * (x - mean)/sqrt(var+eps) + b.

    Centering is an exact linear map. The scalar 1/sqrt(var+eps) is bracketed
    from the concrete bounds of the centred form and applied with the chord
    envelope above, so the result is still linear in alpha.
    """
    D = L.b_lo.shape[-1]
    M = np.eye(D) - np.ones((D, D)) / D
    C = lin(L, M)
    u_lo, u_hi = C.concretize(mid, rad)

    # Coordinate-wise bracket: collapses to eps as soon as every coordinate
    # interval straddles zero, and then 1/sqrt(var_lo) ~ 316.
    sq_hi = np.maximum(u_lo ** 2, u_hi ** 2)
    straddle = (u_lo < 0) & (u_hi > 0)
    sq_lo = np.where(straddle, 0.0, np.minimum(u_lo ** 2, u_hi ** 2))
    var_lo_cw = sq_lo.mean(axis=-1, keepdims=True) + eps
    var_hi_cw = sq_hi.mean(axis=-1, keepdims=True) + eps

    # Norm bracket: var = ||u||^2/D and ||u|| lies within ||centre|| +- ||radius||.
    # This is the bracket that decides whether the engine is usable at all. The
    # chord width below scales with (t_hi - t_lo) times the MAGNITUDE of u, not
    # its width, so a loose 1/sqrt(var) multiplies the full centred signal. With
    # only the coordinate-wise bracket this engine blows up by 24x per LayerNorm
    # even at rho=1e-4. (src/bounds.py brackets the variance the same two ways;
    # that much is a property of LayerNorm, not a shared implementation -- the
    # relaxation applied afterwards is still a chord here and an interval
    # remainder there.)
    cc = 0.5 * (u_lo + u_hi)
    rr = 0.5 * (u_hi - u_lo)
    n_c = np.linalg.norm(cc, axis=-1, keepdims=True)
    n_r = np.linalg.norm(rr, axis=-1, keepdims=True)
    var_lo_nm = np.maximum(n_c - n_r, 0.0) ** 2 / D + eps
    var_hi_nm = (n_c + n_r) ** 2 / D + eps

    var_lo = np.maximum(var_lo_cw, var_lo_nm)
    var_hi = np.minimum(var_hi_cw, var_hi_nm)
    t_lo = 1.0 / np.sqrt(var_hi)
    t_hi = 1.0 / np.sqrt(var_lo)
    t_lo = np.broadcast_to(t_lo, u_lo.shape)
    t_hi = np.broadcast_to(t_hi, u_lo.shape)
    Y = mul_interval(C, u_lo, u_hi, t_lo, t_hi)
    return scale_shift(Y, g, b)


def softmax_bounds(s_lo, s_hi, mask):
    """Monotone corner enclosure of softmax; `mask` True where disallowed."""
    A = ~mask
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        sh = np.max(np.where(A, s_hi, -np.inf), axis=-1, keepdims=True)
        sh = np.where(np.isfinite(sh), sh, 0.0)
        el = np.where(A, np.exp(np.where(A, s_lo - sh, -np.inf)), 0.0)
        eh = np.where(A, np.exp(np.where(A, s_hi - sh, -np.inf)), 0.0)
        sum_l, sum_h = el.sum(-1, keepdims=True), eh.sum(-1, keepdims=True)
        d_l = el + (sum_h - eh)
        d_h = eh + (sum_l - el)
        p_lo = np.where(d_l > 0, el / np.where(d_l > 0, d_l, 1.0), 0.0)
        p_hi = np.where(d_h > 0, eh / np.where(d_h > 0, d_h, 1.0), 1.0)
    ok = np.isfinite(s_lo) & np.isfinite(s_hi) & A
    row_ok = (ok == A).all(axis=-1, keepdims=True)
    p_lo = np.where(row_ok, np.nan_to_num(p_lo, nan=0.0), 0.0)
    p_hi = np.where(row_ok, np.nan_to_num(p_hi, nan=1.0), 1.0)
    return np.where(A, np.clip(p_lo, 0, 1), 0.0), np.where(A, np.clip(p_hi, 0, 1), 0.0)


def softmax_linear(S, s_lo, s_hi, mask, mid, rad):
    """Linear relaxation of softmax, Shi et al. style.

    softmax(s)_i = exp(s_i) / sum_j exp(s_j) is decomposed into three steps,
    each given tangent/secant linear bounds, so the result stays a linear
    function of alpha instead of collapsing to an interval:

      exp   convex, so the SECANT through the endpoints is an upper bound and
            the TANGENT at any interior point is a lower bound.
      sum   exact (linear).
      1/Z   convex decreasing on Z>0, so the secant is an upper bound and the
            tangent a lower bound; its slope is negative, so the one-sided
            forms cross over and must be composed sign-aware.
      e*r   McCormick, both factors non-negative.

    Scores are shifted by the row maximum first. softmax is invariant to that
    shift, so it costs nothing and keeps exp in [0,1] instead of overflowing.

    This replaces an interval enclosure of p combined with a mean-value form,
    which left layer-0 attention at 39.5x the attainable width.
    """
    A = ~mask
    m = np.max(np.where(A, s_hi, -np.inf), axis=-1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    Sh = LB(S.A_lo, S.b_lo - m, S.A_hi, S.b_hi - m)      # shift is a constant
    l = np.where(A, s_lo - m, 0.0)
    u = np.where(A, s_hi - m, 0.0)

    # --- exp: secant above, tangent below
    el, eu = np.exp(l), np.exp(u)
    wid = u - l
    ok = wid > 1e-12
    a_up = np.where(ok, (eu - el) / np.where(ok, wid, 1.0), el)
    b_up = el - a_up * l
    t = 0.5 * (l + u)
    a_lo_ = np.exp(t)
    b_lo_ = a_lo_ * (1.0 - t)
    # both slopes are positive
    E = LB(Sh.A_lo * a_lo_, Sh.b_lo * a_lo_ + b_lo_,
           Sh.A_hi * a_up, Sh.b_hi * a_up + b_up)
    E = LB(np.where(A[None], E.A_lo, 0.0), np.where(A, E.b_lo, 0.0),
           np.where(A[None], E.A_hi, 0.0), np.where(A, E.b_hi, 0.0))
    e_lo, e_hi = E.concretize(mid, rad)
    e_lo = np.where(A, np.maximum(e_lo, 0.0), 0.0)
    e_hi = np.where(A, np.maximum(e_hi, 0.0), 0.0)

    # --- Z = sum_j e_j (exact linear), then r = 1/Z
    Z = LB(E.A_lo.sum(-1, keepdims=True), E.b_lo.sum(-1, keepdims=True),
           E.A_hi.sum(-1, keepdims=True), E.b_hi.sum(-1, keepdims=True))
    z_lo, z_hi = Z.concretize(mid, rad)
    z_lo = np.maximum(z_lo, 1e-12)
    z_hi = np.maximum(z_hi, z_lo + 1e-15)
    a_r = -1.0 / (z_lo * z_hi)                 # secant slope, negative
    b_r = 1.0 / z_lo + 1.0 / z_hi              # secant intercept -> UPPER bound
    tz = 0.5 * (z_lo + z_hi)
    a_t = -1.0 / (tz * tz)                     # tangent slope, negative
    b_t = 2.0 / tz                             # tangent -> LOWER bound
    # negative slopes: lower bound of r uses the UPPER form of Z and vice versa
    R = LB(Z.A_hi * a_t, Z.b_hi * a_t + b_t,
           Z.A_lo * a_r, Z.b_lo * a_r + b_r)
    r_lo, r_hi = R.concretize(mid, rad)
    r_lo = np.maximum(r_lo, 0.0)

    # --- p = e * r, McCormick with both factors non-negative
    # lower: e*r >= e_lo*r + e*r_lo - e_lo*r_lo   (both coefficients >= 0)
    # upper: e*r <= e_hi*r + e*r_lo - e_hi*r_lo
    P_A_lo = E.A_lo * r_lo[None] + R.A_lo * e_lo[None]
    P_b_lo = E.b_lo * r_lo + R.b_lo * e_lo - e_lo * r_lo
    P_A_hi = E.A_hi * r_lo[None] + R.A_hi * e_hi[None]
    P_b_hi = E.b_hi * r_lo + R.b_hi * e_hi - e_hi * r_lo
    P = LB(np.where(A[None], P_A_lo, 0.0), np.where(A, P_b_lo, 0.0),
           np.where(A[None], P_A_hi, 0.0), np.where(A, P_b_hi, 0.0))
    p_lo, p_hi = P.concretize(mid, rad)
    return P, np.where(A, np.clip(p_lo, 0.0, 1.0), 0.0), \
        np.where(A, np.clip(p_hi, 0.0, 1.0), 0.0)


def attention(L, WQ, WK, WV, WO, n_heads, mid, rad):
    T, D = L.b_lo.shape
    H, dh = n_heads, D // n_heads
    Q, Kk, V = lin(L, WQ), lin(L, WK), lin(L, WV)
    q_lo, q_hi = Q.concretize(mid, rad)
    k_lo, k_hi = Kk.concretize(mid, rad)
    v_lo, v_hi = V.concretize(mid, rad)
    v_c = 0.5 * (v_lo + v_hi)
    scale = 1.0 / np.sqrt(dh)
    mask = np.triu(np.ones((T, T), dtype=bool), 1)
    K = L.K
    oA_lo = np.zeros((K, T, D)); oA_hi = np.zeros((K, T, D))
    ob_lo = np.zeros((T, D)); ob_hi = np.zeros((T, D))

    for h in range(H):
        sl = slice(h * dh, (h + 1) * dh)
        # ---- scores: sum_d q_td k_sd, a sum of bilinear terms (McCormick)
        sA_lo = np.zeros((K, T, T)); sA_hi = np.zeros((K, T, T))
        sb_lo = np.zeros((T, T)); sb_hi = np.zeros((T, T))
        for d in range(dh):
            i = h * dh + d
            qa = LB(Q.A_lo[:, :, i:i + 1], Q.b_lo[:, i:i + 1],
                    Q.A_hi[:, :, i:i + 1], Q.b_hi[:, i:i + 1])
            kb = LB(np.transpose(Kk.A_lo[:, :, i:i + 1], (0, 2, 1)),
                    Kk.b_lo[:, i:i + 1].T,
                    np.transpose(Kk.A_hi[:, :, i:i + 1], (0, 2, 1)),
                    Kk.b_hi[:, i:i + 1].T)
            P = mccormick(qa, q_lo[:, i:i + 1], q_hi[:, i:i + 1],
                          kb, k_lo[:, i:i + 1].T, k_hi[:, i:i + 1].T)
            sA_lo = sA_lo + P.A_lo; sA_hi = sA_hi + P.A_hi
            sb_lo = sb_lo + P.b_lo; sb_hi = sb_hi + P.b_hi
        S = LB(sA_lo * scale, sb_lo * scale, sA_hi * scale, sb_hi * scale)
        s_lo, s_hi = S.concretize(mid, rad)
        s_lo = np.where(mask, -np.inf, s_lo)
        s_hi = np.where(mask, -np.inf, s_hi)
        p_lo, p_hi = softmax_bounds(s_lo, s_hi, mask)

        # ---- output: sum_s p_ts v_sd.
        #
        # Treating each p_ts as an independent interval discards sum_s p_ts = 1,
        # which is the dominant loss: it costs sum_s (p_hi-p_lo)*|v_s| where
        # |v_s| is O(1), even when the true output barely moves. Measured at 52x
        # the attainable width on layer 0, against 3.9x for the prover.
        #
        # Recover the constraint exactly. With p = p_c + dp and sum_s dp_s = 0,
        #     sum_s p_s v_s = sum_s p_c_s v_s + sum_s dp_s (v_s - r)
        # for ANY r, because r * sum_s dp_s vanishes. Take r to be the nominal
        # output sum_s p_c_s v_c_s. The first term is EXACTLY linear in alpha
        # (p_c is a constant), so it is relaxed not at all; only the deviation
        # (v_s - r) pays for the softmax uncertainty, and that is the signal
        # rather than the whole value.
        s_c = 0.5 * (np.where(mask, 0.0, s_lo) + np.where(mask, 0.0, s_hi))
        ex = np.where(mask, 0.0, np.exp(np.where(mask, 0.0,
                                                 s_c - s_c.max(-1, keepdims=True))))
        p_c = ex / ex.sum(-1, keepdims=True)                  # (T,T)
        r_nom = p_c @ v_c[:, sl]                              # (T,dh)

        # exact term: sum_s p_c_ts v_sd
        oA_lo[:, :, sl] += np.einsum("ts,ksd->ktd", p_c, V.A_lo[:, :, sl])
        oA_hi[:, :, sl] += np.einsum("ts,ksd->ktd", p_c, V.A_hi[:, :, sl])
        ob_lo[:, sl] += p_c @ V.b_lo[:, sl]
        ob_hi[:, sl] += p_c @ V.b_hi[:, sl]

        # dp as a LINEAR FORM in alpha, from the exp / reciprocal / McCormick
        # relaxation above rather than a mean-value bracket on the softmax
        # Jacobian. The MVT version was only marginally better than holding dp
        # as an interval; this one relaxes each elementary operation with its
        # own tangent/secant hyperplanes and composes them.
        if SOFTMAX_MODE == "linear":
            PL, sp_lo, sp_hi = softmax_linear(S, s_lo, s_hi, mask, mid, rad)
            DP = LB(PL.A_lo, PL.b_lo - p_c, PL.A_hi, PL.b_hi - p_c)
            dp_lo, dp_hi = DP.concretize(mid, rad)
            # intersect with the direct monotone enclosure -- neither dominates
            dp_lo = np.maximum(dp_lo, np.maximum(p_lo, sp_lo) - p_c)
            dp_hi = np.minimum(dp_hi, np.minimum(p_hi, sp_hi) - p_c)
        else:
            # ABLATION: dp carries no alpha-dependence at all, just the monotone
            # interval enclosure. Isolates what the linear relaxation buys.
            zeroA = np.zeros_like(S.A_lo)
            dp_lo, dp_hi = p_lo - p_c, p_hi - p_c
            DP = LB(zeroA, dp_lo, zeroA, dp_hi)

        # sum_s dp_ts * w_tsd with w = v_s - r_t: both factors are linear in
        # alpha and both are small, so McCormick's error is second order.
        # McCormick between TWO linear forms, so both factors keep their
        # alpha-dependence:
        #     dp*w >= dp_lo*w + dp*w_lo - dp_lo*w_lo
        #     dp*w <= dp_hi*w + dp*w_lo - dp_hi*w_lo
        # The dp*w_lo term sums over s with dp still linear, which is what
        # preserves the sum_s dp_s = 0 cancellation. Concretizing dp to an
        # interval first (as an earlier revision did) throws that away and the
        # mean-value form buys nothing.
        wl = (v_lo[None, :, sl] - r_nom[:, None, :])          # (T,T,dh)
        bl = V.b_lo[None, :, sl] - r_nom[:, None, :]
        bh = V.b_hi[None, :, sl] - r_nom[:, None, :]
        dpl = dp_lo[:, :, None]; dph = dp_hi[:, :, None]      # (T,T,1)
        dlp, dln = np.maximum(dpl, 0.0), np.minimum(dpl, 0.0)
        dhp, dhn = np.maximum(dph, 0.0), np.minimum(dph, 0.0)
        wlp, wln = np.maximum(wl, 0.0), np.minimum(wl, 0.0)

        mA_lo = (np.einsum("tsd,ksd->ktd", dlp, V.A_lo[:, :, sl])
                 + np.einsum("tsd,ksd->ktd", dln, V.A_hi[:, :, sl])
                 + np.einsum("tsd,kts->ktd", wlp, DP.A_lo)
                 + np.einsum("tsd,kts->ktd", wln, DP.A_hi))
        mA_hi = (np.einsum("tsd,ksd->ktd", dhp, V.A_hi[:, :, sl])
                 + np.einsum("tsd,ksd->ktd", dhn, V.A_lo[:, :, sl])
                 + np.einsum("tsd,kts->ktd", wlp, DP.A_hi)
                 + np.einsum("tsd,kts->ktd", wln, DP.A_lo))
        mb_lo = (dlp * bl + dln * bh
                 + wlp * DP.b_lo[:, :, None] + wln * DP.b_hi[:, :, None]
                 - dpl * wl).sum(axis=1)
        mb_hi = (dhp * bh + dhn * bl
                 + wlp * DP.b_hi[:, :, None] + wln * DP.b_lo[:, :, None]
                 - dph * wl).sum(axis=1)

        # Fallback: treat dp as a plain interval and take the chord in v only.
        # This is looser at layer 0 but far more stable once dp is wide, which
        # is the regime at layer 1. Both are valid linear bounds, so selecting
        # per output element by whichever concretizes tighter is itself sound --
        # each element simply uses one of two admissible certificates.
        vh_ = (v_hi[None, :, sl] - r_nom[:, None, :])
        f_ll = np.minimum(dpl * wl, dph * wl); f_lh = np.minimum(dpl * vh_, dph * vh_)
        f_hl = np.maximum(dpl * wl, dph * wl); f_hh = np.maximum(dpl * vh_, dph * vh_)
        wd_ = vh_ - wl
        sf = wd_ > 1e-300
        dn_ = np.where(sf, wd_, 1.0)
        cs_lo = np.where(sf, (f_lh - f_ll) / dn_, 0.0)
        e_lo = f_ll - cs_lo * wl
        cs_hi = np.where(sf, (f_hh - f_hl) / dn_, 0.0)
        e_hi = f_hl - cs_hi * wl
        qlp, qln = np.maximum(cs_lo, 0.0), np.minimum(cs_lo, 0.0)
        qhp, qhn = np.maximum(cs_hi, 0.0), np.minimum(cs_hi, 0.0)
        cA_lo = (np.einsum("tsd,ksd->ktd", qlp, V.A_lo[:, :, sl])
                 + np.einsum("tsd,ksd->ktd", qln, V.A_hi[:, :, sl]))
        cA_hi = (np.einsum("tsd,ksd->ktd", qhp, V.A_hi[:, :, sl])
                 + np.einsum("tsd,ksd->ktd", qhn, V.A_lo[:, :, sl]))
        cb_lo = (qlp * bl + qln * bh + e_lo).sum(axis=1)
        cb_hi = (qhp * bh + qhn * bl + e_hi).sum(axis=1)

        mm = mid.reshape((-1, 1, 1)); rr_ = rad.reshape((-1, 1, 1))
        w_m = ((mA_hi * mm).sum(0) + (np.abs(mA_hi) * rr_).sum(0) + mb_hi
               - (mA_lo * mm).sum(0) + (np.abs(mA_lo) * rr_).sum(0) - mb_lo)
        w_c = ((cA_hi * mm).sum(0) + (np.abs(cA_hi) * rr_).sum(0) + cb_hi
               - (cA_lo * mm).sum(0) + (np.abs(cA_lo) * rr_).sum(0) - cb_lo)
        pick_m = (w_m <= w_c)[None]
        oA_lo[:, :, sl] += np.where(pick_m, mA_lo, cA_lo)
        oA_hi[:, :, sl] += np.where(pick_m, mA_hi, cA_hi)
        ob_lo[:, sl] += np.where(pick_m[0], mb_lo, cb_lo)
        ob_hi[:, sl] += np.where(pick_m[0], mb_hi, cb_hi)
    Z = LB(oA_lo, ob_lo, oA_hi, ob_hi)
    return lin(Z, WO)


def block(L, bw, n_heads, eps, mid, rad):
    Y1 = layernorm(L, bw["ln1_g"], bw["ln1_b"], eps, mid, rad)
    Aout = attention(Y1, bw["WQ"], bw["WK"], bw["WV"], bw["WO"], n_heads, mid, rad)
    Z1 = L.add(Aout)
    Y2 = layernorm(Z1, bw["ln2_g"], bw["ln2_b"], eps, mid, rad)
    Pre = lin(Y2, bw["fc_in_W"], bw["fc_in_b"])
    p_lo, p_hi = Pre.concretize(mid, rad)
    Post = relu(Pre, p_lo, p_hi)
    Mout = lin(Post, bw["fc_out_W"], bw["fc_out_b"])
    return Z1.add(Mout)


def logits(a_lo, a_hi, U, x_nom, w):
    """Linear bounds on the final logits as functions of alpha."""
    mid = 0.5 * (a_lo + a_hi); rad = 0.5 * (a_hi - a_lo)
    # x0 = x_nom + sum_k alpha_k U_k -- exactly linear, zero relaxation
    L = LB.exact(U.copy(), x_nom.copy())
    for l in range(w["n_layers"]):
        L = block(L, w["blocks"][l], w["n_heads"], w["ln_eps"], mid, rad)
    L = layernorm(L, w["ln_f_g"], w["ln_f_b"], w["ln_eps"], mid, rad)
    return lin(L, w["unembed"]), mid, rad


def certify_margin(a_lo, a_hi, U, x_nom, w, idx_unsafe, idx_safe, pos=-1):
    """Sound upper bound on max_u L_u - max_j L_j.

    The pair difference is formed BEFORE concretizing, so the alpha-dependence
    shared between L_u and L_j cancels instead of being double-counted. An
    interval engine cannot do this and pays for it.
    """
    Lg, mid, rad = logits(a_lo, a_hi, U, x_nom, w)
    Au = Lg.A_hi[:, pos, :][:, idx_unsafe]      # (K, |U|)
    bu = Lg.b_hi[pos, idx_unsafe]
    Aj = Lg.A_lo[:, pos, :][:, idx_safe]        # (K, |S|)
    bj = Lg.b_lo[pos, idx_safe]
    A = Au[:, :, None] - Aj[:, None, :]         # (K, |U|, |S|)
    b = bu[:, None] - bj[None, :]
    val = b + (A * mid[:, None, None]).sum(0) + (np.abs(A) * rad[:, None, None]).sum(0)
    return float(val.max(axis=0).min())
