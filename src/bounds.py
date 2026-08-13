"""Sound bound propagation for transformer residual-stream dynamics.

Representation
--------------
A `HZono` is a hybrid affine form:

    x  in  { c + sum_k G_k eps_k + E .* delta  :  eps in [-1,1]^m, delta in [-1,1]^n }

`G` (m dense generators) carries CORRELATION between coordinates and back to the
original input noise symbols. `E` (elementwise nonnegative radius) is an
uncorrelated interval remainder. Every operation below either keeps a term in
`G` -- exact, no error -- or dumps it into `E` -- sound, but correlation lost.

That split is the whole design. Plain interval arithmetic puts EVERYTHING in `E`
and dies to the wrapping effect within one transformer block. Full DeepZ puts
everything in `G` and grows m without bound (one new symbol per unstable ReLU,
per layer). The hybrid keeps m FIXED at the input dimension, which is what makes
branch-and-bound over thousands of boxes fit in memory, and confines the
relaxation error to terms that are genuinely second order:

    * ReLU crossing offset           -> E   (independent per neuron anyway)
    * LayerNorm 1/sqrt(var) spread   -> E   (scalar per position, x radius)
    * attention score  dq.dk         -> E   (second order)
    * attention  da . v              -> E   (softmax concretized)

Every other term stays in `G`. In particular the ENTIRE identity/residual path
and every weight matrix are exact.

Soundness contract
------------------
Every function here returns an over-approximation of the true reachable set.
`soundness.py` empirically falsifies that claim by dense sampling; any violation
is a bug, not a tuning parameter.
"""
import numpy as np

_ABS = np.abs


class HZono:
    """Hybrid affine form. c: (B,)+S, G: (B,m)+S, E: (B,)+S with E >= 0."""

    __slots__ = ("c", "G", "E")

    def __init__(self, c, G, E=None):
        self.c = np.asarray(c, dtype=np.float64)
        self.G = np.asarray(G, dtype=np.float64)
        self.E = np.zeros_like(self.c) if E is None else np.asarray(E, dtype=np.float64)

    # ------------------------------------------------------------ structure

    @property
    def B(self):
        return self.c.shape[0]

    @property
    def m(self):
        return self.G.shape[1]

    @property
    def S(self):
        return self.c.shape[1:]

    def radius(self):
        return _ABS(self.G).sum(axis=1) + self.E

    def bounds(self):
        r = self.radius()
        return self.c - r, self.c + r

    def width(self):
        lo, hi = self.bounds()
        return hi - lo

    def copy(self):
        return HZono(self.c.copy(), self.G.copy(), self.E.copy())

    def reshape_trailing(self, shape):
        B, m = self.B, self.m
        return HZono(self.c.reshape((B,) + shape),
                     self.G.reshape((B, m) + shape),
                     self.E.reshape((B,) + shape))

    def select(self, idx):
        return HZono(self.c[idx], self.G[idx], self.E[idx])

    # ------------------------------------------------------------ constructors

    @staticmethod
    def from_box(lo, hi):
        """Exact zonotope for an axis-aligned box; one noise symbol per coord."""
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        B = lo.shape[0]
        S = lo.shape[1:]
        n = int(np.prod(S)) if S else 1
        c = 0.5 * (lo + hi)
        r = 0.5 * (hi - lo)
        G = np.zeros((B, n) + S)
        flat = G.reshape(B, n, n)
        rf = r.reshape(B, n)
        idx = np.arange(n)
        flat[:, idx, idx] = rf
        return HZono(c, G)

    @staticmethod
    def from_subspace(c, U, rho):
        """Perturbation confined to span(U): x = c + sum_j alpha_j U_j, |alpha_j| <= rho.

        c: (B,)+S ;  U: (k,)+S direction set ;  returns a k-generator zonotope.

        This is the construction the whole SAE bet cashes out in. A full box over
        the residual stream needs one generator per coordinate (T*d_model = 160
        here), and every bound extraction pays an l1 sum over all of them, so the
        box hull inflates ~2x per dense map REGARDLESS of how tight the
        propagation is. Restricting the perturbation to k feature directions cuts
        the generator count to k and the inflation with it.

        It is also the more defensible threat model. "Arbitrary 160-dimensional
        activation noise" is not what anyone attacks a model with; activation
        steering and feature-level jailbreaks move the residual stream along a
        small number of interpretable directions, which is exactly span(U).
        """
        c = np.asarray(c, dtype=np.float64)
        U = np.asarray(U, dtype=np.float64)
        G = np.broadcast_to(rho * U[None], (c.shape[0],) + U.shape).copy()
        return HZono(c, G)

    @staticmethod
    def point(c):
        c = np.asarray(c, dtype=np.float64)
        return HZono(c, np.zeros((c.shape[0], 0) + c.shape[1:]))

    # ------------------------------------------------------------ algebra

    def _align(self, other):
        if self.m == other.m:
            return self.G, other.G
        m = max(self.m, other.m)
        def pad(z):
            if z.m == m:
                return z.G
            pad_shape = (z.B, m - z.m) + z.S
            return np.concatenate([z.G, np.zeros(pad_shape)], axis=1)
        return pad(self), pad(other)

    def __add__(self, other):
        if isinstance(other, HZono):
            Ga, Gb = self._align(other)
            return HZono(self.c + other.c, Ga + Gb, self.E + other.E)
        return HZono(self.c + other, self.G, self.E)

    def __sub__(self, other):
        if isinstance(other, HZono):
            Ga, Gb = self._align(other)
            return HZono(self.c - other.c, Ga - Gb, self.E + other.E)
        return HZono(self.c - other, self.G, self.E)

    def scale(self, s):
        """Elementwise scale by a deterministic (non-interval) array."""
        return HZono(self.c * s, self.G * s[:, None], self.E * _ABS(s))

    def linear(self, W, b=None):
        """Apply y = W x + b along the LAST trailing axis. Exact in G."""
        c = np.einsum("...i,oi->...o", self.c, W)
        G = np.einsum("...i,oi->...o", self.G, W)
        E = np.einsum("...i,oi->...o", self.E, _ABS(W))
        if b is not None:
            c = c + b
        return HZono(c, G, E)

    def to_interval_remainder(self):
        """Fold all correlation into E. Sound, maximally lossy. Diagnostic use."""
        return HZono(self.c, np.zeros((self.B, 0) + self.S), self.radius())

    def promote_E_topk(self, k):
        """Promote only the k coordinates with the largest interval remainder.

        Full promotion in the MLP hidden layer would add T*d_mlp generators,
        which is unaffordable inside branch-and-bound. The remainder is heavily
        concentrated on a few unstable neurons, so promoting the worst k
        recovers most of the benefit at bounded cost. Sound: promoted
        coordinates move exactly, the rest stay in E untouched.
        """
        n = int(np.prod(self.S))
        k = min(int(k), n)
        if k <= 0 or not self.E.any():
            return self
        Ef = self.E.reshape(self.B, n)
        idx = np.argsort(-Ef, axis=1)[:, :k]
        r = np.arange(self.B)[:, None]
        vals = Ef[r, idx]
        newG = np.zeros((self.B, k, n))
        newG[r, np.arange(k)[None, :], idx] = vals
        keptE = Ef.copy()
        keptE[r, idx] = 0.0
        return HZono(self.c,
                     np.concatenate([self.G, newG.reshape((self.B, k) + self.S)], axis=1),
                     keptE.reshape((self.B,) + self.S))

    def promote_E(self):
        """Move the interval remainder back into explicit diagonal generators.

        Exact (a box IS a zonotope), and it is the single highest-leverage
        operation in the whole engine. A term parked in E is re-boxed at every
        subsequent dense layer, so it grows by the ||W||_1 row norm (~2.8 here);
        the same term carried in G grows by the ||W||_2 norm (~0.5). Over the
        eight dense maps in a 2-layer block that is the difference between a
        usable bound and a 1e11 wrapping factor.

        Cost is m += n, so callers pair it with `compact`.
        """
        if not self.E.any():
            return self
        n = int(np.prod(self.S))
        newG = np.zeros((self.B, n) + self.S)
        flat = newG.reshape(self.B, n, n)
        idx = np.arange(n)
        flat[:, idx, idx] = self.E.reshape(self.B, n)
        return HZono(self.c, np.concatenate([self.G, newG], axis=1),
                     np.zeros_like(self.E))

    def compact(self, m_max):
        """Order reduction: keep the m_max largest generators, box the rest.

        Sound (the discarded generators are replaced by an enclosing box) and
        keeps the dominant correlated directions, which are the ones that
        actually cancel in dV = V' - V.
        """
        if self.m <= m_max:
            return self
        nrm = _ABS(self.G).reshape(self.B, self.m, -1).sum(axis=2)
        order = np.argsort(-nrm, axis=1)
        keep, drop = order[:, :m_max], order[:, m_max:]
        r = np.arange(self.B)[:, None]
        Gk = self.G[r, keep]
        Gd = self.G[r, drop]
        return HZono(self.c, Gk, self.E + _ABS(Gd).sum(axis=1))


# ---------------------------------------------------------------- nonlinearities


def relu(z):
    """DeepZ triangle relaxation with the offset routed into E.

    Stable neurons (lo>=0 or hi<=0) incur ZERO relaxation error. Only crossing
    neurons pay, and they pay mu = -lam*lo/2 which is independent per neuron, so
    putting it in E rather than in a fresh generator loses nothing at creation.
    """
    lo, hi = z.bounds()
    pos = lo >= 0.0
    neg = hi <= 0.0
    cross = ~(pos | neg)
    denom = np.where(cross, hi - lo, 1.0)
    lam = np.where(cross, hi / denom, np.where(pos, 1.0, 0.0))
    mu = np.where(cross, -0.5 * lam * lo, 0.0)
    c = lam * z.c + mu
    G = lam[:, None] * z.G
    E = lam * z.E + mu
    return HZono(c, G, E)


def layernorm(z, gamma, beta, eps=1e-5, axis_size=None):
    """Sound LayerNorm over the last trailing axis.

    Centering is an exact linear map (it is the same projector as frames.P), so
    it stays in G. Only the scalar 1/sqrt(var+eps) is uncertain; its centre
    multiplies the whole centred zonotope (correlation preserved) and only the
    spread times the coordinate magnitude lands in E.
    """
    d = axis_size or z.S[-1]
    P0 = np.eye(d) - np.ones((d, d)) / d
    xc = z.linear(P0)                                   # exact
    lo, hi = xc.bounds()

    # Coordinate-wise variance bracket. Sound but very weak from below: if every
    # coordinate interval straddles zero it yields var_lo = eps, hence
    # 1/sqrt(var) ~ 1/sqrt(eps) ~ 316, and the block blows up by 3 orders of
    # magnitude per layer. It ignores that the coordinates cannot all vanish at
    # once.
    sq_hi = np.maximum(lo ** 2, hi ** 2)
    straddle = (lo < 0) & (hi > 0)
    sq_lo = np.where(straddle, 0.0, np.minimum(lo ** 2, hi ** 2))
    var_lo_cw = sq_lo.mean(axis=-1, keepdims=True) + eps
    var_hi_cw = sq_hi.mean(axis=-1, keepdims=True) + eps

    # Norm-based bracket, which DOES use the correlation the zonotope is
    # carrying: xc = c + w with ||w||_2 <= sum_k ||G_k||_2 + ||E||_2, so
    # ||xc||_2 lies in [ ||c|| - R , ||c|| + R ]. When the centre is far from the
    # origin relative to the perturbation -- the normal case around a nominal
    # trajectory -- this is tighter by orders of magnitude.
    cn = np.linalg.norm(xc.c, axis=-1, keepdims=True)
    # Two incomparable sound bounds on ||w||_2, and their min. The triangle
    # bound sum_k ||G_k|| is tight when generators are ALIGNED and catastrophic
    # when they are orthogonal -- which is exactly the case for a box input
    # (160 axis generators: triangle gives 32*rho, truth is sqrt(32)*rho, a 5.7x
    # over-estimate that then squares inside the variance).
    tot_r = _ABS(xc.G).sum(axis=1) + xc.E
    R2 = np.minimum(np.linalg.norm(tot_r, axis=-1, keepdims=True),
                    np.linalg.norm(xc.G, axis=-1).sum(axis=1)[..., None]
                    + np.linalg.norm(xc.E, axis=-1, keepdims=True))
    var_lo_nm = np.maximum(cn - R2, 0.0) ** 2 / d + eps
    var_hi_nm = (cn + R2) ** 2 / d + eps

    var_lo = np.maximum(var_lo_cw, var_lo_nm)
    var_hi = np.minimum(var_hi_cw, var_hi_nm)
    s_hi = 1.0 / np.sqrt(var_lo)
    s_lo = 1.0 / np.sqrt(var_hi)
    s_c = 0.5 * (s_lo + s_hi)
    s_r = 0.5 * (s_hi - s_lo)
    mag = np.maximum(_ABS(lo), _ABS(hi))
    c = s_c * xc.c * gamma + beta
    G = (s_c * gamma)[:, None] * xc.G
    E = s_c * _ABS(gamma) * xc.E + s_r * _ABS(gamma) * mag
    return HZono(c, G, E)


def softmax_interval(s_lo, s_hi, mask):
    """Sound elementwise bounds on softmax over the LAST axis.

    Exact monotone envelope: a_u is increasing in s_u and decreasing in every
    other s_v, so the extremes are attained at box corners. No relaxation is
    involved -- this bound is tight per-coordinate; what is lost is only the
    joint constraint sum_u a_u = 1.

    mask: bool, True where the entry is ALLOWED (causal).
    """
    neg_inf_lo = np.where(mask, s_lo, -np.inf)
    neg_inf_hi = np.where(mask, s_hi, -np.inf)
    M = np.max(np.where(mask, neg_inf_hi, -np.inf), axis=-1, keepdims=True)
    M = np.where(np.isfinite(M), M, 0.0)
    Elo = np.where(mask, np.exp(np.clip(neg_inf_lo - M, -60, 60)), 0.0)
    Ehi = np.where(mask, np.exp(np.clip(neg_inf_hi - M, -60, 60)), 0.0)
    Slo = Elo.sum(axis=-1, keepdims=True)
    Shi = Ehi.sum(axis=-1, keepdims=True)
    a_lo = Elo / np.maximum(Elo + (Shi - Ehi), 1e-300)
    a_hi = Ehi / np.maximum(Ehi + (Slo - Elo), 1e-300)
    a_lo = np.where(mask, np.clip(a_lo, 0.0, 1.0), 0.0)
    a_hi = np.where(mask, np.clip(a_hi, 0.0, 1.0), 0.0)
    # Simplex tightening: the per-coordinate envelope above ignores sum_u a_u = 1,
    # which is free information. a_u can be no larger than 1 minus what the other
    # entries must at least be, and no smaller than 1 minus what they can at most
    # be. Sound, cheap, and recovers a noticeable amount of the softmax spread.
    lo_sum = a_lo.sum(axis=-1, keepdims=True)
    hi_sum = a_hi.sum(axis=-1, keepdims=True)
    a_hi = np.where(mask, np.minimum(a_hi, 1.0 - (lo_sum - a_lo)), 0.0)
    a_lo = np.where(mask, np.maximum(a_lo, 1.0 - (hi_sum - a_hi)), 0.0)
    a_lo = np.clip(a_lo, 0.0, 1.0)
    return a_lo, np.maximum(np.clip(a_hi, 0.0, 1.0), a_lo)


def causal_mask(T):
    return ~np.triu(np.ones((T, T), dtype=bool), 1)


def softmax_jacobian_bracket(a_c, a_lo, a_hi):
    """Sound interval bracket on the softmax Jacobian J_ij = a_i (d_ij - a_j).

    Used for the mean-value remainder of the linearized softmax. Entries follow
    directly from monotonicity: the off-diagonal -a_i a_j is bracketed by the
    products of the endpoints, and the diagonal x(1-x) attains 1/4 only if 1/2
    lies inside the interval.
    """
    d_lo = np.minimum(a_lo * (1 - a_lo), a_hi * (1 - a_hi))
    peak = (a_lo <= 0.5) & (a_hi >= 0.5)
    d_hi = np.where(peak, 0.25, np.maximum(a_lo * (1 - a_lo), a_hi * (1 - a_hi)))
    off_lo = -a_hi[..., :, None] * a_hi[..., None, :]
    off_hi = -a_lo[..., :, None] * a_lo[..., None, :]
    T = a_c.shape[-1]
    eye = np.eye(T, dtype=bool)
    J_lo = np.where(eye, d_lo[..., :, None] * np.ones_like(off_lo), off_lo)
    J_hi = np.where(eye, d_hi[..., :, None] * np.ones_like(off_hi), off_hi)
    J_c = a_c[..., :, None] * (eye.astype(np.float64) - a_c[..., None, :])
    J_c = np.clip(J_c, J_lo, J_hi)
    return J_c, np.maximum(J_hi - J_c, J_c - J_lo)


def softmax_linear(sc_c, sc_G, sc_E, mask):
    """Linear relaxation of softmax: exact value + exact Jacobian at the centre,
    plus a sound mean-value remainder.

    a(s) = a(s_c) + J(s_c) ds + R,   |R_i| <= sum_j |J_ij(xi) - J_ij(s_c)| |ds_j|

    by the componentwise mean value theorem on the (convex) score box. The point
    of doing this rather than concretizing the softmax is CORRELATION: the
    attention-weight perturbation stays expressed in the same noise symbols as
    the value perturbation, so the two are free to cancel downstream. On a
    routing circuit -- sharp attention over values that differ strongly across
    positions -- concretizing here was worth two orders of magnitude of
    looseness, because a tiny weight uncertainty got multiplied by the full
    spread of v instead of by its correlated variation.

    Returns (a_c, a_G, a_E) with a_c exactly softmax(s_c), hence summing to 1;
    the Jacobian's columns sum to zero, so the linear term preserves that.
    """
    sc_r = _ABS(sc_G).sum(axis=2) + sc_E
    s_lo, s_hi = sc_c - sc_r, sc_c + sc_r
    a_lo, a_hi = softmax_interval(s_lo, s_hi, mask)

    shifted = np.where(mask, sc_c, -np.inf)
    shifted = shifted - np.max(shifted, axis=-1, keepdims=True)
    ex = np.where(mask, np.exp(np.clip(shifted, -60, 60)), 0.0)
    a_c = ex / np.maximum(ex.sum(axis=-1, keepdims=True), 1e-300)

    J_c, J_w = softmax_jacobian_bracket(a_c, a_lo, a_hi)
    a_G = np.einsum("bhtij,bhmtj->bhmti", J_c, sc_G)
    a_E = (np.einsum("bhtij,bhtj->bhti", _ABS(J_c), sc_E)
           + np.einsum("bhtij,bhtj->bhti", J_w, sc_r))
    a_c = np.where(mask, a_c, 0.0)
    a_G = np.where(mask[:, :, None], a_G, 0.0)
    a_E = np.where(mask, a_E, 0.0)
    return a_c, a_G, a_E, (a_lo, a_hi)


def attention(z, WQ, WK, WV, WO, n_heads):
    """Sound multi-head causal attention on a (B, T, D) hybrid zonotope.

    Two bilinear products appear and each is split first-order / second-order:

        scores  = q.k     ->  qc.dk + kc.dq  kept in G ;  dq.dk  -> E
        out     = a @ v   ->  ac . Gv        kept in G ;  da . v -> E

    Keeping ac.Gv in G is what preserves the link from the attention output back
    to the ORIGINAL input perturbation, and is the single biggest tightness win
    in the whole propagation.
    """
    B, T, D = z.c.shape
    H, dh = n_heads, D // n_heads
    scale = 1.0 / np.sqrt(dh)

    q = z.linear(WQ); k = z.linear(WK); v = z.linear(WV)
    qc = q.c.reshape(B, T, H, dh); kc = k.c.reshape(B, T, H, dh)
    qG = q.G.reshape(B, -1, T, H, dh); kG = k.G.reshape(B, -1, T, H, dh)
    qr = q.radius().reshape(B, T, H, dh); kr = k.radius().reshape(B, T, H, dh)

    # scores (B, H, T, T)
    sc_c = np.einsum("bthd,buhd->bhtu", qc, kc) * scale
    sc_G = (np.einsum("bmthd,buhd->bhmtu", qG, kc)
            + np.einsum("bthd,bmuhd->bhmtu", qc, kG)) * scale
    sc_E = np.einsum("bthd,buhd->bhtu", qr, kr) * scale     # second order
    mask = np.broadcast_to(causal_mask(T)[None, None], sc_c.shape)
    a_c, a_G, a_E, (a_lo, a_hi) = softmax_linear(sc_c, sc_G, sc_E, mask)
    a_r = _ABS(a_G).sum(axis=2) + a_E

    vc = v.c.reshape(B, T, H, dh)
    vG = v.G.reshape(B, -1, T, H, dh)
    vE = v.E.reshape(B, T, H, dh)
    vr = v.radius().reshape(B, T, H, dh)

    # o = sum_u a_u v_u with BOTH factors uncertain. Split the bilinear product
    # first order / second order, and use a reference point for the a-side.
    # Because a_c is the exact softmax it sums to 1 and the Jacobian's columns
    # sum to 0, so sum_u da_u = 0 exactly and vref cancels with no correction
    # term. Choosing vref = o_c makes the a-side error scale with the VARIATION
    # of v across key positions rather than with its magnitude.
    o_c = np.einsum("bhtu,buhd->bthd", a_c, vc)
    dvc = vc[:, None] - o_c[:, :, None]                      # (B,Tq,Tk,H,dh)
    o_G = (np.einsum("bhtu,bmuhd->bmthd", a_c, vG)           # a_c * dv   (exact)
           + np.einsum("bhmtu,btuhd->bmthd", a_G, dvc))      # da  * v_c  (exact)
    o_E = (np.einsum("bhtu,buhd->bthd", a_c, vE)
           + np.einsum("bhtu,btuhd->bthd", a_E, _ABS(dvc))
           + np.einsum("bhtu,buhd->bthd", a_r, vr))          # second order only

    o = HZono(o_c.reshape(B, T, D), o_G.reshape(B, -1, T, D), o_E.reshape(B, T, D))
    return o.linear(WO), (a_lo, a_hi)


def mlp(z, fc_in_W, fc_in_b, fc_out_W, fc_out_b, promote_k=128, m_max=None):
    """MLP with partial re-correlation before the contracting output map.

    fc_out is d_mlp -> d_model, so it sums 128 terms; an interval remainder is
    amplified by that l1 row norm while a generator is not. Promoting the worst
    `promote_k` hidden coordinates first is where most of the MLP's looseness is
    recovered.
    """
    h = relu(z.linear(fc_in_W, fc_in_b))
    if promote_k:
        h = h.promote_E_topk(promote_k)
        if m_max:
            h = h.compact(m_max)
    return h.linear(fc_out_W, fc_out_b)


def block(z, bw, n_heads, ln_eps=1e-5, m_max=128, promote=True, promote_k=48,
          ln_promote_k=48):
    """One pre-LN transformer block as a state transition x -> x'.

    `promote` re-correlates the interval remainder at each LayerNorm input --
    both because LN's variance bracket exploits generator structure directly and
    because everything downstream of LN is dense linear algebra where E is
    punished by the l1 row norm. Set promote=False to reproduce the naive
    hybrid behaviour; the stress module uses that switch to measure what the
    re-correlation is worth.
    """
    # SYMBOL ALIGNMENT INVARIANT. `__add__` matches generators by index, so two
    # operands may only be added when one's symbol list is a PREFIX of the
    # other's. `promote_E*` appends and is therefore always safe; `compact`
    # REORDERS and drops, so it may only be applied where a single lineage
    # exists -- at the block boundary, before anything branches. Compacting
    # inside the block (e.g. on the LayerNorm output while the identity path
    # still carries the old ordering) silently pairs up unrelated noise symbols
    # and produces a bound that is not an over-approximation at all. Sound-
    # looking and wrong. tests/test_alignment.py pins this down.
    zc = z.promote_E().compact(m_max) if promote else z

    # LayerNorm ITSELF manufactures interval remainder (the 1/sqrt(var) spread),
    # so promoting only at the block input is not enough -- that remainder then
    # rides through W_V and W_O picking up a factor of ||W||_1 each time.
    #
    # ln_promote_k caps how many coordinates get promoted. Full promotion adds
    # T*d_model = 160 generators at EVERY LayerNorm, and cost is linear in m
    # through the MLP's fc_out einsum (m * T * d_mlp * d_model), so full promotion
    # costs ~100x a no-promotion pass. Branch-and-bound converts throughput into
    # tightness as well, so the useful operating point is the one maximizing
    # bound quality per second, not per pass. See tests/calib_speed.py.
    def lnp(zz):
        if not promote:
            return zz
        return zz.promote_E() if ln_promote_k is None else zz.promote_E_topk(ln_promote_k)

    y1 = lnp(layernorm(zc, bw["ln1_g"], bw["ln1_b"], ln_eps))
    a, patt = attention(y1, bw["WQ"], bw["WK"], bw["WV"], bw["WO"], n_heads)
    z1 = zc + a                                   # zc's symbols prefix a's

    y2 = lnp(layernorm(z1, bw["ln2_g"], bw["ln2_b"], ln_eps))
    m = mlp(y2, bw["fc_in_W"], bw["fc_in_b"], bw["fc_out_W"], bw["fc_out_b"],
            promote_k=(promote_k if promote else 0))
    out = z1 + m                                  # z1's symbols prefix m's
    return (out.compact(m_max) if promote else out), patt


def propagate(z0, w, n_layers=None, m_max=192, promote=True):
    """Full residual trace as hybrid zonotopes: [x_0, x_1, ..., x_L]."""
    n_layers = w["n_layers"] if n_layers is None else n_layers
    trace = [z0]
    z = z0
    for l in range(n_layers):
        z, _ = block(z, w["blocks"][l], w["n_heads"], w["ln_eps"],
                     m_max=m_max, promote=promote)
        trace.append(z)
    return trace


def readout_logits(z, w):
    """Final LN + unembed. (B, T, D) -> (B, T, VOCAB)."""
    y = layernorm(z, w["ln_f_g"], w["ln_f_b"], w["ln_eps"])
    return y.linear(w["unembed"])


def unsafe_margin_upper(logit_z, idx_unsafe, idx_safe, pos=-1):
    """Tight sound upper bound on  s = max_u L_u - max_j L_j.

    The competitor term is a MAXIMUM over safe logits, so it must be bounded from
    BELOW. For any fixed safe index j,

        s = max_u L_u - max_{j'} L_{j'}  <=  max_u (L_u - L_j)

    since max_{j'} L_{j'} >= L_j. Each pair difference (L_u - L_j) is affine in
    the noise symbols, so its supremum over the zonotope is exact up to the E
    term. The inequality holds for EVERY j, so the tightest valid bound is

        s_upper = min_j  max_u  sup(L_u - L_j)

    -- a min over safe indices of a max over unsafe ones.

    Taking a max over both indices instead computes max_u L_u - min_j L_j, which
    is not the margin at all. That bug inflated this bound by a constant ~11.7
    logits that did not vanish as the perturbation radius went to zero, and it
    made the safety certificate unprovable at every radius.
    """
    c = logit_z.c[:, pos, :]
    G = logit_z.G[:, :, pos, :]
    E = logit_z.E[:, pos, :]
    cu = c[:, idx_unsafe][:, :, None]; cs = c[:, idx_safe][:, None, :]
    Gu = G[:, :, idx_unsafe][:, :, :, None]; Gs = G[:, :, idx_safe][:, :, None, :]
    Eu = E[:, idx_unsafe][:, :, None]; Es = E[:, idx_safe][:, None, :]
    pair_hi = (cu - cs) + _ABS(Gu - Gs).sum(axis=1) + Eu + Es   # (B, |U|, |S|)
    return pair_hi.max(axis=1).min(axis=1)
