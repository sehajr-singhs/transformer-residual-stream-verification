"""Interpretable feature-routing task.

A sequence is  [f_0, f_1, f_2, f_3, SEL_k]  and the target is  f_k.

The task is chosen because it forces the attention layer to implement a
*routing* circuit that is legible: head activity has ground-truth semantics
(which slot is being copied), so an SAE trained on the residual stream has real
structure to find rather than arbitrary directions. That matters -- the whole
architectural bet is that the latent feature frame is where the dynamics is
linearizable, and that claim is only testable if ground-truth features exist.

Safety semantics
----------------
Feature tokens 6 and 7 are designated UNSAFE. The safety monitor is the
unsafe-logit margin

    s(x) = max_{v in UNSAFE} logit_v(x) - max_{v not in UNSAFE} logit_v(x)

with s(x) < 0 meaning "the model does not prefer an unsafe token". The target
safe set is S_m = {x : s(x) <= -m} for a margin m > 0. This is a *linear*
readout of the final residual stream (after the final LayerNorm), which is what
makes set containment {V <= c*} subset-of S_m checkable rather than aspirational.
"""
import numpy as np

N_FEAT = 8            # feature tokens, ids 0..7
N_SEL = 4             # selector tokens, ids 8..11
VOCAB = N_FEAT + N_SEL
SEQ_LEN = N_SEL + 1   # 4 slots + 1 selector
UNSAFE_FEATURES = (6, 7)


def safe_mask(vocab=VOCAB):
    """Boolean mask over the vocabulary, True where the token is safe."""
    m = np.ones(vocab, dtype=bool)
    for v in UNSAFE_FEATURES:
        m[v] = False
    return m


def sample_batch(n, rng, safe_only=False):
    """Return (tokens (n, SEQ_LEN) int64, target (n,) int64).

    safe_only=True restricts to prompts whose *selected* slot holds a safe
    feature. Those are the prompts the certificate is stated over: the nominal
    trajectory has to start inside the safe set before invariance means anything.
    """
    slots = rng.integers(0, N_FEAT, size=(n, N_SEL))
    sel = rng.integers(0, N_SEL, size=n)
    if safe_only:
        tgt = slots[np.arange(n), sel]
        bad = np.isin(tgt, UNSAFE_FEATURES)
        while bad.any():
            k = int(bad.sum())
            slots[bad, :] = rng.integers(0, N_FEAT, size=(k, N_SEL))
            sel[bad] = rng.integers(0, N_SEL, size=k)
            tgt = slots[np.arange(n), sel]
            bad = np.isin(tgt, UNSAFE_FEATURES)
    toks = np.concatenate([slots, (N_FEAT + sel)[:, None]], axis=1)
    target = slots[np.arange(n), sel]
    return toks.astype(np.int64), target.astype(np.int64)


def enumerate_prompts(limit=None, rng=None, safe_only=True):
    """A deterministic sample of prompt classes used as verification anchors.

    Certification is stated per prompt class (the discrete context is fixed;
    the perturbation is continuous, in activation space), so we need a concrete
    finite anchor set. Full enumeration is N_FEAT**N_SEL * N_SEL = 16384.
    """
    rng = rng or np.random.default_rng(0)
    if limit is None:
        grids = np.array(np.meshgrid(*[np.arange(N_FEAT)] * N_SEL, indexing="ij"))
        slots = grids.reshape(N_SEL, -1).T
        slots = np.repeat(slots, N_SEL, axis=0)
        sel = np.tile(np.arange(N_SEL), slots.shape[0] // N_SEL)
        toks = np.concatenate([slots, (N_FEAT + sel)[:, None]], axis=1)
        target = slots[np.arange(len(slots)), sel]
        if safe_only:
            keep = ~np.isin(target, UNSAFE_FEATURES)
            toks, target = toks[keep], target[keep]
        return toks.astype(np.int64), target.astype(np.int64)
    return sample_batch(limit, rng, safe_only=safe_only)


def unsafe_margin(logits, unsafe=UNSAFE_FEATURES):
    """s(x) for a batch of logits (..., VOCAB). Negative means safe."""
    logits = np.asarray(logits)
    um = np.max(logits[..., list(unsafe)], axis=-1)
    mask = safe_mask(logits.shape[-1])
    sm = np.max(logits[..., mask], axis=-1)
    return um - sm


def margin_readout(vocab=VOCAB, unsafe=UNSAFE_FEATURES):
    """(rows_unsafe, rows_safe) index arrays for the max-of-linear safety form.

    s(x) = max_i (W_u x)_i - max_j (W_s x)_j is a difference of two maxima of
    linear forms. Upper-bounding s over a set therefore needs an upper bound on
    the first max and a *lower* bound on the second -- both directly available
    from a zonotope, no relaxation required.
    """
    idx_u = np.array(list(unsafe), dtype=np.int64)
    idx_s = np.where(safe_mask(vocab))[0].astype(np.int64)
    return idx_u, idx_s
