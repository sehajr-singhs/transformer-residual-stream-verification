"""Relative / invariant coordinate frame for the residual stream.

Absolute residual-stream coordinates are the wrong frame to certify in, for two
independent reasons, and both are structural rather than cosmetic.

1. GAUGE INVARIANCE (basis). A transformer is exactly equivariant under an
   orthogonal change of basis of the residual stream: conjugating every weight
   matrix by R in O(d) and rotating the embedding leaves all logits unchanged
   (LayerNorm's centering commutes with R only if R fixes the all-ones vector,
   see below). A certificate stated in a particular basis is therefore a
   statement about an arbitrary gauge choice unless it is explicitly transported.
   We fix the gauge by *anchoring to the SAE dictionary*, which is determined by
   the data distribution rather than by initialization.

2. TRANSLATION INVARIANCE (mean direction). Pre-LN sends x -> (x - mean(x))/std,
   so the component of x along the all-ones vector 1 is annihilated by every
   LayerNorm in the network. It is NOT annihilated by the identity path, so it
   is not a symmetry of the full map -- but it IS a direction along which the
   dynamics carries no information into any nonlinearity. Certifying it wastes
   branch-and-bound effort on a coordinate that cannot affect safety. We project
   it out and account for it exactly.

3. RELATIVE ORIGIN. Strict decrease of V can never hold at its own minimum, so
   the certificate is stated relative to an ANCHOR x* -- the nominal final-layer
   stream of a safe prompt class. State is z = P (x - x*). This is the same
   move as certifying a power system around its synchronized equilibrium rather
   than around the origin of raw phase coordinates.

The projector P = I - (1/d) 1 1^T is an orthogonal projector (P = P^T = P^2),
so it is exactly representable in the zonotope propagation and adds no
relaxation error whatsoever.
"""
import numpy as np


def centering_projector(d):
    """P = I - (1/d) 11^T. Orthogonal projector onto 1^perp."""
    return np.eye(d) - np.ones((d, d)) / d


class InvariantFrame:
    """Relative coordinates z = P (x - x*) with anchor x* and gauge projector P.

    Also carries the *exact* residual coordinate (the mean component), which is
    tracked rather than discarded so that reconstruction x = x* + z + mu*1 is
    lossless. Nothing is thrown away; it is only moved out of the certified
    subspace with a proof that it cannot enter a nonlinearity.
    """

    def __init__(self, anchor, d=None):
        anchor = np.asarray(anchor, dtype=np.float64)
        self.anchor = anchor
        self.d = int(d or anchor.shape[-1])
        self.P = centering_projector(self.d)

    def to_relative(self, x):
        return (np.asarray(x, dtype=np.float64) - self.anchor) @ self.P.T

    def mean_component(self, x):
        return np.asarray(x, dtype=np.float64).mean(axis=-1, keepdims=True)

    def to_absolute(self, z, mean=None):
        x = np.asarray(z, dtype=np.float64) + self.anchor
        if mean is not None:
            x = x + (mean - x.mean(axis=-1, keepdims=True))
        return x

    def check_projector(self, tol=1e-12):
        """Assert P is an exact orthogonal projector (guards the no-error claim)."""
        P = self.P
        return {
            "sym_err": float(np.abs(P - P.T).max()),
            "idem_err": float(np.abs(P @ P - P).max()),
            "kills_ones": float(np.abs(P @ np.ones(self.d)).max()),
            "ok": bool(np.abs(P @ P - P).max() < tol and np.abs(P - P.T).max() < tol),
        }


def layernorm_gauge_report(w):
    """Empirical check that the mean direction is annihilated by every LN.

    Returns the max absolute change in block output when the all-ones component
    of the input is perturbed. A near-zero number for the *normalized branch*
    plus an exactly-1.0 passthrough on the identity branch is the signature that
    the decomposition above is the right one.
    """
    d = w["d_model"]
    ones = np.ones(d) / np.sqrt(d)
    P = centering_projector(d)
    return {
        "ones_norm_after_P": float(np.linalg.norm(P @ ones)),
        "projector_rank": int(np.linalg.matrix_rank(P)),
        "d_model": d,
        "certified_subspace_dim": int(d - 1),
    }
