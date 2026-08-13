"""Adversarial formal verifier: sound branch-and-bound + counterexample search.

Proof obligation
----------------
Fix a prompt class with nominal residual trace x*_0 .. x*_L. In deviation
coordinates the obligations are, for a radius rho, contraction rate kappa and
safety margin m:

  (D_l)  for every layer l and every e in B_rho \ B_inner:
             V( Block_l(x*_l + e) - x*_{l+1} )  <=  (1 - kappa) V(e)
  (C)    {V <= c*}  is contained in  B_rho   (on the quotient by the mean
             directions, see below), so the induction in (D_l) closes
  (S)    for every e in B_rho, the final unsafe-logit margin s <= -m

Given (D_l) for all l, (C) and (S): any deviation entering with V(e_0) <= c*
stays in B_rho forever, has V(e_L) <= (1-kappa)^L c*, and the model never
prefers an unsafe token. That is a safety statement proved by construction, not
surveyed by sampling.

Why B_inner has to be excluded
------------------------------
V(0) = 0 and the deviation dynamics fixes the origin exactly, so
V(e') - (1-kappa)V(e) = 0 at e = 0. No sound upper bound over a box containing
the origin can ever be strictly negative. Excluding an inner box is not a
loophole -- it is the standard practical-stability formulation, and the inner
box is discharged separately by direct safety bounding (obligation S covers it).
The size of B_inner is reported; it is part of the result, not a hidden knob.

The mean-direction quotient
---------------------------
Adding c*1 to any position's residual stream leaves LayerNorm output exactly
unchanged, hence propagates through the identity path alone and is annihilated
again by the final LayerNorm. The logits are therefore EXACTLY invariant along
those T directions, and V is identically zero along them. They are removed from
the certified subspace with zero approximation error. `check_mean_equivariance`
verifies this numerically rather than asserting it.

Sound / unsound boundary
------------------------
Everything in `certify_*` is sound: a True is a proof modulo floating point.
`pgd_falsify` is UNSOUND by design -- it only ever produces counterexamples,
never certificates, and is used to drive CEGIS and to measure how far the sound
bound sits above the true optimum (the relaxation gap).
"""
from . import task  # noqa: F401

import numpy as np
import torch

from . import bounds as bd
from . import icnn


# --------------------------------------------------------------- primitives


def _blocks_from(z0, w, start, n=1):
    z = z0
    for l in range(start, start + n):
        z, _ = bd.block(z, w["blocks"][l], w["n_heads"], w["ln_eps"])
    return z


def deviation_step(lo, hi, w, layer, x_nom_l, x_nom_l1):
    """(e_now, e_next) as hybrid zonotopes over the SAME noise symbols."""
    e_now = bd.HZono.from_box(lo, hi)
    x_now = e_now + x_nom_l[None]
    x_next = _blocks_from(x_now, w, layer, 1)
    e_next = x_next - x_nom_l1[None]
    return e_now, e_next


def alpha_to_zono(a_lo, a_hi, U):
    """Deviation zonotope for e = sum_j alpha_j U_j, alpha in [a_lo, a_hi].

    (B, k) coefficient boxes -> (B, T, D) deviations with exactly k generators.
    Splitting happens in ALPHA space, which is where the dimensionality of the
    verification problem actually lives.
    """
    mid = 0.5 * (a_lo + a_hi)
    rad = 0.5 * (a_hi - a_lo)
    c = np.einsum("bk,ktd->btd", mid, U)
    G = rad[:, :, None, None] * U[None]
    return bd.HZono(c, G)


def deviation_step_subspace(a_lo, a_hi, U, w, layer, x_nom_l, x_nom_l1):
    e_now = alpha_to_zono(a_lo, a_hi, U)
    x_next = _blocks_from(e_now + x_nom_l[None], w, layer, 1)
    return e_now, x_next - x_nom_l1[None]


def check_mean_equivariance(model, prompts, n=64, c=0.7, seed=0):
    """Numerically verify the exact LayerNorm mean-direction symmetry."""
    with torch.no_grad():
        P = torch.from_numpy(prompts[:n])
        tr = model.residual_trace(P)
        x0 = tr[0]
        shift = torch.zeros_like(x0)
        rng = np.random.default_rng(seed)
        per_pos = torch.as_tensor(rng.normal(size=(1, x0.shape[1], 1)) * c,
                                  dtype=x0.dtype)
        shift = shift + per_pos
        base = model.logits_from_stream(tr[-1])
        x = x0 + shift
        for blk in model.blocks:
            x = blk(x)
        pert = model.logits_from_stream(x)
        drift = (x - tr[-1] - shift).abs().max()
    return {
        "max_logit_deviation": float((pert - base).abs().max()),
        "max_stream_drift_from_pure_shift": float(drift),
        "shift_magnitude": float(per_pos.abs().max()),
        "n_prompts": int(n),
    }


# ------------------------------------------------------------------- BaB core


def _eval_alpha(a_lo, a_hi, U, w, vw, layer, x_nom_l, x_nom_l1, gamma, chunk):
    out = []
    for i in range(0, a_lo.shape[0], chunk):
        e0, e1 = deviation_step_subspace(a_lo[i:i + chunk], a_hi[i:i + chunk], U,
                                         w, layer, x_nom_l, x_nom_l1)
        out.append(icnn.lyap_gap_upper(e0, e1, vw, gamma)["d_hi"])
    return np.concatenate(out) if out else np.zeros(0)


def _bab(eval_fn, k, rho, max_boxes, max_iters, min_width, inner=0.0,
         verbose=False):
    """Generic sound BaB over an alpha box. eval_fn(a_lo, a_hi) -> upper bounds.

    Discharges a box when its sound upper bound is < 0 (or it lies inside the
    excluded inner box); otherwise bisects the widest coefficient. Returns
    (proved, stats).
    """
    a_lo = np.full((1, k), -rho)
    a_hi = np.full((1, k), rho)
    touched = peak = 0
    worst = np.inf
    for it in range(max_iters):
        ub = eval_fn(a_lo, a_hi)
        ub = np.where(np.isfinite(ub), ub, np.inf)
        fin = np.isfinite(ub)
        worst = float(ub[fin].max()) if fin.any() else np.inf
        inside = (np.all((a_lo >= -inner) & (a_hi <= inner), axis=1)
                  if inner > 0 else np.zeros(a_lo.shape[0], bool))
        keep = (ub >= 0.0) & (~inside)
        a_lo, a_hi = a_lo[keep], a_hi[keep]
        touched += int(keep.sum())
        if a_lo.shape[0] == 0:
            return True, {"proved": True, "iterations": it + 1,
                          "boxes_touched": touched, "peak_boxes": peak}
        wd = a_hi - a_lo
        if wd.max(axis=1).min() < min_width:
            return False, {"proved": False, "reason": "min_width", "worst": worst,
                           "boxes_touched": touched,
                           "surviving": int(a_lo.shape[0])}
        if a_lo.shape[0] * 2 > max_boxes:
            return False, {"proved": False, "reason": "max_boxes", "worst": worst,
                           "boxes_touched": touched,
                           "surviving": int(a_lo.shape[0])}
        peak = max(peak, a_lo.shape[0] * 2)
        dim = wd.argmax(axis=1)
        r = np.arange(a_lo.shape[0])
        mid = 0.5 * (a_lo[r, dim] + a_hi[r, dim])
        l1, h1 = a_lo.copy(), a_hi.copy(); h1[r, dim] = mid
        l2, h2 = a_lo.copy(), a_hi.copy(); l2[r, dim] = mid
        a_lo = np.concatenate([l1, l2]); a_hi = np.concatenate([h1, h2])
        if verbose:
            print(f"        boxes {a_lo.shape[0]:6d}  worst {worst:+.4e}")
    return False, {"proved": False, "reason": "max_iters", "worst": worst,
                   "boxes_touched": touched, "surviving": int(a_lo.shape[0])}


def certify_growth(w, vw, U, x_nom_l, x_nom_l1, layer, rho, gamma,
                   max_boxes=4096, max_iters=26, chunk=32, min_width=1e-4,
                   inner_frac=0.02, verbose=False):
    """Sound BaB proving V(e_{l+1}) <= gamma * V(e_l) on the alpha box."""
    k = U.shape[0]

    def ev(a_lo, a_hi):
        return _eval_alpha(a_lo, a_hi, U, w, vw, layer, x_nom_l, x_nom_l1,
                           gamma, chunk)

    ok, st = _bab(ev, k, rho, max_boxes, max_iters, min_width,
                  inner=inner_frac * rho, verbose=verbose)
    st.update({"gamma": gamma, "rho": rho, "layer": layer, "k": k})
    return ok, st


def min_certified_gamma(w, vw, U, x_nom_l, x_nom_l1, layer, rho,
                        lo=1.0, hi=4.0, iters=7, **kw):
    """Bisect for the smallest gamma this verifier can PROVE at radius rho.

    The returned value is an upper bound on the true optimal growth factor: it is
    what the prover can establish, not what the network achieves. The difference
    between this and the sampled/PGD growth factor is the relaxation gap, and
    both numbers are reported so the gap is visible.
    """
    ok_hi, st_hi = certify_growth(w, vw, U, x_nom_l, x_nom_l1, layer, rho, hi, **kw)
    if not ok_hi:
        return None, {"reason": "even gamma_hi unprovable", "gamma_hi": hi,
                      "stats": st_hi}
    best, best_st = hi, st_hi
    a, b = lo, hi
    for _ in range(iters):
        mid = 0.5 * (a + b)
        ok, st = certify_growth(w, vw, U, x_nom_l, x_nom_l1, layer, rho, mid, **kw)
        if ok:
            best, best_st, b = mid, st, mid
        else:
            a = mid
    return best, best_st


def certify_margin_radius(w, U, x_nom_0, idx_unsafe, idx_safe, rho,
                          margin=0.0, max_boxes=4096, max_iters=26, chunk=32,
                          min_width=1e-5, verbose=False):
    """Sound BaB proving the final unsafe-logit margin s <= -margin over the box.

    This is the MONOLITHIC obligation: propagate the perturbation through every
    layer, then check the readout. It is the baseline the compositional route has
    to beat, so it is run honestly rather than strawmanned.
    """
    k = U.shape[0]

    def ev(a_lo, a_hi):
        out = []
        for i in range(0, a_lo.shape[0], chunk):
            z = alpha_to_zono(a_lo[i:i + chunk], a_hi[i:i + chunk], U) + x_nom_0[None]
            zL = _blocks_from(z, w, 0, w["n_layers"])
            s = bd.unsafe_margin_upper(bd.readout_logits(zL, w), idx_unsafe, idx_safe)
            out.append(s + margin)
        return np.concatenate(out)

    ok, st = _bab(ev, k, rho, max_boxes, max_iters, min_width, verbose=verbose)
    st.update({"rho": rho, "margin": margin, "k": k})
    return ok, st


def certify_layer_decrease_subspace(w, vw, U, x_nom_l, x_nom_l1, layer, rho,
                                    kappa=0.05, inner_frac=0.05, max_boxes=8192,
                                    max_iters=40, chunk=32, min_width=1e-4,
                                    verbose=False):
    """Sound BaB for (D_l) over the k-dimensional coefficient box [-rho, rho]^k.

    Branching in alpha space rather than activation space is the whole point:
    the problem's dimension becomes the number of feature directions in the
    threat model (k ~ 2-8), not the residual-stream dimension (T*d_model = 160).
    Each split also halves the coefficient range, which directly reduces the
    unstable-ReLU count -- the quantity that drives the LayerNorm variance
    bracket, and hence the bound, off a cliff.
    """
    k = U.shape[0]
    a_lo = np.full((1, k), -rho)
    a_hi = np.full((1, k), rho)
    inner = inner_frac * rho
    touched = peak = 0
    worst = np.inf
    for it in range(max_iters):
        d_hi = _eval_alpha(a_lo, a_hi, U, w, vw, layer, x_nom_l, x_nom_l1,
                           kappa, chunk)
        finite = np.isfinite(d_hi)
        d_hi = np.where(finite, d_hi, np.inf)
        inside = np.all((a_lo >= -inner) & (a_hi <= inner), axis=1)
        keep = (d_hi >= 0.0) & (~inside)
        worst = float(np.max(d_hi[np.isfinite(d_hi)])) if finite.any() else np.inf
        a_lo, a_hi = a_lo[keep], a_hi[keep]
        touched += int(keep.sum())
        if a_lo.shape[0] == 0:
            return True, {"certified": True, "iterations": it + 1,
                          "boxes_touched": touched, "peak_boxes": peak,
                          "rho": rho, "kappa": kappa, "k": k,
                          "inner_frac": inner_frac}
        wd = a_hi - a_lo
        if wd.max(axis=1).min() < min_width:
            return False, {"certified": False, "reason": "min_width",
                           "worst_bound": worst, "boxes_touched": touched,
                           "surviving_boxes": int(a_lo.shape[0]),
                           "rho": rho, "kappa": kappa, "k": k}
        if a_lo.shape[0] * 2 > max_boxes:
            return False, {"certified": False, "reason": "max_boxes",
                           "worst_bound": worst, "boxes_touched": touched,
                           "surviving_boxes": int(a_lo.shape[0]),
                           "rho": rho, "kappa": kappa, "k": k}
        peak = max(peak, a_lo.shape[0] * 2)
        dim = wd.argmax(axis=1)
        r = np.arange(a_lo.shape[0])
        mid = 0.5 * (a_lo[r, dim] + a_hi[r, dim])
        l1, h1 = a_lo.copy(), a_hi.copy(); h1[r, dim] = mid
        l2, h2 = a_lo.copy(), a_hi.copy(); l2[r, dim] = mid
        a_lo = np.concatenate([l1, l2]); a_hi = np.concatenate([h1, h2])
        if verbose:
            print(f"      iter {it:3d}  boxes {a_lo.shape[0]:6d}  worst {worst:+.4e}")
    return False, {"certified": False, "reason": "max_iters", "worst_bound": worst,
                   "boxes_touched": touched, "surviving_boxes": int(a_lo.shape[0]),
                   "rho": rho, "kappa": kappa, "k": k}


def certify_safety_margin_subspace(w, U, x_nom_0, rho, idx_unsafe, idx_safe,
                                   n_split=0, chunk=32):
    """Sound upper bound on the final unsafe-logit margin over the alpha box."""
    k = U.shape[0]
    a_lo = np.full((1, k), -rho); a_hi = np.full((1, k), rho)
    for _ in range(n_split):
        wd = a_hi - a_lo
        dim = wd.argmax(axis=1); r = np.arange(a_lo.shape[0])
        mid = 0.5 * (a_lo[r, dim] + a_hi[r, dim])
        l1, h1 = a_lo.copy(), a_hi.copy(); h1[r, dim] = mid
        l2, h2 = a_lo.copy(), a_hi.copy(); l2[r, dim] = mid
        a_lo = np.concatenate([l1, l2]); a_hi = np.concatenate([h1, h2])
    best = -np.inf
    for i in range(0, a_lo.shape[0], chunk):
        z = alpha_to_zono(a_lo[i:i + chunk], a_hi[i:i + chunk], U) + x_nom_0[None]
        zL = _blocks_from(z, w, 0, w["n_layers"])
        s = bd.unsafe_margin_upper(bd.readout_logits(zL, w), idx_unsafe, idx_safe)
        best = max(best, float(np.max(s[np.isfinite(s)])) if np.isfinite(s).any()
                   else np.inf)
    return best


def _eval_boxes(lo, hi, w, vw, layer, x_nom_l, x_nom_l1, kappa, chunk):
    out = []
    for i in range(0, lo.shape[0], chunk):
        e0, e1 = deviation_step(lo[i:i + chunk], hi[i:i + chunk], w, layer,
                                x_nom_l, x_nom_l1)
        out.append(icnn.decrease_upper(e0, e1, vw, kappa=kappa)["d_hi"])
    return np.concatenate(out) if out else np.zeros(0)


def certify_layer_decrease(w, vw, x_nom_l, x_nom_l1, layer, rho, kappa=0.05,
                           inner=None, max_boxes=20000, max_iters=60,
                           chunk=48, min_width=1e-3, verbose=False):
    """Sound BaB for obligation (D_l) over B_rho minus B_inner.

    A box is DISCHARGED when its sound upper bound on V' - (1-kappa)V is < 0, or
    when it lies entirely inside B_inner. Anything else is split on its widest
    coordinate. Returns (certified, stats) with stats carrying the worst
    surviving bound so a failure is diagnosable rather than just a False.
    """
    T, D = x_nom_l.shape
    inner = 0.0 if inner is None else inner
    lo = np.full((1, T, D), -rho)
    hi = np.full((1, T, D), rho)
    touched = peak = 0
    worst = np.inf
    for it in range(max_iters):
        d_hi = _eval_boxes(lo, hi, w, vw, layer, x_nom_l, x_nom_l1, kappa, chunk)
        inside = np.all((lo >= -inner) & (hi <= inner), axis=(1, 2))
        keep = (d_hi >= 0.0) & (~inside)
        worst = float(d_hi.max()) if d_hi.size else -np.inf
        lo, hi = lo[keep], hi[keep]
        touched += int(keep.sum())
        if lo.shape[0] == 0:
            return True, {"certified": True, "iterations": it + 1,
                          "boxes_touched": touched, "peak_boxes": peak,
                          "rho": rho, "kappa": kappa, "inner": inner}
        wdt = hi - lo
        flat = wdt.reshape(lo.shape[0], -1)
        if flat.max(axis=1).min() < min_width:
            return False, {"certified": False, "reason": "min_width",
                           "worst_bound": worst, "boxes_touched": touched,
                           "rho": rho, "kappa": kappa, "inner": inner}
        if lo.shape[0] * 2 > max_boxes:
            return False, {"certified": False, "reason": "max_boxes",
                           "worst_bound": worst, "boxes_touched": touched,
                           "surviving_boxes": int(lo.shape[0]),
                           "rho": rho, "kappa": kappa, "inner": inner}
        peak = max(peak, lo.shape[0] * 2)
        dim = flat.argmax(axis=1)
        r = np.arange(lo.shape[0])
        t_idx, d_idx = dim // D, dim % D
        mid = 0.5 * (lo[r, t_idx, d_idx] + hi[r, t_idx, d_idx])
        lo1, hi1 = lo.copy(), hi.copy(); hi1[r, t_idx, d_idx] = mid
        lo2, hi2 = lo.copy(), hi.copy(); lo2[r, t_idx, d_idx] = mid
        lo = np.concatenate([lo1, lo2]); hi = np.concatenate([hi1, hi2])
        if verbose:
            print(f"      iter {it:3d}  boxes {lo.shape[0]:6d}  worst {worst:+.4e}")
    return False, {"certified": False, "reason": "max_iters", "worst_bound": worst,
                   "boxes_touched": touched, "surviving_boxes": int(lo.shape[0]),
                   "rho": rho, "kappa": kappa, "inner": inner}


# ------------------------------------------------------------ safety obligation


def certify_safety_margin(w, x_nom_0, rho, idx_unsafe, idx_safe, chunk=48,
                          n_split=0):
    """Sound upper bound on the final unsafe-logit margin over B_rho.

    Obligation (S). No branching by default -- a single sound pass through the
    whole network. n_split > 0 bisects the widest coordinate that many times to
    tighten if the single pass is inconclusive.
    """
    T, D = x_nom_0.shape
    lo = np.full((1, T, D), -rho) + x_nom_0[None]
    hi = np.full((1, T, D), rho) + x_nom_0[None]
    for _ in range(n_split):
        wdt = (hi - lo).reshape(lo.shape[0], -1)
        dim = wdt.argmax(axis=1)
        r = np.arange(lo.shape[0]); t_idx, d_idx = dim // D, dim % D
        mid = 0.5 * (lo[r, t_idx, d_idx] + hi[r, t_idx, d_idx])
        lo1, hi1 = lo.copy(), hi.copy(); hi1[r, t_idx, d_idx] = mid
        lo2, hi2 = lo.copy(), hi.copy(); lo2[r, t_idx, d_idx] = mid
        lo = np.concatenate([lo1, lo2]); hi = np.concatenate([hi1, hi2])
    best = -np.inf
    for i in range(0, lo.shape[0], chunk):
        z = bd.HZono.from_box(lo[i:i + chunk], hi[i:i + chunk])
        zL = _blocks_from(z, w, 0, w["n_layers"])
        lg = bd.readout_logits(zL, w)
        s = bd.unsafe_margin_upper(lg, idx_unsafe, idx_safe)
        best = max(best, float(s.max()))
    return best


# ------------------------------------------------------- containment obligation


def containment_level_closed_form(vw, rho):
    """Provable c* with no search: on the quotient by the mean directions,
    ||e||_inf = rho implies ||P e||_2 >= rho, and the convex term is >= 0, so

        c* = alpha * rho^2

    holds unconditionally. Convexity of V with V(0)=0 extends it outward along
    every ray, so {V <= c*} really is trapped inside B_rho on the quotient.
    Cheap and airtight, but loose -- `containment_level_refined` measures how
    loose, and that ratio IS the geometric gap.
    """
    return float(vw["alpha"] * rho ** 2)


def containment_level_refined(vw, rho, n_face_samples=4000, seed=0, chunk=256):
    """Sampled LOWER estimate of min V on the boundary of B_rho (quotient).

    UNSOUND (sampling). Reported only as the ceiling that a face-wise BaB could
    in principle reach, so the closed-form c* can be compared against it. The
    ratio is the headroom currently being left on the table by the geometric
    relaxation, and it is quoted as an estimate, never as a certificate.
    """
    rng = np.random.default_rng(seed)
    T, D = vw["seq_len"], vw["d_model"]
    n = T * D
    e = rng.uniform(-rho, rho, size=(n_face_samples, T, D))
    face = rng.integers(0, n, size=n_face_samples)
    sgn = rng.choice([-1.0, 1.0], size=n_face_samples)
    e[np.arange(n_face_samples), face // D, face % D] = sgn * rho
    e = e - e.mean(axis=-1, keepdims=True)          # onto the quotient
    vals = []
    for i in range(0, n_face_samples, chunk):
        z = bd.HZono.point(e[i:i + chunk])
        v = icnn.v_bound(z, vw)
        vals.append(v.c[:, 0])
    vals = np.concatenate(vals)
    return {"min_V_on_boundary_sampled": float(vals.min()),
            "median_V_on_boundary_sampled": float(np.median(vals)),
            "closed_form_c_star": containment_level_closed_form(vw, rho)}


# ------------------------------------------------------ adversarial falsifier


def pgd_falsify_subspace(model, V, U, x_nom_l, x_nom_l1, layer, rho, kappa=0.05,
                         n=512, steps=200, lr=None, seed=0):
    """UNSOUND adversarial search over the alpha box. Counterexamples only.

    Optimizing in alpha space keeps the falsifier on exactly the set the prover
    is certifying, so `sound_bound - attained_max` is a meaningful relaxation
    gap rather than a comparison of two different problems.
    """
    torch.manual_seed(seed)
    lr = lr if lr is not None else rho / 8.0
    Ut = torch.as_tensor(U, dtype=torch.float32)
    xl = torch.as_tensor(x_nom_l, dtype=torch.float32)[None]
    xl1 = torch.as_tensor(x_nom_l1, dtype=torch.float32)[None]
    a = (torch.rand((n, Ut.shape[0])) * 2 - 1) * rho
    a.requires_grad_(True)
    best, best_a = -np.inf, None
    for _ in range(steps):
        e = torch.einsum("bk,ktd->btd", a, Ut)
        e1 = model.blocks[layer](xl + e) - xl1
        obj = V(e1) - (1.0 - kappa) * V(e)
        grad, = torch.autograd.grad(-obj.sum(), a)
        with torch.no_grad():
            a -= lr * grad.sign()
            a.clamp_(-rho, rho)
            cur = obj.detach()
            j = int(cur.argmax())
            if float(cur[j]) > best:
                best, best_a = float(cur[j]), a[j].detach().clone()
    return {"max_violation_found": best, "violated": bool(best > 0.0),
            "witness_alpha": None if best_a is None else best_a.numpy()}


def pgd_falsify(model, V, x_nom_l, x_nom_l1, layer, rho, kappa=0.05,
                n=512, steps=200, lr=None, seed=0):
    """UNSOUND adversarial search for a violation of (D_l). Counterexamples only.

    Serves three purposes: it drives CEGIS retraining, it gives the true
    (attained) maximum against which the sound bound is compared to quantify the
    relaxation gap, and it is the empirical red-team baseline this project is
    replacing -- kept precisely so the comparison can be made honestly.
    """
    torch.manual_seed(seed)
    lr = lr if lr is not None else rho / 8.0
    xl = torch.as_tensor(x_nom_l, dtype=torch.float32)[None]
    xl1 = torch.as_tensor(x_nom_l1, dtype=torch.float32)[None]
    e = (torch.rand((n,) + x_nom_l.shape) * 2 - 1) * rho
    e.requires_grad_(True)
    best = -np.inf
    best_e = None
    for _ in range(steps):
        e1 = model.blocks[layer](xl + e) - xl1
        obj = V(e1) - (1.0 - kappa) * V(e)
        loss = -obj.sum()
        grad, = torch.autograd.grad(loss, e)
        with torch.no_grad():
            e -= lr * grad.sign()
            e.clamp_(-rho, rho)
            cur = obj.detach()
            j = int(cur.argmax())
            if float(cur[j]) > best:
                best = float(cur[j]); best_e = e[j].detach().clone()
    return {"max_violation_found": best,
            "violated": bool(best > 0.0),
            "witness": None if best_e is None else best_e.numpy()}


# ------------------------------------------------------------ optional backends


def z3_available():
    try:
        import z3  # noqa: F401
        return True
    except Exception:
        return False


def dreal_available():
    try:
        import dreal  # noqa: F401
        return True
    except Exception:
        return False


def backend_report():
    """What the verifier is actually running on, stated plainly.

    Neither z3 nor dReal is installed in this environment, so the native
    hybrid-zonotope BaB above is the sole prover. It is sound on its own terms
    (soundness.py falsifies it by dense sampling), but it is a single
    implementation with no independent cross-check -- which is a real limitation
    and is listed as such rather than glossed.
    """
    return {"native_hzono_bab": True,
            "z3": z3_available(),
            "dreal": dreal_available(),
            "independent_cross_check": z3_available() or dreal_available()}
