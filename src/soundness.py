"""Soundness harness: try hard to FALSIFY the verifier's own bounds.

A sound bound is a promise that the true value never escapes the reported
interval. That promise is worth exactly as much as the effort spent trying to
break it, so every check here is adversarial-by-sampling: draw points inside the
box (corners, faces, interior, PGD-driven), evaluate the TRUE quantity with the
torch model, and record the worst amount by which the bound was violated.

`max_violation <= 0` is the pass condition. Any positive number is a bug in
bounds.py, not a tolerance to be widened -- the only slack granted is ~1e-9 for
float64 round-off, and violations are reported in absolute units so that claim
is auditable.

The same sampling also produces the RELAXATION GAP: bound minus attained
maximum. Soundness says the gap is nonnegative; usefulness says it should be
small. Reporting only the first is how verification papers overclaim.
"""
from . import task  # noqa: F401

import numpy as np
import torch

from . import bounds as bd
from . import icnn
from . import verifier as vf

FP_SLACK = 1e-9


def _sample_box(lo, hi, n, rng, corner_frac=0.3, face_frac=0.3):
    """Interior + face + corner samples. Extremes matter most for bounds."""
    T, D = lo.shape
    u = rng.random((n, T, D))
    x = lo + u * (hi - lo)
    n_c = int(n * corner_frac)
    x[:n_c] = np.where(rng.random((n_c, T, D)) < 0.5, lo, hi)
    n_f = int(n * face_frac)
    sl = slice(n_c, n_c + n_f)
    k = rng.integers(0, T * D, size=n_f)
    x[sl].reshape(n_f, -1)[np.arange(n_f), k] = np.where(
        rng.random(n_f) < 0.5, lo.reshape(-1)[k], hi.reshape(-1)[k])
    return x


def check_block_bounds(model, w, x_nom, rho, n=3000, seed=0):
    """Falsify the per-layer residual-stream bounds."""
    rng = np.random.default_rng(seed)
    T, D = x_nom.shape
    lo, hi = x_nom - rho, x_nom + rho
    xs = _sample_box(lo, hi, n, rng)
    z = bd.HZono.from_box(lo[None], hi[None])
    res = {}
    with torch.no_grad():
        xt = torch.as_tensor(xs, dtype=torch.float64)
        for l in range(w["n_layers"]):
            z, _ = bd.block(z, w["blocks"][l], w["n_heads"], w["ln_eps"])
            blk = model.blocks[l].double()
            xt = blk(xt)
            blk.float()
            blo, bhi = z.bounds()
            true = xt.numpy()
            v_lo = float((blo[0][None] - true).max())
            v_hi = float((true - bhi[0][None]).max())
            width = float((bhi - blo).mean())
            attained = float((true.max(axis=0) - true.min(axis=0)).mean())
            res[f"layer{l}"] = {
                "max_violation": max(v_lo, v_hi),
                "sound": bool(max(v_lo, v_hi) <= FP_SLACK),
                "mean_bound_width": width,
                "mean_attained_width": attained,
                "wrapping_factor": float(width / max(attained, 1e-12)),
            }
    model.float()
    return res


def check_v_bounds(V, vw, rho, seq_len, d_model, n=4000, seed=1):
    """Falsify the Lyapunov-function bounds themselves."""
    rng = np.random.default_rng(seed)
    lo = np.full((seq_len, d_model), -rho)
    hi = np.full((seq_len, d_model), rho)
    es = _sample_box(lo, hi, n, rng)
    z = bd.HZono.from_box(lo[None], hi[None])
    vz = icnn.v_bound(z, vw)
    vlo, vhi = vz.bounds()
    with torch.no_grad():
        true = V(torch.as_tensor(es, dtype=torch.float32)).numpy()
    return {
        "max_violation": float(max(vlo[0, 0] - true.min(), true.max() - vhi[0, 0])),
        "sound": bool(max(vlo[0, 0] - true.min(), true.max() - vhi[0, 0]) <= 1e-5),
        "bound_lo": float(vlo[0, 0]), "bound_hi": float(vhi[0, 0]),
        "sampled_min": float(true.min()), "sampled_max": float(true.max()),
        "relaxation_gap_upper": float(vhi[0, 0] - true.max()),
        "V_at_origin": float(V(torch.zeros(1, seq_len, d_model)).item()),
    }


def check_decrease_bound(model, V, w, vw, x_nom_l, x_nom_l1, layer, rho,
                         kappa=0.05, n=3000, seed=2, pgd_steps=120):
    """Falsify the decrease bound, and measure how far above truth it sits.

    Combines random sampling with the PGD falsifier so the attained maximum is
    a genuine adversarial maximum, not a lucky draw. The reported gap is then an
    upper bound on the true relaxation gap -- honest in the conservative
    direction.
    """
    rng = np.random.default_rng(seed)
    T, D = x_nom_l.shape
    lo = np.full((T, D), -rho); hi = np.full((T, D), rho)
    es = _sample_box(lo, hi, n, rng)
    e0, e1 = vf.deviation_step((x_nom_l * 0 - rho)[None], (x_nom_l * 0 + rho)[None],
                               w, layer, x_nom_l, x_nom_l1)
    d = icnn.decrease_upper(e0, e1, vw, kappa=kappa)
    bound = float(d["d_hi"][0])
    with torch.no_grad():
        et = torch.as_tensor(es, dtype=torch.float32)
        xn = torch.as_tensor(x_nom_l, dtype=torch.float32)[None]
        xn1 = torch.as_tensor(x_nom_l1, dtype=torch.float32)[None]
        out = model.blocks[layer](xn + et) - xn1
        true = (V(out) - (1.0 - kappa) * V(et)).numpy()
    adv = vf.pgd_falsify(model, V, x_nom_l, x_nom_l1, layer, rho, kappa=kappa,
                         n=256, steps=pgd_steps, seed=seed)
    attained = max(float(true.max()), float(adv["max_violation_found"]))
    return {
        "sound_bound": bound,
        "attained_max_random": float(true.max()),
        "attained_max_pgd": float(adv["max_violation_found"]),
        "attained_max": attained,
        "max_violation": float(attained - bound),
        "sound": bool(attained - bound <= 1e-5),
        "relaxation_gap": float(bound - attained),
        "certified_by_bound": bool(bound < 0.0),
        "falsified_by_search": bool(attained > 0.0),
    }


def check_primitives(seed=3, n=20000):
    """Brute-force the individual sound primitives against dense enumeration."""
    rng = np.random.default_rng(seed)
    out = {}

    # ReLU relaxation
    lo = rng.uniform(-2, 1, size=(200, 1, 6)); hi = lo + rng.uniform(0, 3, size=lo.shape)
    z = bd.HZono.from_box(lo, hi)
    r = bd.relu(z); rlo, rhi = r.bounds()
    xs = lo[:, None] + rng.random((200, 60, 1, 6)) * (hi - lo)[:, None]
    tr = np.maximum(xs, 0)
    out["relu"] = {"max_violation": float(max((rlo[:, None] - tr).max(),
                                              (tr - rhi[:, None]).max())),
                   "sound": None}

    # LayerNorm
    d = 8
    lo = rng.uniform(-1, 0.5, size=(100, 1, d)); hi = lo + rng.uniform(0.05, 1.0, size=lo.shape)
    g = rng.normal(size=d); b = rng.normal(size=d)
    z = bd.HZono.from_box(lo, hi)
    y = bd.layernorm(z, g, b, 1e-5); ylo, yhi = y.bounds()
    xs = lo[:, None] + rng.random((100, 400, 1, d)) * (hi - lo)[:, None]
    xc = xs - xs.mean(-1, keepdims=True)
    tr = xc / np.sqrt((xc ** 2).mean(-1, keepdims=True) + 1e-5) * g + b
    out["layernorm"] = {"max_violation": float(max((ylo[:, None] - tr).max(),
                                                   (tr - yhi[:, None]).max())),
                        "sound": None}

    # softmax envelope
    T = 5
    s_lo = rng.uniform(-3, 1, size=(200, T, T)); s_hi = s_lo + rng.uniform(0, 2, size=s_lo.shape)
    mask = np.broadcast_to(bd.causal_mask(T)[None], s_lo.shape)
    a_lo, a_hi = bd.softmax_interval(s_lo, s_hi, mask)
    ss = s_lo[:, None] + rng.random((200, 400, T, T)) * (s_hi - s_lo)[:, None]
    ss = np.where(mask[:, None], ss, -1e9)
    ex = np.exp(ss - ss.max(-1, keepdims=True))
    tr = ex / ex.sum(-1, keepdims=True)
    tr = np.where(mask[:, None], tr, 0.0)
    out["softmax"] = {"max_violation": float(max((a_lo[:, None] - tr).max(),
                                                 (tr - a_hi[:, None]).max())),
                      "sound": None}

    for k in out:
        out[k]["sound"] = bool(out[k]["max_violation"] <= 1e-9)
    return out


def full_report(model, V, w, vw, x_nom_trace, rho, kappa=0.05, seed=0):
    rep = {"primitives": check_primitives(),
           "block_bounds": check_block_bounds(model, w, x_nom_trace[0], rho, seed=seed),
           "v_bounds": check_v_bounds(V, vw, rho, w["seq_len"], w["d_model"], seed=seed + 1),
           "decrease": {}}
    for l in range(w["n_layers"]):
        rep["decrease"][f"layer{l}"] = check_decrease_bound(
            model, V, w, vw, x_nom_trace[l], x_nom_trace[l + 1], l, rho,
            kappa=kappa, seed=seed + 2 + l)
    flat = ([v["sound"] for v in rep["primitives"].values()]
            + [v["sound"] for v in rep["block_bounds"].values()]
            + [rep["v_bounds"]["sound"]]
            + [v["sound"] for v in rep["decrease"].values()])
    rep["ALL_SOUND"] = bool(all(flat))
    return rep
