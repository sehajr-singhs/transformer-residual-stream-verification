"""c25 -- validate src/torch_bounds.py.

Two independent checks, because they catch different failures:

  AGREEMENT   the torch IBP engine must reproduce audit/ibp_ref.py, which was
              transcribed from toy_transformer.forward and shares no code with
              the prover. Same mathematical object, so they must agree to
              float64 round-off. Tolerance 1e-12 relative.

  CONTAINMENT sampled forward passes through the real function must lie inside
              the returned box. This is the only check available for the
              `fixnorm` variant, which ibp_ref does not implement, and it is
              the check that would catch a bound that is self-consistent but
              unsound.

AGREEMENT alone is not enough: both engines could share a sign error. Sampling
alone is not enough either: it cannot prove tightness, only detect gross
unsoundness. Neither replaces src/soundness.py for the zonotope prover.
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "audit"))

import ibp_ref  # noqa: E402
from src import torch_bounds as tb  # noqa: E402

torch.set_default_dtype(torch.float64)
DT = torch.float64


def synth(n_layers, d_model, n_heads, seed, vocab=65, d_ff_mult=4):
    """Random weights in BOTH representations, from one rng, so the numpy and
    torch engines see bit-identical inputs."""
    rng = np.random.default_rng(seed)
    s = 1.0 / np.sqrt(d_model)

    def R(*shape):
        return rng.normal(0.0, s, size=shape)

    blocks = []
    for _ in range(n_layers):
        blocks.append({
            "ln1_g": rng.normal(1.0, 0.1, d_model),
            "ln1_b": rng.normal(0.0, 0.1, d_model),
            "WQ": R(d_model, d_model), "WK": R(d_model, d_model),
            "WV": R(d_model, d_model), "WO": R(d_model, d_model),
            "ln2_g": rng.normal(1.0, 0.1, d_model),
            "ln2_b": rng.normal(0.0, 0.1, d_model),
            "fc_in_W": R(d_ff_mult * d_model, d_model),
            "fc_in_b": rng.normal(0.0, 0.1, d_ff_mult * d_model),
            "fc_out_W": R(d_model, d_ff_mult * d_model),
            "fc_out_b": rng.normal(0.0, 0.1, d_model),
            # c18 calibration: nominal RMS at the site. Random init has unit
            # scale by construction, so 1.0 is the right stand-in here.
            "scale1": 1.0, "scale2": 1.0,
        })
    w = {"n_layers": n_layers, "n_heads": n_heads, "ln_eps": 1e-5,
         "blocks": blocks,
         "ln_f_g": rng.normal(1.0, 0.1, d_model),
         "ln_f_b": rng.normal(0.0, 0.1, d_model),
         "scale_f": 1.0,
         "unembed": R(vocab, d_model)}

    def to_t(o):
        if isinstance(o, dict):
            return {k: to_t(v) for k, v in o.items()}
        if isinstance(o, list):
            return [to_t(v) for v in o]
        if isinstance(o, np.ndarray):
            return torch.tensor(o, dtype=DT)
        return o

    return w, to_t(w)


def forward_torch(x, wt, variant):
    """The real function the bounds must enclose. x: (B, T, D)."""
    for l in range(wt["n_layers"]):
        bw = wt["blocks"][l]
        B, T, D = x.shape
        H, dh = wt["n_heads"], D // wt["n_heads"]

        def nrm(v, g, b, sc):
            u = v - v.mean(-1, keepdim=True)
            if variant == "fixnorm":
                return g * u / sc + b
            var = (u * u).mean(-1, keepdim=True)
            return g * u / torch.sqrt(var + wt["ln_eps"]) + b

        n = nrm(x, bw["ln1_g"], bw["ln1_b"], bw["scale1"])
        q = (n @ bw["WQ"].T).view(B, T, H, dh).transpose(1, 2)
        k = (n @ bw["WK"].T).view(B, T, H, dh).transpose(1, 2)
        v = (n @ bw["WV"].T).view(B, T, H, dh).transpose(1, 2)
        sc = q @ k.transpose(-1, -2) / np.sqrt(dh)
        m = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        sc = sc.masked_fill(m, float("-inf"))
        o = torch.softmax(sc, -1) @ v
        x = x + (o.transpose(1, 2).reshape(B, T, D) @ bw["WO"].T)

        n = nrm(x, bw["ln2_g"], bw["ln2_b"], bw["scale2"])
        h = torch.relu(n @ bw["fc_in_W"].T + bw["fc_in_b"])
        x = x + (h @ bw["fc_out_W"].T + bw["fc_out_b"])

    y = x - x.mean(-1, keepdim=True)
    if variant == "fixnorm":
        y = wt["ln_f_g"] * y / wt["scale_f"] + wt["ln_f_b"]
    else:
        var = (y * y).mean(-1, keepdim=True)
        y = wt["ln_f_g"] * y / torch.sqrt(var + wt["ln_eps"]) + wt["ln_f_b"]
    return y @ wt["unembed"].T


def reldev(a, b):
    scale = np.maximum(np.abs(a), np.abs(b))
    scale = np.where(scale > 1.0, scale, 1.0)
    return float(np.max(np.abs(a - b) / scale))


def main():
    T, RHO = 8, 1e-3
    grid = [(2, 32, 4), (2, 64, 4), (4, 32, 4), (4, 64, 8), (8, 32, 4)]
    tol = 1e-12

    print(f"{'cell':>14} {'rel dev lo':>12} {'rel dev hi':>12}  agree")
    worst = 0.0
    for (L, d, H) in grid:
        for seed in (0, 1):
            w, wt = synth(L, d, H, seed)
            rng = np.random.default_rng(1000 + seed)
            x0 = rng.normal(0.0, 1.0, (T, d))
            lo_n, hi_n = x0 - RHO, x0 + RHO

            n_lo, n_hi, _ = ibp_ref.propagate(lo_n, hi_n, w)
            n_lo, n_hi = ibp_ref.readout(n_lo, n_hi, w)

            t_lo = torch.tensor(lo_n, dtype=DT)[None]
            t_hi = torch.tensor(hi_n, dtype=DT)[None]
            p_lo, p_hi = tb.propagate(t_lo, t_hi, wt, "standard")
            p_lo, p_hi = tb.readout(p_lo, p_hi, wt, "standard")

            dl = reldev(n_lo, p_lo[0].numpy())
            dh = reldev(n_hi, p_hi[0].numpy())
            worst = max(worst, dl, dh)
            ok = "OK" if max(dl, dh) <= tol else "FAIL"
            print(f"L={L:2d} d={d:3d} s={seed}  {dl:12.3e} {dh:12.3e}  {ok}")

    print(f"\nAGREEMENT vs ibp_ref: worst {worst:.3e} (tol {tol:.0e}) "
          f"-> {'PASS' if worst <= tol else 'FAIL'}")

    # ---------------------------------------------------------- containment
    print(f"\n{'cell':>14} {'variant':>9} {'max viol':>12}  sound")
    worst_v = 0.0
    for (L, d, H) in grid:
        for variant in ("standard", "fixnorm"):
            w, wt = synth(L, d, H, seed=7)
            rng = np.random.default_rng(99)
            x0 = torch.tensor(rng.normal(0.0, 1.0, (1, T, d)), dtype=DT)
            lo, hi = x0 - RHO, x0 + RHO
            b_lo, b_hi = tb.propagate(lo, hi, wt, variant)
            b_lo, b_hi = tb.readout(b_lo, b_hi, wt, variant)

            u = torch.rand(256, T, d, dtype=DT)
            xs = lo + u * (hi - lo)
            xs = torch.cat([xs, lo, hi], 0)
            out = forward_torch(xs, wt, variant)
            viol = float(torch.max(torch.maximum(b_lo - out, out - b_hi)))
            worst_v = max(worst_v, viol)
            print(f"L={L:2d} d={d:3d}   {variant:>9} {viol:12.3e}  "
                  f"{'OK' if viol <= 1e-9 else 'VIOLATION'}")

    print(f"\nCONTAINMENT: worst violation {worst_v:.3e} "
          f"-> {'PASS' if worst_v <= 1e-9 else 'FAIL'}")
    return 0 if (worst <= tol and worst_v <= 1e-9) else 1


if __name__ == "__main__":
    sys.exit(main())
