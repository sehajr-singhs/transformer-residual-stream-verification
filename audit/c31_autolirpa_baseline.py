# -*- coding: utf-8 -*-
"""c31: real CROWN baseline via auto_LiRPA (Verified-Intelligence), compared
against this project's own torch IBP engine (src/torch_bounds.py), on the
SAME trained model and SAME threat model: a full L_inf box of radius rho on
the final residual-stream position, which is what certified training in
c25/c27 actually optimises against (a documented superset of the 4-dim
steering subspace the NumPy zonotope prover certifies -- this script does
NOT reproduce that zonotope number, it compares the two training-signal-
style engines honestly on their own shared threat model).
"""
import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r"C:\Users\sehaj\experiments\ai_safety_lyapunov")
sys.path.insert(0, r"C:\Users\sehaj\experiments\ai_safety_lyapunov\audit")
import numpy as np
import torch
import torch.nn as nn

import c24_scaling as c24
import c25_certified as c25c
from src import torch_bounds as tb

RESULTS = r"C:\Users\sehaj\experiments\ai_safety_lyapunov\results\c31_autolirpa.json"


class TailModel(nn.Module):
    """Embedding stream -> logits, the exact segment the threat model
    perturbs (mirrors c29_fuzzer.forward_from_embed)."""
    def __init__(self, model):
        super().__init__()
        self.blocks = model.blocks
        self.nf = model.nf
        self.unembed = model.unembed

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.unembed(self.nf(x))


def our_ibp_bound(model, x0, rho):
    W = c25c.live_W(model)
    r = torch.zeros_like(x0)
    r[:, -1, :] = rho
    lo, hi = x0 - r, x0 + r
    lo, hi = tb.propagate(lo, hi, W, "fixnorm")
    L_lo, L_hi = tb.readout(lo, hi, W, "fixnorm")
    return float((L_hi - L_lo).max())


def attained_width(tail, x0, rho, n_sample=2000, seed=0):
    rng = np.random.default_rng(seed)
    delta = rng.uniform(-rho, rho, size=(n_sample, x0.shape[-1])).astype(np.float32)
    xs = x0.repeat(n_sample, 1, 1).clone()
    xs[:, -1, :] += torch.from_numpy(delta)
    with torch.no_grad():
        out = tail(xs)[:, -1, :]
    return float((out.max(0).values - out.min(0).values).max())


def main():
    torch.set_num_threads(4)
    t0 = __import__("time").time()

    def stamp(m):
        print(f"[{__import__('time').time()-t0:8.1f}s] {m}", flush=True)

    C = c24.load_corpus()
    VX, VY = c24.sample(4096, np.random.default_rng(12345), C["val"])
    TX, TY = c24.sample(2048, np.random.default_rng(999), C["val"])
    C["VX"], C["VY"] = torch.from_numpy(VX), torch.from_numpy(VY)
    C["TX"], C["TY"] = torch.from_numpy(TX), torch.from_numpy(TY)
    stamp("corpus loaded")

    ca = c25c.build_args([
        "--layers", "2", "--widths", "32", "--variants", "fixnorm",
        "--lrs", "3e-3", "--seeds", "1", "--steps", "3000",
        "--ramp-start", "100", "--ramp-len", "1000",
        "--eps-train", "0.05", "--rho", "0.02",
    ])
    model, rec = c25c.train_certified("fixnorm", 0, 32, 2, 3e-3, C, ca, "cpu")
    stamp(f"trained: ppl={rec['val_ppl']:.4f} diverged={rec['diverged']}")
    if rec["diverged"]:
        json.dump({"error": "training diverged"}, open(RESULTS, "w"))
        return 1
    model.eval()
    tail = TailModel(model)

    rho = 0.02
    results = []
    for anchor_seed in range(2):
        x_full, _ = c24.sample(1, np.random.default_rng(700 + anchor_seed), C["val"])
        toks = torch.from_numpy(x_full)
        with torch.no_grad():
            x0 = model.embed_stream(toks)
        stamp(f"  anchor {anchor_seed}: embedded, starting our_ibp_bound")

        ours = our_ibp_bound(model, x0, rho)
        stamp(f"  anchor {anchor_seed}: our_ibp done ({ours:.4g}), "
              f"starting attained_width sampling")
        att = attained_width(tail, x0, rho, seed=anchor_seed)
        stamp(f"  anchor {anchor_seed}: attained_width done ({att:.4g}), "
              f"constructing BoundedModule")

        try:
            from auto_LiRPA import BoundedModule, BoundedTensor
            from auto_LiRPA.perturbations import PerturbationLpNorm
            lirpa_model = BoundedModule(tail, x0, device="cpu",
                                         bound_opts={"conv_mode": "matrix"})
            stamp(f"  anchor {anchor_seed}: BoundedModule built, "
                  f"calling compute_bounds(method=CROWN)")
            r = torch.zeros_like(x0)
            r[:, -1, :] = rho
            ptb = PerturbationLpNorm(norm=float("inf"), x_L=x0 - r, x_U=x0 + r)
            bx = BoundedTensor(x0, ptb)
            with torch.no_grad():
                lb, ub = lirpa_model.compute_bounds(x=(bx,), method="CROWN")
            crown = float((ub - lb).max())
            crown_err = None
            stamp(f"  anchor {anchor_seed}: CROWN done, gap width={crown:.4g}")
        except Exception as e:  # noqa: BLE001
            crown = None
            crown_err = repr(e)
            stamp(f"  anchor {anchor_seed}: CROWN FAILED: {crown_err}")

        row = {
            "anchor_seed": anchor_seed,
            "prompt": "".join(C["chars"][t] for t in x_full[0]),
            "our_ibp_bound_width": ours,
            "attained_width": att,
            "our_ibp_gap": ours / att,
            "crown_bound_width": crown,
            "crown_gap": (crown / att) if crown is not None else None,
            "crown_error": crown_err,
        }
        results.append(row)
        stamp(f"  anchor {anchor_seed}: our_ibp_gap={row['our_ibp_gap']:.3g} "
              f"crown_gap={row['crown_gap']} err={crown_err}")

    json.dump({"config": {"rho": rho, "d_model": 32, "n_layers": 2,
                           "variant": "fixnorm", "lr": 3e-3},
               "training": {k: v for k, v in rec.items() if k != "curve"},
               "results": results}, open(RESULTS, "w"), indent=2, default=str)
    stamp(f"wrote {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
