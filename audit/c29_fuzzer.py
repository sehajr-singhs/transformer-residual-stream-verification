"""c29 -- honest adversarial stress test of the certified fixnorm cell.

WHAT THIS DOES
---------------
Gumbel-Softmax discrete relaxation over the SEQ_LEN context tokens, trained
by gradient ascent to maximise the torch IBP relaxation gap (bound_width /
attained_width) of the trained, certified L=2/d=32 fixnorm model -- the exact
cell behind the "1.08x variance collapse, zero unstable ReLUs" headline.

WHY THE TORCH ENGINE, NOT THE NUMPY ZONOTOPE
----------------------------------------------
Every banked "relaxation_gap" number in this project (c24/c25/c26) comes from
the NumPy hybrid-zonotope prover in src/bounds.py via branch-and-bound. That
prover is not differentiable -- it is a discrete box-splitting search, not a
computation graph -- so a gradient-based (Gumbel-Softmax) adversary cannot
attack it directly. src/torch_bounds.py IS differentiable (it is literally
the training signal for certified training), so this fuzzer attacks that
engine instead, using the SAME bound_width/attained_width construction
c24_scaling.gap_on_weights uses on the NumPy side, just implemented in torch
so gradients can flow back into which context tokens were chosen.

WHAT THIS IS NOT
-----------------
This is not a soundness proof and does not attempt to be one. A sound prover
cannot be "broken" by an adversarial search; a failure to find a large gap is
a failure of THIS search (finite steps, finite restarts, one temperature
schedule), not evidence of universal immunity. Report whatever is found,
including a null result, as an empirical search-completeness statement.

SCOPE, DELIBERATELY NARROWED
------------------------------
Only the fixnorm+relu+standard-mlp architecture is attacked, because that is
the only architecture with an actual banked, NumPy-certified cell (c25). The
GeLU/SiLU golden-section engine in torch_bounds.py has NEVER been NumPy
zonotope certified -- arch.certifiable() skips it by construction (no sound
relaxation in src/bounds.py for those activations) -- so an OOD attack on it
would be attacking a code path this project has never actually claimed a
certificate for. Out of scope for this run; noted, not silently ignored.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "audit"))

import c24_scaling as c24          # noqa: E402
import c25_certified as c25c       # noqa: E402
from src import torch_bounds as tb  # noqa: E402

OUT = os.path.join(ROOT, "results", "c29_fuzz.json")


def build_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=0.05,
                     help="fuzz radius; matches c25's eps_train, NOT the "
                          "NumPy-certified rho -- these are different "
                          "quantities and must not be conflated")
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--fuzz-lr", type=float, default=0.3)
    ap.add_argument("--n-sample", type=int, default=2000,
                     help="samples used for the honest attained-width readout")
    ap.add_argument("--baseline-anchors", type=int, default=8,
                     help="natural TinyShakespeare contexts to compare against")
    return ap.parse_args(argv)


def forward_from_embed(model, x):
    for b in model.blocks:
        x = b(x)
    return model.unembed(model.nf(x))


def ibp_gap_differentiable(model, W, x0, eps, mode="fixnorm"):
    """bound_width, DIFFERENTIABLE w.r.t. x0 (so gradients reach the
    Gumbel-Softmax context logits that produced x0)."""
    r = torch.zeros_like(x0)
    r[:, -1, :] = eps
    lo, hi = x0 - r, x0 + r
    lo, hi = tb.propagate(lo, hi, W, mode)
    L_lo, L_hi = tb.readout(lo, hi, W, mode)
    return (L_hi - L_lo).max()


def honest_gap(model, W, toks, eps, mode="fixnorm", n_sample=2000, seed=0):
    """Exact (no relaxation) bound_width/attained_width on a DISCRETE context,
    mirroring c24_scaling.gap_on_weights but on the torch engine."""
    with torch.no_grad():
        x0 = model.embed_stream(toks)
        bound = float(ibp_gap_differentiable(model, W, x0, eps, mode))
        rng = np.random.default_rng(seed)
        delta = rng.uniform(-eps, eps, size=(n_sample, x0.shape[-1])).astype(np.float32)
        xs = x0.repeat(n_sample, 1, 1).clone()
        xs[:, -1, :] += torch.from_numpy(delta)
        out = forward_from_embed(model, xs)[:, -1, :]
        att = float((out.max(0).values - out.min(0).values).max())
    gap = bound / max(att, 1e-300)
    return {"bound_width": bound, "attained_width": att, "relaxation_gap": gap,
            "context": toks[0].tolist()}


def fuzz_once(model, W, seq_len, vocab, eps, steps, tau, lr, seed):
    torch.manual_seed(seed)
    theta = torch.zeros(seq_len, vocab, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    best_bound, best_tau_snap = -1.0, None
    for step in range(steps):
        y_soft = F.gumbel_softmax(theta, tau=tau, hard=False, dim=-1)
        tok_emb = y_soft @ model.embed.weight
        x0 = (tok_emb + model.pos[:seq_len])[None]
        bound = ibp_gap_differentiable(model, W, x0, eps)
        loss = -bound
        opt.zero_grad()
        loss.backward()
        opt.step()
        bv = float(bound.detach())
        if bv > best_bound:
            best_bound = bv
            best_tau_snap = theta.detach().argmax(-1).clone()
    return best_tau_snap, best_bound


def numpy_zonotope_check(model, d, rho, toks, seed=0):
    """Black-box cross-check on the ACTUAL certifying engine.

    Every banked gap number in this project (c24/c25/c26) comes from the
    NumPy hybrid-zonotope prover via branch-and-bound, not from
    src/torch_bounds.py. That prover is not differentiable, so the
    Gumbel-Softmax fuzzer above could not attack it directly -- it could only
    attack the differentiable torch IBP engine used as the training signal.
    This closes the loop: it evaluates the zonotope prover, black-box, on
    whatever context the differentiable fuzzer found, at the REAL certified
    rho=0.02 (not the eps_train=0.05 the fuzzer searched at -- those are
    different quantities throughout this project and must not be conflated).
    Also recovers the containment check for free: IBP/zonotope soundness is
    a property of interval arithmetic, unconditional on the input
    distribution, so containment holding here is expected, not surprising --
    but expected is not the same as verified, so it is verified anyway.
    """
    from src import bounds as bd
    import c18_variants as c18
    W = c24.export_W(model, d)
    emb = model.embed_stream(toks).detach().cpu().numpy().astype(np.float64)[0]
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(4, d))
    U = np.zeros((4, c24.SEQ_LEN, d))
    U[np.arange(4), c24.SEQ_LEN - 1, :] = dirs / np.linalg.norm(
        dirs, axis=1, keepdims=True)
    z = bd.HZono(emb[None], np.full((1, 4), rho)[:, :, None, None] * U[None])
    unstable = 0
    with np.errstate(over="ignore", invalid="ignore"):
        for l in range(W["n_layers"]):
            z, st = c18.block(z, W["blocks"][l], W["n_heads"], "fixnorm", c24.EPS)
            unstable += st["unstable"]
        bound = float(z.width().max())
        al = rng.uniform(-rho, rho, size=(2000, 4))
        xs = np.einsum("bk,ktd->btd", al, U) + emb[None]
        out = c18.exact_forward(W, "fixnorm", xs)
        att = float((out.max(0) - out.min(0)).max())
        lo, hi = z.bounds()
        viol = float(np.max(np.maximum(lo[0][None] - out, out - hi[0][None])))
    gap = bound / max(att, 1e-300) if np.isfinite(bound) else float("inf")
    return {"bound_width": bound, "attained_width": att, "relaxation_gap": gap,
            "gap_meaningful": bool(np.isfinite(gap) and gap <= 1e3),
            "unstable_relus": int(unstable), "max_containment_violation": viol,
            "rho": rho}


def main(argv=None):
    A = build_args(argv)
    torch.set_num_threads(4)
    t0 = time.time()

    def stamp(m):
        print(f"[{time.time() - t0:8.1f}s] {m}", flush=True)

    C = c24.load_corpus()
    stamp(f"corpus loaded, vocab={C['vocab']}")

    # ---- train the exact flagship certified cell (L=2, d=32, fixnorm, lr=3e-3) ----
    ca = c25c.build_args([
        "--layers", str(A.layers), "--widths", str(A.d), "--variants", "fixnorm",
        "--lrs", str(A.lr), "--seeds", "1", "--steps", "3000",
        "--ramp-start", "100", "--ramp-len", "1000", "--eps-train", "0.05",
        "--rho", "0.02",
    ])
    VX, VY = c24.sample(4096, np.random.default_rng(12345), C["val"])
    TX, TY = c24.sample(2048, np.random.default_rng(999), C["val"])
    C["VX"], C["VY"] = torch.from_numpy(VX), torch.from_numpy(VY)
    C["TX"], C["TY"] = torch.from_numpy(TX), torch.from_numpy(TY)
    model, rec = c25c.train_certified("fixnorm", A.train_seed, A.d, A.layers,
                                       A.lr, C, ca, "cpu")
    stamp(f"trained: ppl={rec['val_ppl']:.4f} diverged={rec['diverged']}")
    if rec["diverged"]:
        stamp("TRAINING DIVERGED -- nothing to fuzz. Recording and exiting.")
        json.dump({"config": vars(A), "training": rec, "fuzz": None},
                   open(OUT, "w"), indent=2)
        return 0
    model.eval()
    W = c25c.live_W(model)

    # ---- baseline: honest gap on natural TinyShakespeare contexts ----
    rng = np.random.default_rng(777)
    baselines = []
    for i in range(A.baseline_anchors):
        x, _ = c24.sample(1, rng, C["val"])
        toks = torch.from_numpy(x)
        g = honest_gap(model, W, toks, A.eps, n_sample=A.n_sample, seed=i)
        g["prompt"] = "".join(C["chars"][t] for t in x[0])
        baselines.append(g)
        stamp(f"  baseline[{i}] gap={g['relaxation_gap']:.4g} "
              f"prompt={g['prompt']!r}")
    base_gaps = [b["relaxation_gap"] for b in baselines]
    stamp(f"baseline gap: median={np.median(base_gaps):.4g} "
          f"max={np.max(base_gaps):.4g}")

    # ---- adversarial search: multiple restarts, snap to hard tokens, verify ----
    found = []
    for r in range(A.restarts):
        best_toks, best_soft_bound = fuzz_once(
            model, W, c24.SEQ_LEN, C["vocab"], A.eps, A.steps, A.tau,
            A.fuzz_lr, seed=1000 + r)
        toks = best_toks[None]
        g = honest_gap(model, W, toks, A.eps, n_sample=A.n_sample, seed=2000 + r)
        g["prompt"] = "".join(C["chars"][t] for t in best_toks.tolist())
        g["soft_bound_at_snap"] = best_soft_bound
        found.append(g)
        stamp(f"  restart[{r}] honest gap={g['relaxation_gap']:.4g} "
              f"(soft objective saw {best_soft_bound:.4g}) "
              f"prompt={g['prompt']!r}")

    adv_gaps = [f["relaxation_gap"] for f in found]
    best_adv = found[int(np.argmax(adv_gaps))]
    verdict = {
        "baseline_median_gap": float(np.median(base_gaps)),
        "baseline_max_gap": float(np.max(base_gaps)),
        "adversarial_max_gap": float(np.max(adv_gaps)),
        "adversarial_median_gap": float(np.median(adv_gaps)),
        "ratio_adv_max_over_baseline_median": float(np.max(adv_gaps) / np.median(base_gaps)),
        "exceeded_gap_meaningful_1e3": bool(np.max(adv_gaps) > 1e3),
        "n_restarts": A.restarts, "n_steps_per_restart": A.steps,
    }
    stamp(f"VERDICT: adv max {verdict['adversarial_max_gap']:.4g} vs "
          f"baseline median {verdict['baseline_median_gap']:.4g} "
          f"({verdict['ratio_adv_max_over_baseline_median']:.2f}x)")

    # ---- does this transfer to the ACTUAL certifying engine? ----
    nat_toks = torch.tensor([baselines[0]["context"]], dtype=torch.long)
    adv_toks = torch.tensor([best_adv["context"]], dtype=torch.long)
    nz_nat = numpy_zonotope_check(model, A.d, 0.02, nat_toks, seed=0)
    nz_adv = numpy_zonotope_check(model, A.d, 0.02, adv_toks, seed=0)
    stamp(f"NumPy zonotope cross-check @ rho=0.02: "
          f"natural gap={nz_nat['relaxation_gap']:.4g} "
          f"(unstable={nz_nat['unstable_relus']}, "
          f"viol={nz_nat['max_containment_violation']:+.1e}) vs "
          f"adversarial gap={nz_adv['relaxation_gap']:.4g} "
          f"(unstable={nz_adv['unstable_relus']}, "
          f"viol={nz_adv['max_containment_violation']:+.1e})")

    json.dump({
        "config": vars(A), "training": {k: v for k, v in rec.items() if k != "curve"},
        "baselines": baselines, "adversarial_restarts": found,
        "verdict": verdict,
        "numpy_zonotope_crosscheck": {
            "note": "black-box eval of the ACTUAL certifying engine (not "
                    "attacked directly -- not differentiable) on the "
                    "torch-IBP fuzzer's winning context, at the real "
                    "certified rho=0.02",
            "natural_context": baselines[0]["prompt"], "natural": nz_nat,
            "adversarial_context": best_adv["prompt"], "adversarial": nz_adv,
        },
    }, open(OUT, "w"), indent=2, default=str)
    stamp(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
