"""c25 -- certified-by-construction training on TinyShakespeare.

WHAT THIS DOES
--------------
Trains the c18/c19/c23/c24 LM variants with an IBP robust-loss term in the
objective (Gowal et al. 2018 elision + kappa/eps ramp), then certifies the
result with the UNCHANGED zonotope prover so the numbers stay comparable to
every banked cell.

THREE THINGS THAT ARE EASY TO GET WRONG HERE
--------------------------------------------
1. TWO DIFFERENT THREAT MODELS, DELIBERATELY.
   The certificate this project reports is a 4-direction subspace on the final
   residual position (see c24_scaling.gap_on_weights). IBP cannot represent a
   k-dimensional subspace -- ibp_ref's own docstring notes it "loses the
   k-dimensional subspace structure of the threat model at the input". So we
   TRAIN against an L_inf box on that same final position, which is a strict
   SUPERSET of the subspace ball and therefore a sound regulariser, and we
   CERTIFY with the untouched zonotope machinery on the original subspace.
   The training eps and the certified rho are NOT the same quantity and must
   never be plotted on a shared axis.

2. THE IBP BOUND IS NOT THE CERTIFICATE.
   src/torch_bounds.py is plain interval arithmetic, orders of magnitude looser
   than src/bounds.py. It is a training signal, nothing else. Every certified
   number in the output JSON comes from c24_scaling.gap_on_weights, i.e. from
   the NumPy hybrid-zonotope engine, which c25 does not modify.

3. FIXNORM SCALES ARE SELF-CALIBRATED, NOT LOADED.
   Norm.set_calibrating fills the scale buffer from the model's own activation
   RMS on one batch before training. Scales are per-site and per-architecture;
   importing c23's (which exist only for L=2, d in {32,64}, and were never
   exported to disk anyway) would inject a wrong constant. We calibrate the
   model being trained, exactly as c24 does, so c25 cells stay comparable.

src/bounds.py is untouched. c24_scaling.py is imported, not edited, so its
fingerprint d22561792a22e50a and every banked cell stay valid.
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

import c24_scaling as c24  # noqa: E402  (importable: argv parsed inside main)
import c25_arch as arch  # noqa: E402
from src import torch_bounds as tb  # noqa: E402

PARTS = os.path.join(ROOT, "results", "c25_certified_parts")


def build_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="2,4")
    ap.add_argument("--widths", default="32,64")
    ap.add_argument("--variants", default="standard,fixnorm")
    ap.add_argument("--lrs", default="1e-3,3e-3,1e-2")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    # certified-training schedule
    ap.add_argument("--eps-train", type=float, default=0.05,
                    help="final L_inf radius on the last residual position")
    ap.add_argument("--ramp-start", type=int, default=100)
    ap.add_argument("--ramp-len", type=int, default=1000)
    ap.add_argument("--kappa-final", type=float, default=0.5)
    ap.add_argument("--rho", type=float, default=0.02,
                    help="certification radius for the zonotope prover")
    # c26 architecture ablation. Defaults reproduce the c24/c25 model exactly:
    # when all three are default the ORIGINAL c24.LMVariant is used, so banked
    # cells stay bit-comparable and the fingerprint is untouched.
    ap.add_argument("--norm", default="standard",
                    choices=arch.NORMS,
                    help="capnorm floors the RMS denominator -> Lipschitz gain")
    ap.add_argument("--activation", default="relu", choices=arch.ACTS)
    ap.add_argument("--mlp", default="standard", choices=arch.MLPS)
    ap.add_argument("--norm-floor", type=float, default=0.5,
                    help="capnorm denominator floor; gain is capped at 1/floor")
    ap.add_argument("--out", default=None)
    ap.add_argument("--parts", default=None,
                    help="output dir; the cell tag does NOT encode eps_train or "
                         "rho, so a control pass MUST write somewhere else or "
                         "it will silently collide with the main grid")
    ap.add_argument("--pilot", type=int, default=0,
                    help="if >0, run this many steps and only time them")
    return ap.parse_args(argv)


# ------------------------------------------------------------------ live view

def live_W(model):
    """The weight dict torch_bounds consumes, holding LIVE parameters.

    c24.export_W detaches to float64 numpy, which is right for the prover and
    useless here: the robust loss has to backprop into these tensors.
    """
    blocks = []
    for b in model.blocks:
        d = {"ln1_g": b.n1.g, "ln1_b": b.n1.b, "scale1": b.n1.scale,
             "ln2_g": b.n2.g, "ln2_b": b.n2.b, "scale2": b.n2.scale,
             "WQ": b.WQ.weight, "WK": b.WK.weight,
             "WV": b.WV.weight, "WO": b.WO.weight}
        if getattr(b, "mlp_kind", "standard") == "swiglu":
            d.update({"W_gate": b.fc_gate.weight, "b_gate": b.fc_gate.bias,
                      "W_up": b.fc_up.weight, "b_up": b.fc_up.bias,
                      "W_down": b.fc_down.weight, "b_down": b.fc_down.bias})
        else:
            d.update({"fc_in_W": b.fc_in.weight, "fc_in_b": b.fc_in.bias,
                      "fc_out_W": b.fc_out.weight, "fc_out_b": b.fc_out.bias})
        blocks.append(d)
    return {"n_layers": model.n_layers, "n_heads": 4, "ln_eps": c24.EPS,
            "blocks": blocks,
            "ln_f_g": model.nf.g, "ln_f_b": model.nf.b,
            "scale_f": model.nf.scale, "unembed": model.unembed.weight}


def robust_ce(model, toks, y, eps, mode, act="relu", mlp_kind="standard",
              floor=None):
    """Upper bound on cross-entropy over an L_inf box of radius eps applied to
    the FINAL residual position of the embedding stream.

    Perturbing only the last position matches where the certificate lives, and
    keeps the box from being a statement about the whole context window.
    """
    x0 = model.embed_stream(toks)
    r = torch.zeros_like(x0)
    r[:, -1, :] = eps
    lo, hi = x0 - r, x0 + r
    W = live_W(model)
    lo, hi = tb.propagate(lo, hi, W, mode, act, mlp_kind, floor)
    L_lo, L_hi = tb.readout(lo, hi, W, mode, floor)
    V = L_lo.shape[-1]
    z = tb.worst_case_logits(L_lo.reshape(-1, V), L_hi.reshape(-1, V),
                             y.reshape(-1))
    return F.cross_entropy(z, y.reshape(-1))


def schedule(step, A):
    """Gowal-style ramp. At step 0 this is exactly the c24 objective, which is
    what makes a c25 cell reduce to its c24 counterpart when eps_train=0."""
    t = (step - A.ramp_start) / max(1, A.ramp_len)
    f = float(min(1.0, max(0.0, t)))
    return A.eps_train * f, 1.0 - (1.0 - A.kappa_final) * f


def train_certified(variant, seed, d, L, lr, C, A, dev):
    torch.manual_seed(seed)
    np.random.seed(seed)
    stock = (A.norm == "standard" and A.activation == "relu"
             and A.mlp == "standard")
    if stock:
        # Bit-for-bit the c24/c25 model. Do not route the default path through
        # c25_arch: a different module means a different parameter creation
        # order and therefore different weights under the same seed.
        model = c24.LMVariant(variant, d, C["vocab"], n_layers=L).to(dev)
        mode = "standard" if variant == "standard" else "fixnorm"
    else:
        mode = A.norm if variant == "standard" else "fixnorm"
        model = arch.ArchLM(d, C["vocab"], c24.SEQ_LEN, n_layers=L, norm=mode,
                            act=A.activation, mlp=A.mlp,
                            floor=A.norm_floor).to(dev)
    rng = np.random.default_rng(seed)

    x0, _ = c24.sample(A.batch, rng, C["train"])
    model.set_calibrating(True)
    with torch.no_grad():
        model(torch.from_numpy(x0).to(dev))
    model.set_calibrating(False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, A.warmup)))

    curve, diverged = [], False
    n = A.pilot if A.pilot else A.steps
    t0 = time.time()
    for step in range(n):
        x, y = c24.sample(A.batch, rng, C["train"])
        xt = torch.from_numpy(x).to(dev)
        yt = torch.from_numpy(y).to(dev)
        eps, kappa = schedule(step, A)

        logits = model(xt)
        ce = F.cross_entropy(logits.reshape(-1, C["vocab"]), yt.reshape(-1))
        if eps > 0.0 and kappa < 1.0:
            rce = robust_ce(model, xt, yt, eps, mode, A.activation, A.mlp,
                            A.norm_floor)
            loss = kappa * ce + (1.0 - kappa) * rce
        else:
            rce = torch.zeros((), device=ce.device)
            loss = ce

        opt.zero_grad()
        loss.backward()
        gn = float(torch.sqrt(sum((p.grad ** 2).sum()
                                  for p in model.parameters()
                                  if p.grad is not None)))
        if A.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), A.clip)
        opt.step()
        sched.step()

        if not np.isfinite(float(loss.detach())):
            diverged = True
            break
        if step % A.eval_every == 0 or step == n - 1:
            vce = c24.evaluate(model, C["TX"], C["TY"], C["vocab"])
            curve.append({"step": step, "eps": eps, "kappa": kappa,
                          "clean_ce": float(ce.detach()),
                          "robust_ce": float(rce.detach()),
                          "grad_norm": gn, "val_ce": vce,
                          "val_ppl": float(np.exp(vce))})
    wall = time.time() - t0

    vce = c24.evaluate(model, C["VX"], C["VY"], C["vocab"])
    return model, {
        "variant": variant, "seed": seed, "d_model": d, "n_layers": L,
        "arch": {"norm": A.norm, "activation": A.activation, "mlp": A.mlp,
                 "norm_floor": A.norm_floor, "stock": stock},
        "lr": lr, "eps_train": A.eps_train, "kappa_final": A.kappa_final,
        "steps_run": n, "curve": curve, "diverged": diverged,
        "val_ce": vce, "val_ppl": float(np.exp(vce)),
        "val_bpc": vce / np.log(2.0),
        "train_seconds": wall, "sec_per_step": wall / max(1, n),
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }


def main(argv=None):
    A = build_args(argv)
    torch.set_num_threads(A.threads)
    global PARTS
    if A.parts:
        PARTS = A.parts if os.path.isabs(A.parts) else os.path.join(
            ROOT, "results", A.parts)
    os.makedirs(PARTS, exist_ok=True)
    dev = c24.pick_device(A.device)
    C = c24.load_corpus()
    VX, VY = c24.sample(4096, np.random.default_rng(12345), C["val"])
    TX, TY = c24.sample(2048, np.random.default_rng(999), C["val"])
    C["VX"], C["VY"] = torch.from_numpy(VX).to(dev), torch.from_numpy(VY).to(dev)
    C["TX"], C["TY"] = torch.from_numpy(TX).to(dev), torch.from_numpy(TY).to(dev)

    t00 = time.time()

    def stamp(m):
        print(f"[{time.time() - t00:9.1f}s] {m}", flush=True)

    stamp(f"device={dev} eps_train={A.eps_train} ramp={A.ramp_start}"
          f"+{A.ramp_len} kappa_final={A.kappa_final} vocab={C['vocab']}")

    Ls = [int(x) for x in A.layers.split(",")]
    ds = [int(x) for x in A.widths.split(",")]
    vs = A.variants.split(",")
    lrs = [float(x) for x in A.lrs.split(",")]

    for L in Ls:
        for d in ds:
            for v in vs:
                for lr in lrs:
                    for s in range(A.seeds):
                        tag = f"L{L}_d{d}_{v}_lr{lr:g}_s{s}"
                        pj = os.path.join(PARTS, tag + ".json")
                        if os.path.exists(pj) and not A.pilot:
                            stamp(f"  {tag} CACHED")
                            continue
                        model, rec = train_certified(v, s, d, L, lr, C, A, dev)
                        stamp(f"  {tag} ppl={rec['val_ppl']:.4f} "
                              f"div={rec['diverged']} "
                              f"{rec['sec_per_step']*1000:.0f}ms/step "
                              f"[{rec['train_seconds']:.0f}s]")
                        if A.pilot:
                            continue
                        # A diverged model has NaN weights; certifying it costs
                        # minutes and yields a meaningless gap. Record the
                        # divergence as the result, which is the finding.
                        if rec["diverged"] or not np.isfinite(rec["val_ppl"]):
                            rec["certified"] = {"skipped": "diverged"}
                            json.dump(rec, open(pj, "w"), indent=2)
                            continue
                        # The NumPy prover implements ReLU + LayerNorm/fixnorm
                        # only. Handing it GeLU/SiLU/SwiGLU/capnorm weights
                        # would certify a DIFFERENT function than was trained.
                        can, why = arch.certifiable(A.norm, A.activation, A.mlp)
                        if not can:
                            rec["certified"] = {"skipped": why}
                            stamp(f"  {tag} certification skipped: {why}")
                            json.dump(rec, open(pj, "w"), indent=2)
                            continue
                        # certify with the UNCHANGED NumPy zonotope prover
                        W = c24.export_W(model, d)
                        W["__emb__"] = model.embed_stream(
                            C["VX"][:1]).detach().cpu().numpy().astype(
                                np.float64)[0]
                        try:
                            g = c24.gap_on_weights(W, v, d, A.rho, C, seed=s)
                        except Exception as e:  # noqa: BLE001
                            g = {"error": repr(e)}
                        rec["certified"] = g
                        if "relaxation_gap" in g:
                            stamp(f"  {tag} gap={g['relaxation_gap']:.4g} "
                                  f"meaningful={g['gap_meaningful']} "
                                  f"unstable={g['unstable_relus']} "
                                  f"viol={g['max_containment_violation']:+.1e}")
                        json.dump(rec, open(pj, "w"), indent=2)
    stamp(f"done: {len(os.listdir(PARTS))} parts in {PARTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
