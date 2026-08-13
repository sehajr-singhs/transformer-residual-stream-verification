"""c18 -- do architectural changes flatten the relaxation-explosion curve?

c17 measured a width scaling wall on standard blocks: relaxation amplification
28.9 -> 650 -> 4.24e9 for d_model 32/64/128 at rho=1e-4, with two mechanisms
visibly engaged, saturated unstable ReLUs and a degraded LayerNorm variance
bracket. This isolates each mechanism by construction rather than by argument.

Variants
--------
standard   pre-LN + ReLU. The baseline, matching src/toy_transformer.
fixnorm    LayerNorm's data-dependent 1/sqrt(var+eps) replaced by a FIXED
           constant scale: y = g*(x - mean)/c + b. This map is exactly linear,
           so its relaxation error is identically zero -- not small, zero. Any
           remaining explosion cannot be attributed to normalization.
fixnorm_tanh
           fixnorm plus tanh in place of ReLU. tanh is smooth with a globally
           bounded second derivative, so its secant relaxation error is
           O((u-l)^2) rather than ReLU's O(u-l) kink at every unstable unit.

Soundness of the new primitives is checked by sampling in `verify_sound()`
before any scaling number is reported. A faster engine that is not sound
measures nothing.

Everything here uses RANDOMLY INITIALISED weights. That is adequate for the
question asked -- how relaxation width propagates as a function of architecture
and d_model -- and inadequate for any claim about trained models, task
performance, or certified radius on a real network. Those claims are not made.
"""
import sys, os, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src import bounds as bd, task

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ABS = np.abs

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="32,64,128,256")
    ap.add_argument("--rhos", default="1e-4,1e-3,1e-2")
    ap.add_argument("--variants", default="standard,fixnorm,fixnorm_tanh")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "c18.json"))
    return ap.parse_args()


t0 = time.time()
def stamp(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

# max |tanh''| = 4/(3*sqrt(3)); used for a rigorous secant-deviation bound
TANH_D2_MAX = 4.0 / (3.0 * np.sqrt(3.0))


# ------------------------------------------------------------- new primitives

def fixnorm(z, g, b, scale):
    """y = g * (x - mean(x)) / scale + b with `scale` a CONSTANT.

    Centering is the same exact projector LayerNorm uses; dividing by a
    constant is exact too. The whole map is affine, so it introduces no
    relaxation error whatsoever -- this is the point of the variant.
    """
    d = z.S[-1]
    P0 = np.eye(d) - np.ones((d, d)) / d
    xc = z.linear(P0)
    s = (g / scale)
    return bd.HZono(xc.c * s + b, xc.G * s, xc.E * _ABS(s))


def tanh_relax(z):
    """Sound secant relaxation of tanh.

    On [l, u] take the secant slope lam and intercept beta. Since |tanh''| is
    bounded by K = 4/(3*sqrt(3)), the deviation of tanh from its secant is at
    most K (u-l)^2 / 8, which goes into the interval remainder E. Unlike ReLU
    there is no kink, so the error is second order in the box width instead of
    first order.
    """
    lo, hi = z.bounds()
    w = hi - lo
    tl, th = np.tanh(lo), np.tanh(hi)
    wide = w > 1e-12
    lam = np.where(wide, (th - tl) / np.where(wide, w, 1.0), 1.0 - tl ** 2)
    beta = np.where(wide, tl - lam * lo, tl - lam * lo)
    mu = TANH_D2_MAX * (w ** 2) / 8.0
    return bd.HZono(lam * z.c + beta, lam[:, None] * z.G,
                    _ABS(lam) * z.E + mu)


def relu_or_tanh(z, act):
    return bd.relu(z) if act == "relu" else tanh_relax(z)


def norm_op(z, g, b, variant, scale, eps):
    if variant == "standard":
        return bd.layernorm(z, g, b, eps)
    return fixnorm(z, g, b, scale)


def block(z, bw, n_heads, variant, eps=1e-5):
    act = "tanh" if variant.endswith("tanh") else "relu"
    zc = z.promote_E().compact(128)
    y1 = norm_op(zc, bw["ln1_g"], bw["ln1_b"], variant, bw["scale1"], eps)
    y1 = y1.promote_E_topk(48)
    a, _ = bd.attention(y1, bw["WQ"], bw["WK"], bw["WV"], bw["WO"], n_heads)
    z1 = zc + a
    y2 = norm_op(z1, bw["ln2_g"], bw["ln2_b"], variant, bw["scale2"], eps)
    y2 = y2.promote_E_topk(48)
    pre = y2.linear(bw["fc_in_W"], bw["fc_in_b"])
    plo, phi = pre.bounds()
    unstable = int(((plo < 0) & (phi > 0)).sum()) if act == "relu" else 0
    post = relu_or_tanh(pre, act)
    m = post.linear(bw["fc_out_W"], bw["fc_out_b"])
    out = (z1 + m).compact(128)
    return out, {"unstable": unstable, "n_units": int(plo.size),
                 "pre_width": float((phi - plo).mean())}


# ---------------------------------------------------------------- generators

def synth(d_model, seed, n_heads=4, n_layers=2, seq_len=5):
    g = np.random.default_rng(seed)
    s = 1.0 / np.sqrt(d_model)
    W = {"d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
         "seq_len": seq_len, "ln_eps": 1e-5, "blocks": []}
    for _ in range(n_layers):
        W["blocks"].append({
            "ln1_g": np.ones(d_model), "ln1_b": np.zeros(d_model),
            "ln2_g": np.ones(d_model), "ln2_b": np.zeros(d_model),
            "scale1": 1.0, "scale2": 1.0,
            "WQ": g.normal(scale=s, size=(d_model, d_model)),
            "WK": g.normal(scale=s, size=(d_model, d_model)),
            "WV": g.normal(scale=s, size=(d_model, d_model)),
            "WO": g.normal(scale=s, size=(d_model, d_model)),
            "fc_in_W": g.normal(scale=s, size=(4 * d_model, d_model)),
            "fc_in_b": np.zeros(4 * d_model),
            "fc_out_W": g.normal(scale=1.0 / np.sqrt(4 * d_model),
                                 size=(d_model, 4 * d_model)),
            "fc_out_b": np.zeros(d_model)})
    return W


def calibrate(W, x_nom):
    """Set each fixnorm scale to the nominal RMS at that site.

    A fixed normalizer is only usable if its constant is near the value the
    data-dependent one would take; otherwise the variant is 'verifiable'
    only by being a different, worse function.
    """
    d = W["d_model"]
    P0 = np.eye(d) - np.ones((d, d)) / d
    x = x_nom.copy()
    for bw in W["blocks"]:
        xc = x @ P0.T
        bw["scale1"] = float(np.sqrt((xc ** 2).mean() + W["ln_eps"]))
        z = bd.HZono.point(x[None])
        y1 = fixnorm(z, bw["ln1_g"], bw["ln1_b"], bw["scale1"])
        a, _ = bd.attention(y1, bw["WQ"], bw["WK"], bw["WV"], bw["WO"],
                            W["n_heads"])
        x1 = (z + a).c[0]
        xc1 = x1 @ P0.T
        bw["scale2"] = float(np.sqrt((xc1 ** 2).mean() + W["ln_eps"]))
        y2 = fixnorm(bd.HZono.point(x1[None]), bw["ln2_g"], bw["ln2_b"],
                     bw["scale2"])
        pre = y2.linear(bw["fc_in_W"], bw["fc_in_b"])
        x = (bd.HZono.point(x1[None])
             + bd.relu(pre).linear(bw["fc_out_W"], bw["fc_out_b"])).c[0]
    return W


def exact_forward(W, variant, xs):
    """Exact float64 forward for a variant, batched over samples."""
    act = "tanh" if variant.endswith("tanh") else "relu"
    d, T, H = W["d_model"], W["seq_len"], W["n_heads"]
    dh = d // H
    P0 = np.eye(d) - np.ones((d, d)) / d
    mask = np.triu(np.ones((T, T), dtype=bool), 1)
    for bw in W["blocks"]:
        xc = xs @ P0.T
        if variant == "standard":
            s1 = np.sqrt((xc ** 2).mean(-1, keepdims=True) + W["ln_eps"])
        else:
            s1 = bw["scale1"]
        y1 = xc / s1 * bw["ln1_g"] + bw["ln1_b"]
        q, k, v = y1 @ bw["WQ"].T, y1 @ bw["WK"].T, y1 @ bw["WV"].T
        outs = []
        for h in range(H):
            sl = slice(h * dh, (h + 1) * dh)
            sc = np.einsum("btd,bsd->bts", q[:, :, sl], k[:, :, sl]) / np.sqrt(dh)
            sc = np.where(mask[None], -1e9, sc)
            p = np.exp(sc - sc.max(-1, keepdims=True))
            p = p / p.sum(-1, keepdims=True)
            outs.append(np.einsum("bts,bsd->btd", p, v[:, :, sl]))
        x1 = xs + np.concatenate(outs, -1) @ bw["WO"].T
        xc1 = x1 @ P0.T
        if variant == "standard":
            s2 = np.sqrt((xc1 ** 2).mean(-1, keepdims=True) + W["ln_eps"])
        else:
            s2 = bw["scale2"]
        y2 = xc1 / s2 * bw["ln2_g"] + bw["ln2_b"]
        pre = y2 @ bw["fc_in_W"].T + bw["fc_in_b"]
        post = np.maximum(pre, 0.0) if act == "relu" else np.tanh(pre)
        xs = x1 + post @ bw["fc_out_W"].T + bw["fc_out_b"]
    return xs


def probe(W, variant, rho, seed):
    g = np.random.default_rng(seed + 1000)
    d, T = W["d_model"], W["seq_len"]
    x_nom = g.normal(size=(T, d))
    if variant != "standard":
        calibrate(W, x_nom)
    dirs = g.normal(size=(4, d))
    U = np.zeros((4, T, d))
    U[np.arange(4), T - 1, :] = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    mid = np.zeros((1, 4)); radv = np.full((1, 4), rho)
    c = np.einsum("bk,ktd->btd", mid, U) + x_nom[None]
    G = radv[:, :, None, None] * U[None]
    z = bd.HZono(c, G)
    in_w = float(z.width().max())
    layers = []
    for l in range(W["n_layers"]):
        z, st = block(z, W["blocks"][l], W["n_heads"], variant, W["ln_eps"])
        layers.append({"layer": l, **st, "out_width": float(z.width().max())})
    fin = float(z.width().max())
    # Attained width: without this the comparison is confounded. A variant can
    # look "more verifiable" merely by being a LESS SENSITIVE function, which is
    # a different claim with different costs. The quantity that isolates
    # verifiability is bound width divided by attained width.
    al = g.uniform(-rho, rho, size=(2000, 4))
    xs = np.einsum("bk,ktd->btd", al, U) + x_nom[None]
    out = exact_forward(W, variant, xs)
    att = float((out.max(0) - out.min(0)).max())
    return {"input_width": in_w, "final_width": fin,
            "amplification": fin / max(in_w, 1e-300),
            "attained_width": att,
            "attained_amplification": att / max(in_w, 1e-300),
            "relaxation_gap": fin / max(att, 1e-300),
            "layers": layers}


def verify_sound(d_model=32, seed=0, n=4000):
    """Sample-check the two new primitives inside a real propagation."""
    W = calibrate(synth(d_model, seed), np.random.default_rng(seed + 1000)
                  .normal(size=(5, d_model)))
    g = np.random.default_rng(seed + 1000)
    x_nom = g.normal(size=(5, d_model))
    dirs = g.normal(size=(4, d_model))
    U = np.zeros((4, 5, d_model))
    U[np.arange(4), 4, :] = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    rho = 1e-3
    worst = -np.inf
    for variant in ("fixnorm", "fixnorm_tanh"):
        act = "tanh" if variant.endswith("tanh") else "relu"
        z = bd.HZono(x_nom[None], np.full((1, 4), rho)[:, :, None, None] * U[None])
        al = g.uniform(-rho, rho, size=(n, 4))
        xs = np.einsum("bk,ktd->btd", al, U) + x_nom[None]
        for l in range(W["n_layers"]):
            bw = W["blocks"][l]
            z, _ = block(z, bw, W["n_heads"], variant, W["ln_eps"])
            # exact forward for the same variant
            P0 = np.eye(d_model) - np.ones((d_model, d_model)) / d_model
            xc = xs @ P0.T
            y1 = xc / bw["scale1"] * bw["ln1_g"] + bw["ln1_b"]
            q = y1 @ bw["WQ"].T; k = y1 @ bw["WK"].T; v = y1 @ bw["WV"].T
            H, dh = W["n_heads"], d_model // W["n_heads"]
            outs = []
            mask = np.triu(np.ones((5, 5), dtype=bool), 1)
            for h in range(H):
                sl = slice(h * dh, (h + 1) * dh)
                sc = np.einsum("btd,bsd->bts", q[:, :, sl], k[:, :, sl]) / np.sqrt(dh)
                sc = np.where(mask[None], -1e9, sc)
                p = np.exp(sc - sc.max(-1, keepdims=True))
                p = p / p.sum(-1, keepdims=True)
                outs.append(np.einsum("bts,bsd->btd", p, v[:, :, sl]))
            a = np.concatenate(outs, -1) @ bw["WO"].T
            x1 = xs + a
            xc1 = x1 @ P0.T
            y2 = xc1 / bw["scale2"] * bw["ln2_g"] + bw["ln2_b"]
            pre = y2 @ bw["fc_in_W"].T + bw["fc_in_b"]
            post = np.maximum(pre, 0.0) if act == "relu" else np.tanh(pre)
            xs = x1 + post @ bw["fc_out_W"].T + bw["fc_out_b"]
            lo, hi = z.bounds()
            worst = max(worst, float(np.max(np.maximum(lo[0][None] - xs,
                                                       xs - hi[0][None]))))
    return worst


def main():
    args = _parse_args()
    stamp("soundness check on the new primitives")
    viol = verify_sound()
    stamp(f"  max containment violation = {viol:+.3e}")
    assert viol <= 1e-9, "c18 primitives are UNSOUND -- no scaling number is meaningful"

    widths = [int(x) for x in args.widths.split(",")]
    rhos = [float(x) for x in args.rhos.split(",")]
    variants = args.variants.split(",")
    rows = []
    for d in widths:
        for variant in variants:
            for rho in rhos:
                amps = []
                for s in range(args.seeds):
                    W = synth(d, s)
                    amps.append(probe(W, variant, rho, s))
                a_vals = [p["amplification"] for p in amps]
                t_vals = [p["attained_amplification"] for p in amps]
                g_vals = [p["relaxation_gap"] for p in amps]
                l1 = amps[0]["layers"][-1]
                rows.append({"d_model": d, "variant": variant, "rho": rho,
                             "amplification_median": float(np.median(a_vals)),
                             "amplification_min": float(np.min(a_vals)),
                             "amplification_max": float(np.max(a_vals)),
                             "attained_amplification_median": float(np.median(t_vals)),
                             "relaxation_gap_median": float(np.median(g_vals)),
                             "seeds": args.seeds,
                             "L1_unstable": l1["unstable"],
                             "L1_units": l1["n_units"]})
                stamp(f"  d={d:4d} {variant:13s} rho={rho:<7g} "
                      f"amp={np.median(a_vals):.3e} attained={np.median(t_vals):.3e} "
                      f"gap={np.median(g_vals):.3e}  unstable={l1['unstable']}")

    rep = {"config": {"widths": widths, "rhos": rhos, "variants": variants,
                      "seeds": args.seeds},
           "soundness": {"max_containment_violation": viol, "sound": viol <= 1e-9},
           "rows": rows,
           "caveat": ("Randomly initialised, untrained transformers. This measures "
                      "how relaxation width propagates as a function of "
                      "architecture and d_model. It says nothing about trained "
                      "models, task performance, or certified radius on a real "
                      "network, and the fixnorm variants are different functions "
                      "from the baseline, not merely better-conditioned ones.")}


    def series(variant, rho, key):
        return [(r["d_model"], r[key]) for r in rows
                if r["variant"] == variant and r["rho"] == rho]


    def slope_of(pts):
        if len(pts) < 2:
            return None
        xs_ = np.log(np.array([p[0] for p in pts], float))
        ys_ = np.log(np.array([max(p[1], 1e-300) for p in pts], float))
        return float(np.polyfit(xs_, ys_, 1)[0])


    summary = {}
    r0 = min(rhos)
    for variant in variants:
        amp = series(variant, r0, "amplification_median")
        att = series(variant, r0, "attained_amplification_median")
        gap = series(variant, r0, "relaxation_gap_median")
        summary[variant] = {
            "rho": r0,
            "bound_amplification_by_width": {str(p[0]): p[1] for p in amp},
            "attained_amplification_by_width": {str(p[0]): p[1] for p in att},
            "relaxation_gap_by_width": {str(p[0]): p[1] for p in gap},
            "log_log_slope_bound": slope_of(amp),
            "log_log_slope_attained": slope_of(att),
            "log_log_slope_gap": slope_of(gap),
        }
    summary["_reading"] = (
        "log_log_slope_gap is the number that matters. The bound slope alone is "
        "confounded: a variant can shrink its bound by being a less sensitive "
        "function, which the attained slope exposes. Only if the GAP slope falls "
        "does the variant become genuinely easier to verify rather than merely "
        "quieter.")
    rep["scaling_summary"] = summary
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rep, open(args.out, "w"), indent=2)
    stamp(f"wrote {args.out}")
    print(json.dumps(summary, indent=2))



if __name__ == "__main__":
    main()
