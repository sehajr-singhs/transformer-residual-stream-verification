"""c14 part 1 -- exact SMT audit of the prover's primitives.

An end-to-end independent sound bound is not achievable here (see
audit/README.md: every decorrelating method dies through layer 1), so the audit
attacks the engine where an exact solver actually has traction: the primitive
relaxations the end-to-end proof is assembled from. z3 works over exact rational
arithmetic, so unlike an interval engine it has no wrapping problem at all.

A hybrid affine form  L_v = c_v + sum_k G_kv eps_k + delta_v,  eps_k in [-1,1],
|delta_v| <= E_v  denotes a set. Every primitive claims that its output form
contains the image of its input form. Each claim below is discharged by asking
z3 for a counterexample and requiring UNSAT.

Checks
------
margin_soundness  no point of the logit form has margin above the reported bound
margin_tightness  some point comes within tol of it. This is the check that
                  fails for the ORIGINAL min/max bug: that version returned
                  max_u L_u - min_j L_j, which is still an upper bound (so the
                  soundness check passes) but is inflated by ~11.7 logits, so
                  nothing attains it. Soundness alone cannot see that bug.
relu_containment  the DeepZ output form contains relu of every input point,
                  with generator symbols paired by index -- the exact invariant
                  the compact()/promote() alignment bug violated.
ln_containment    same for LayerNorm (nonlinear: 1/sqrt(var+eps)), under a
                  per-query timeout since this one is genuinely hard for nlsat.
"""
import sys, os, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, torch
import z3
from src import toy_transformer as tt, task, sae as sae_mod, bounds as bd, verifier as vf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "checkpoints")

ap = argparse.ArgumentParser()
ap.add_argument("--rho", type=float, default=0.02)
ap.add_argument("--n-boxes", type=int, default=8)
ap.add_argument("--n-relu", type=int, default=2)
ap.add_argument("--n-ln", type=int, default=1)
# The margin query is a near-degenerate LP: the prover's bound IS the exact
# supremum, so refuting it exercises exact rational simplex over ~128 box
# variables. Measured 13-18s per query and erratic, so 30s produced `unknown`
# everywhere and looked like a solver failure. It is not; it just needs budget.
ap.add_argument("--timeout-ms", type=int, default=180000)
ap.add_argument("--tol", type=float, default=1e-6)
ap.add_argument("--out", default=os.path.join(ROOT, "results", "c14_z3.json"))
args = ap.parse_args()

t0 = time.time()
def stamp(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)


# ------------------------------------------------------------------ rebuild
model = tt.ToyTransformer()
model.load_state_dict(torch.load(os.path.join(CK, "model.pt")))
model.eval()
w = model.export_weights()
model64 = tt.ToyTransformer()
model64.load_state_dict(torch.load(os.path.join(CK, "model.pt")))
model64.eval().double()
sae = sae_mod.SAE(w["d_model"], 64)
sae.load_state_dict(torch.load(os.path.join(CK, "sae.pt")))
Wdec = sae.W_dec.detach().numpy().astype(np.float64).T
Wdec = Wdec / np.linalg.norm(Wdec, axis=1, keepdims=True)
pick = np.random.default_rng(11).choice(Wdec.shape[0], size=4, replace=False)
U = np.zeros((4, w["seq_len"], w["d_model"]))
U[np.arange(4), w["seq_len"] - 1, :] = Wdec[pick]
prompts, _ = task.enumerate_prompts(limit=256, rng=np.random.default_rng(0),
                                    safe_only=True)
with torch.no_grad():
    tr = model64.residual_trace(torch.from_numpy(prompts))
x_nom_0 = tr[0][0].numpy().astype(np.float64)
iu, isf = task.margin_readout()
stamp(f"rebuilt; feature_ids={pick.tolist()}  z3 {z3.get_version_string()}")


def R(x):
    """Exact rational from a float64 -- no decimal rounding anywhere."""
    return z3.RealVal(float(x))


def affine(c, Gcol, E, eps, name):
    """c + sum_k G_k eps_k + delta,  |delta| <= E.  Returns (expr, [constraints])."""
    e = R(c)
    for k in range(len(eps)):
        if Gcol[k] != 0.0:
            e = e + R(Gcol[k]) * eps[k]
    cons = []
    if E > 0:
        d = z3.Real(name)
        cons = [d >= R(-E), d <= R(E)]
        e = e + d
    return e, cons


# -------------------------------------------------- 1. margin readout audit
def audit_margin(zL, bound, tight_slack=1e-3):
    """zL: logit HZono for one box (B=1). bound: what the prover reported.

    max_u L_u - max_j L_j > B  <=>  EXISTS u such that FORALL j: L_u - L_j > B.

    Using that identity keeps the query in pure linear real arithmetic. Encoding
    the two maxima with nested If instead makes z3 branch over 2*10 cases and it
    returns `unknown` within a minute; this form is decided in milliseconds.
    """
    c = zL.c[0, -1, :]
    G = zL.G[0, :, -1, :]
    E = zL.E[0, -1, :]
    m, V = G.shape[0], c.shape[0]
    eps = [z3.Real(f"e{k}") for k in range(m)]
    base = [z3.And(x >= R(-1), x <= R(1)) for x in eps]
    L, extra = [], []
    for v in range(V):
        e, cn = affine(c[v], G[:, v], E[v], eps, f"d{v}")
        L.append(e); extra += cn

    def exceeds(thr):
        return z3.Or([z3.And([L[u] - L[j] > R(thr) for j in isf]) for u in iu])

    s = z3.Solver(); s.set("timeout", args.timeout_ms)
    s.add(base + extra); s.add(exceeds(bound + args.tol))
    r_sound = s.check()

    s2 = z3.Solver(); s2.set("timeout", args.timeout_ms)
    s2.add(base + extra); s2.add(exceeds(bound - tight_slack))
    r_tight = s2.check()
    return str(r_sound), str(r_tight)


# ------------------------------------------------- 2. relu containment audit
def audit_relu(ci, Gi, Ei, co, Go, Eo):
    """For all eps, delta_in: |relu(x) - (c_out + G_out eps)| <= E_out.

    Symbols are paired by index between input and output forms, which is exactly
    the invariant that compacting mid-block destroys.
    """
    n, m = ci.shape[0], Gi.shape[0]
    eps = [z3.Real(f"e{k}") for k in range(m)]
    cons = [z3.And(x >= R(-1), x <= R(1)) for x in eps]
    viol = []
    for i in range(n):
        xi, cn = affine(ci[i], Gi[:, i], Ei[i], eps, f"din{i}")
        cons += cn
        rx = z3.If(xi > 0, xi, R(0))
        centre = R(co[i])
        for k in range(m):
            if Go[k, i] != 0.0:
                centre = centre + R(Go[k, i]) * eps[k]
        viol.append(z3.Or(rx - centre > R(Eo[i] + args.tol),
                          centre - rx > R(Eo[i] + args.tol)))
    s = z3.Solver(); s.set("timeout", args.timeout_ms)
    s.add(cons); s.add(z3.Or(viol))
    return str(s.check())


# --------------------------------------------- 3. layernorm containment audit
def audit_layernorm(ci, Gi, Ei, co, Go, Eo, gamma, beta, eps_ln, d):
    """Exact 1/sqrt(var+eps) via an auxiliary t with t^2 (var+eps) = 1, t > 0."""
    m = Gi.shape[0]
    eps = [z3.Real(f"e{k}") for k in range(m)]
    cons = [z3.And(x >= R(-1), x <= R(1)) for x in eps]
    X = []
    for i in range(d):
        xi, cn = affine(ci[i], Gi[:, i], Ei[i], eps, f"dl{i}")
        cons += cn; X.append(xi)
    mean = z3.Sum(X) / R(d)
    Xc = [X[i] - mean for i in range(d)]
    var = z3.Sum([Xc[i] * Xc[i] for i in range(d)]) / R(d)
    t = z3.Real("t_inv_sigma")
    cons += [t > 0, t * t * (var + R(eps_ln)) == 1]
    viol = []
    for i in range(d):
        y = R(gamma[i]) * Xc[i] * t + R(beta[i])
        centre = R(co[i])
        for k in range(m):
            if Go[k, i] != 0.0:
                centre = centre + R(Go[k, i]) * eps[k]
        viol.append(z3.Or(y - centre > R(Eo[i] + args.tol),
                          centre - y > R(Eo[i] + args.tol)))
    s = z3.Solver(); s.set("timeout", args.timeout_ms)
    s.add(cons); s.add(z3.Or(viol))
    return str(s.check())


# ---------------------------------------------------------------- run audits
rho = args.rho
rng = np.random.default_rng(5)
rep = {"config": {"rho": rho, "n_boxes": args.n_boxes, "tol": args.tol,
                  "timeout_ms": args.timeout_ms,
                  "z3_version": z3.get_version_string()},
       "margin_readout": [], "relu": [], "layernorm": []}

def hz_margin(a_lo, a_hi):
    z = vf.alpha_to_zono(a_lo, a_hi, U) + x_nom_0[None]
    zL = vf._blocks_from(z, w, 0, w["n_layers"])
    return bd.unsafe_margin_upper(bd.readout_logits(zL, w), iu, isf)


def discharged_boxes(rho, max_boxes=4096, max_iters=20):
    """Replay the prover's BaB and return the boxes it actually discharges.

    Auditing arbitrary sub-boxes is not the same thing: at radius rho/2 the
    bound is still in the tens of thousands and no certificate is being made.
    The claims worth auditing are the ones the proof actually rests on.
    """
    a_lo = np.full((1, 4), -rho); a_hi = np.full((1, 4), rho)
    out_lo, out_hi = [], []
    for _ in range(max_iters):
        ub = np.where(np.isfinite(hz_margin(a_lo, a_hi)), hz_margin(a_lo, a_hi), np.inf)
        keep = ub >= 0.0
        if (~keep).any():
            out_lo.append(a_lo[~keep]); out_hi.append(a_hi[~keep])
        a_lo, a_hi = a_lo[keep], a_hi[keep]
        if a_lo.shape[0] == 0 or a_lo.shape[0] * 2 > max_boxes:
            break
        wd = a_hi - a_lo
        dim = wd.argmax(axis=1); r = np.arange(a_lo.shape[0])
        mid = 0.5 * (a_lo[r, dim] + a_hi[r, dim])
        l1, h1 = a_lo.copy(), a_hi.copy(); h1[r, dim] = mid
        l2, h2 = a_lo.copy(), a_hi.copy(); l2[r, dim] = mid
        a_lo = np.concatenate([l1, l2]); a_hi = np.concatenate([h1, h2])
    return np.concatenate(out_lo), np.concatenate(out_hi)


DLO, DHI = discharged_boxes(rho)
stamp(f"prover discharges {DLO.shape[0]} boxes at rho={rho}; auditing "
      f"{min(args.n_boxes, DLO.shape[0])}")
sel = rng.choice(DLO.shape[0], size=min(args.n_boxes, DLO.shape[0]), replace=False)

for n, bi in enumerate(sel):
    a_lo, a_hi = DLO[bi][None], DHI[bi][None]

    z0 = vf.alpha_to_zono(a_lo, a_hi, U) + x_nom_0[None]
    zL_stream = vf._blocks_from(z0, w, 0, w["n_layers"])
    zlog = bd.readout_logits(zL_stream, w)
    bound = float(bd.unsafe_margin_upper(zlog, iu, isf)[0])

    rs, rt = audit_margin(zlog, bound)
    rep["margin_readout"].append({"box": int(bi), "bound": bound,
                                  "soundness_query": rs, "tightness_query": rt,
                                  "sound": rs == "unsat", "tight": rt == "sat"})
    stamp(f"  box {bi}: margin {bound:+.4f}  soundness={rs}  tightness={rt}")

    # Mirror block() exactly so the audited forms are the ones the prover builds.
    bw = w["blocks"][0]
    zc = z0.promote_E().compact(128)
    y1 = bd.layernorm(zc, bw["ln1_g"], bw["ln1_b"], w["ln_eps"]).promote_E_topk(48)
    a1, _ = bd.attention(y1, bw["WQ"], bw["WK"], bw["WV"], bw["WO"], w["n_heads"])
    z1 = zc + a1
    ln2_in = z1
    y2 = bd.layernorm(z1, bw["ln2_g"], bw["ln2_b"], w["ln_eps"])
    y2p = y2.promote_E_topk(48)
    pre = y2p.linear(bw["fc_in_W"], bw["fc_in_b"])
    post = bd.relu(pre)

    if n < args.n_relu:
        rr = audit_relu(pre.c[0, -1, :], pre.G[0, :, -1, :], pre.E[0, -1, :],
                        post.c[0, -1, :], post.G[0, :, -1, :], post.E[0, -1, :])
        rep["relu"].append({"box": int(bi), "query": rr, "sound": rr == "unsat",
                            "conclusive": rr in ("unsat", "sat"),
                            "n_units": int(pre.c.shape[-1]),
                            "n_symbols": int(pre.G.shape[1])})
        stamp(f"  box {bi}: relu containment = {rr}  "
              f"({pre.c.shape[-1]} units, {pre.G.shape[1]} symbols)")

    if n < args.n_ln:             # nonlinear (1/sqrt), genuinely hard for nlsat
        rl = audit_layernorm(ln2_in.c[0, -1, :], ln2_in.G[0, :, -1, :],
                             ln2_in.E[0, -1, :], y2.c[0, -1, :],
                             y2.G[0, :, -1, :], y2.E[0, -1, :],
                             bw["ln2_g"], bw["ln2_b"], w["ln_eps"], w["d_model"])
        rep["layernorm"].append({"box": int(bi), "query": rl,
                                 "sound": rl == "unsat",
                                 "conclusive": rl in ("unsat", "sat")})
        stamp(f"  box {bi}: layernorm containment = {rl}")

def tally(rows, key, qkey="query"):
    return {"conclusive": sum(1 for r in rows if r[qkey] in ("sat", "unsat")),
            "attempted": len(rows),
            "passed": sum(1 for r in rows if r.get(key)),
            "queries": [r[qkey] for r in rows]}


rep["summary"] = {
    "margin_soundness": tally(rep["margin_readout"], "sound", "soundness_query"),
    "margin_tightness": {"passed": sum(1 for r in rep["margin_readout"] if r["tight"]),
                         "attempted": len(rep["margin_readout"]),
                         "queries": [r["tightness_query"] for r in rep["margin_readout"]]},
    "relu": tally(rep["relu"], "sound"),
    "layernorm": tally(rep["layernorm"], "sound"),
    "note": ("`unknown` is a solver timeout, not a refutation. Only `unsat` on a "
             "soundness query is a positive result; only `sat` on a tightness "
             "query is."),
}
rep["runtime_sec"] = time.time() - t0
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(rep, open(args.out, "w"), indent=2)
stamp(f"wrote {args.out}")
print(json.dumps(rep["summary"], indent=2))
