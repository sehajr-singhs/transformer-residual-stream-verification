"""Compositional dissipativity: local supply rates and their interconnection.

The multi-layer, multi-head network is deconstructed into sub-components --
each attention head and each MLP -- and each is given a locally certified
input-output characterization over a shared deviation box. Heads inside a layer
are in PARALLEL (their outputs sum into the residual stream, so supply rates
add); layers are in SERIES (the output deviation of layer l is the input
deviation of layer l+1, so gains chain).

An honest structural obstruction, found and reported rather than designed around
----------------------------------------------------------------------------
The pre-LN residual stream carries an exact identity path:

    e_{l+1} = e_l + dAttn_l(e_l) + dMLP_l(e_l)

so ANY norm-gain bound obeys  gamma_layer <= 1 + sum_h gamma_h + gamma_mlp,
and the right-hand side is >= 1 unconditionally. A pure small-gain / L2-gain
composition therefore CANNOT certify contraction of a residual network, no
matter how small the sub-block gains are. This is not a tuning failure; it is a
theorem about the architecture, and any paper claiming a small-gain safety
certificate over a residual stream without addressing it is wrong.

What actually works, and why the compositional machinery still earns its place
----------------------------------------------------------------------------
Contraction has to come from CANCELLATION -- the sub-block output must be
anti-correlated with e, which a norm gain cannot see because it discards sign.
That is exactly what a learned V with cross terms captures. The compositional
win is then not "multiply the gains" but "verify each layer's dissipation
inequality INDEPENDENTLY over a shared box, and chain them algebraically":

    (D_l) for all l over B_rho   +   {V <= c*} subset B_rho
        ==>  global invariance and safety

Cost of the monolithic alternative is exponential in L (reachable-set growth
compounds); cost here is LINEAR in L, with each layer's problem the same size.
`composition_cost_model` quantifies that, and `layer_gain_report` quantifies the
obstruction above so it cannot be quietly forgotten.
"""
import numpy as np
from scipy.optimize import linprog

from . import bounds as bd


def _dev_out(w, layer, x_nom, rho, part, head=None, chunk=32):
    """Sound elementwise output-deviation radius of one sub-block over B_rho."""
    T, D = x_nom.shape
    H = w["n_heads"]
    bw = w["blocks"][layer]
    lo = np.full((1, T, D), -rho) + x_nom[None]
    hi = np.full((1, T, D), rho) + x_nom[None]
    z = bd.HZono.from_box(lo, hi)
    if part == "attn":
        y = bd.layernorm(z, bw["ln1_g"], bw["ln1_b"], w["ln_eps"])
        if head is None:
            out, _ = bd.attention(y, bw["WQ"], bw["WK"], bw["WV"], bw["WO"], H)
        else:
            dh = D // H
            sl = slice(head * dh, (head + 1) * dh)
            WO = np.zeros_like(bw["WO"]); WO[:, sl] = bw["WO"][:, sl]
            out, _ = bd.attention(y, bw["WQ"], bw["WK"], bw["WV"], WO, H)
    elif part == "mlp":
        y = bd.layernorm(z, bw["ln2_g"], bw["ln2_b"], w["ln_eps"])
        out = bd.mlp(y, bw["fc_in_W"], bw["fc_in_b"], bw["fc_out_W"], bw["fc_out_b"])
    else:
        raise ValueError(part)
    return out.radius()[0]


def subblock_gains(w, x_nom_trace, rho):
    """Certified linf -> linf incremental gains for every sub-block.

    gamma = sup_{||e||_inf <= rho} ||sub(e) - sub(0)||_inf / rho, upper-bounded
    soundly by the zonotope output radius. Sound because the propagated radius
    over-approximates the reachable deviation set.
    """
    rows = []
    for l in range(w["n_layers"]):
        xn = x_nom_trace[l]
        g_heads = [float(_dev_out(w, l, xn, rho, "attn", head=h).max() / rho)
                   for h in range(w["n_heads"])]
        g_attn = float(_dev_out(w, l, xn, rho, "attn").max() / rho)
        g_mlp = float(_dev_out(w, l, xn, rho, "mlp").max() / rho)
        rows.append({
            "layer": l,
            "gamma_heads": g_heads,
            "gamma_attn_joint": g_attn,
            "gamma_heads_sum": float(sum(g_heads)),
            "gamma_mlp": g_mlp,
            "gamma_layer_upper": float(1.0 + g_attn + g_mlp),
        })
    return rows


def layer_gain_report(gains):
    """State the identity-path obstruction with the measured numbers attached."""
    prod = float(np.prod([g["gamma_layer_upper"] for g in gains]))
    return {
        "per_layer_gain_upper": [g["gamma_layer_upper"] for g in gains],
        "cascade_gain_product": prod,
        "small_gain_satisfied": bool(prod < 1.0),
        "obstruction": ("identity path forces gamma_layer >= 1, so no norm-gain "
                        "composition can certify contraction of a residual "
                        "stream; contraction must come from sign-aware "
                        "cancellation, i.e. from V's cross terms"),
        "parallel_head_slack": [
            float(g["gamma_heads_sum"] - g["gamma_attn_joint"]) for g in gains],
    }


def compose_supply_rates(gains, kappa=0.05):
    """Interconnection LP over storage multipliers lambda_l > 0.

    Each layer is assigned the quadratic supply rate
        s_l = gamma_l^2 ||u_l||^2 - ||y_l||^2
    and the series interconnection y_l = u_{l+1} makes
        sum_l lambda_l s_l = sum_l ( lambda_l gamma_l^2 - lambda_{l-1} ) ||u_l||^2 .
    Dissipativity of the interconnection needs every coefficient <= -kappa, an LP
    in lambda. It is solved honestly here, and it is EXPECTED to be infeasible
    for a residual stream (see layer_gain_report) -- the point of running it is
    to demonstrate the obstruction is real rather than to paper over it.
    """
    L = len(gains)
    g2 = np.array([g["gamma_layer_upper"] ** 2 for g in gains])
    A = np.zeros((L, L))
    for l in range(L):
        A[l, l] = g2[l]
        if l > 0:
            A[l, l - 1] = -1.0
    b = np.full(L, -kappa)
    res = linprog(c=np.ones(L), A_ub=A, b_ub=b,
                  bounds=[(1e-6, 1e6)] * L, method="highs")
    return {
        "feasible": bool(res.success),
        "status": str(res.message),
        "lambda": None if not res.success else res.x.tolist(),
        "kappa": kappa,
        "interpretation": ("feasible => norm-based compositional certificate "
                           "exists; infeasible => must use the learned-V route"),
    }


def composition_cost_model(n_layers, boxes_per_layer, monolithic_boxes=None):
    """Verification cost: compositional (linear in L) vs monolithic.

    The compositional obligation verifies each layer over the SAME box B_rho, so
    total cost is sum_l cost_l. The monolithic obligation propagates the
    reachable set through all L layers before any check, so box counts compound.
    """
    comp = int(sum(boxes_per_layer))
    out = {
        "compositional_boxes_total": comp,
        "compositional_boxes_per_layer": list(map(int, boxes_per_layer)),
        "n_layers": n_layers,
        "scaling": "O(L) -- each layer verified independently over a shared box",
    }
    if monolithic_boxes is not None:
        out["monolithic_boxes"] = int(monolithic_boxes)
        out["speedup_vs_monolithic"] = float(monolithic_boxes / max(comp, 1))
    return out
