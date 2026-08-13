"""Input-Convex Neural Lyapunov function, positive definite BY CONSTRUCTION.

State
-----
The certificate is stated in DEVIATION coordinates around the nominal
trajectory of a fixed prompt class:

    e_l = x_l - x*_l ,      e_{l+1} = Block_l(x*_l + e_l) - x*_{l+1}

This is the "relative / invariant frame" the design calls for, and it is not
cosmetic. A 2-layer transformer has no asymptotic behaviour to appeal to, so
"V decreases along layers" only means something as an INCREMENTAL contraction
statement: a perturbation injected into the residual stream must shrink as it
propagates. In deviation coordinates the origin e = 0 is an exact fixed point of
the deviation dynamics for every prompt, by construction, with no equilibrium
solve required.

Form
----
    V(e) = ReLU( g(W_enc e) - g(0) )  +  alpha * || P e ||^2

  * ReLU(g - g(0))  forces V(0) = 0 and V >= 0. Not a penalty -- a violation is
    not representable by any parameter setting.
  * alpha ||P e||^2 forces STRICT positivity off the origin, on the quotient
    that drops the LayerNorm-annihilated mean direction (frames.py). Small by
    design: it is the only term whose bound propagation loses correlation, so
    alpha directly buys relaxation gap. That price is measured, not assumed.
  * g is an ICNN (nonnegative recurrent weights, convex nondecreasing
    activations) so g is convex; W_enc e is LINEAR in e, therefore V is convex
    in e outright, not merely in feature coordinates. Sublevel sets are convex
    convex sets in state space, which is what makes containment in the safe
    halfspace a checkable rather than aspirational condition.

The feature frame enters as W_enc: V measures perturbation energy resolved
along the SAE's learned feature directions, with the ICNN free to weight and
couple them convexly. Using the linear readout rather than ReLU(W_enc e + b)
is deliberate -- a displacement is signed, and it removes an entire layer of
ReLU relaxation from the verified path.
"""
from . import task  # noqa: F401

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import bounds as bd
from . import frames


class ICNNLyapunov(nn.Module):
    def __init__(self, sae, d_model, seq_len, widths=(64, 64), alpha=1e-3):
        super().__init__()
        self.alpha = alpha
        self.seq_len, self.d_model, self.d_dict = seq_len, d_model, sae.d_dict
        self.register_buffer("W_enc", sae.W_enc.detach().clone())
        self.register_buffer("P", torch.as_tensor(frames.centering_projector(d_model),
                                                  dtype=torch.float32))
        din = seq_len * self.d_dict
        self.Wy = nn.ModuleList()
        self.Wz_raw = nn.ParameterList()
        prev = None
        for wdt in widths:
            self.Wy.append(nn.Linear(din, wdt))
            if prev is not None:
                self.Wz_raw.append(nn.Parameter(torch.randn(wdt, prev) * 0.05 - 2.0))
            prev = wdt
        self.out_y = nn.Linear(din, 1)
        self.out_z_raw = nn.Parameter(torch.randn(1, prev) * 0.05 - 2.0)

    def df(self, e):
        return (e @ self.W_enc.T).reshape(e.shape[0], -1)

    def _g(self, y):
        u = F.relu(self.Wy[0](y))
        for i in range(1, len(self.Wy)):
            u = F.relu(F.linear(u, F.softplus(self.Wz_raw[i - 1])) + self.Wy[i](y))
        return F.linear(u, F.softplus(self.out_z_raw)) + self.out_y(y)

    def g0(self):
        z = torch.zeros(1, self.seq_len * self.d_dict, device=self.W_enc.device)
        return self._g(z)

    def forward(self, e):
        """e: (B, seq_len, d_model) deviation -> V: (B,)"""
        conv = F.relu(self._g(self.df(e)) - self.g0()).squeeze(-1)
        er = e @ self.P.T
        return conv + self.alpha * (er ** 2).sum(dim=(1, 2))

    @torch.no_grad()
    def export_weights(self):
        return {
            "alpha": float(self.alpha),
            "P": self.P.detach().numpy().astype(np.float64),
            "W_enc": self.W_enc.detach().numpy().astype(np.float64),
            "Wy": [(l.weight.detach().numpy().astype(np.float64),
                    l.bias.detach().numpy().astype(np.float64)) for l in self.Wy],
            "Wz": [F.softplus(p).detach().numpy().astype(np.float64)
                   for p in self.Wz_raw],
            "out_y": (self.out_y.weight.detach().numpy().astype(np.float64),
                      self.out_y.bias.detach().numpy().astype(np.float64)),
            "out_z": F.softplus(self.out_z_raw).detach().numpy().astype(np.float64),
            "g0": float(self.g0().item()),
            "seq_len": self.seq_len,
            "d_dict": self.d_dict,
            "d_model": self.d_model,
        }


# ----------------------------------------------------------------- sound bounds


def quad_form(e, P, alpha):
    """Sound HZono scalar for alpha * ||P e||^2.

    The cross term 2 c.w is affine in the noise symbols and is kept EXACTLY in
    the generator part; only the genuinely quadratic ||w||^2 and the E-cross
    term are relaxed. This matters because V appears twice in dV = V' - V and
    correlated parts cancel there; anything routed to E does not cancel.
    """
    er = e.linear(P)
    B, m = er.B, er.m
    cf = er.c.reshape(B, -1)
    Gf = er.G.reshape(B, m, -1)
    Ef = er.E.reshape(B, -1)
    r = np.abs(Gf).sum(axis=1) + Ef
    sum_r2 = (r ** 2).sum(axis=1)
    c_q = alpha * ((cf ** 2).sum(axis=1) + 0.5 * sum_r2)
    G_q = alpha * 2.0 * np.einsum("bn,bmn->bm", cf, Gf)
    E_q = alpha * (0.5 * sum_r2 + 2.0 * (np.abs(cf) * Ef).sum(axis=1))
    return bd.HZono(c_q[:, None], G_q[:, :, None], E_q[:, None])


def quadratic_only_vw(vw):
    """A trivial baseline metric V_q(e) = alpha ||P e||^2 in the same interface.

    The control question for this whole design: does the learned ICNN metric
    certify a SMALLER growth factor than the trivial quadratic one? If it does
    not, the ICNN is decoration and the honest certificate is just "the l2 norm
    of the perturbation grows by at most sqrt(gamma) per layer". Reported either
    way. Achieved by zeroing the ICNN output path so only the quadratic term
    survives, which keeps every bound routine identical.
    """
    out = dict(vw)
    Wo, bo = vw["out_y"]
    out["out_y"] = (np.zeros_like(Wo), np.zeros_like(bo))
    out["out_z"] = np.zeros_like(vw["out_z"])
    out["g0"] = 0.0
    return out


def v_bound(e, vw, want_quad=False):
    """Sound HZono scalar bound on V(e) for a hybrid zonotope deviation e.

    With want_quad, also returns the quadratic term alone. That matters because
    the DeepZ ReLU relaxation gives ReLU output a NEGATIVE lower bound
    (y_lo = lambda * x_lo < 0 for a crossing neuron) -- sound, but it admits
    values ReLU cannot produce. Since the true convex term is >= 0, V >= quad,
    so quad's lower bound is a strictly better certified floor on V than the
    concretized one. Without this, -gamma * V_lo is POSITIVE and increasing gamma
    makes the growth bound worse instead of better, which is exactly the
    gamma-insensitivity that stalled certification.
    """
    df = e.linear(vw["W_enc"]).reshape_trailing((e.S[0] * vw["d_dict"],))
    u = None
    for i, (W, b) in enumerate(vw["Wy"]):
        lin = df.linear(W, b)
        u = bd.relu(lin) if i == 0 else bd.relu(u.linear(vw["Wz"][i - 1]) + lin)
    Wo, bo = vw["out_y"]
    g = u.linear(vw["out_z"]) + df.linear(Wo, bo)
    conv = bd.relu(g - vw["g0"])
    quad = quad_form(e, vw["P"], vw["alpha"])
    V = conv + quad
    return (V, quad) if want_quad else V


def lyap_gap_upper(e_now, e_next, vw, gamma):
    """Sound upper bound on  V(e_{l+1}) - gamma * V(e_l).

    gamma < 1 is a contraction claim; gamma >= 1 is a bounded-GROWTH claim. For
    a finite-depth residual network the second is the well-posed one: the
    deviation Jacobian restricted to the certified subspace has spectral radius
    1.273 > 1 for this model, and a positive-definite V satisfying
    V(e') <= gamma V(e) with gamma < 1 exists if and ONLY IF that radius is < 1.
    Asking for decrease is therefore asking for a false theorem, and no amount of
    training or tighter relaxation can supply it. Certifying the smallest
    provable gamma instead is both true and sufficient for safety, because the
    composition gamma_0 * ... * gamma_{L-1} bounds how far a layer-0 perturbation
    can travel before the readout.

    Both V's are propagated over the SAME noise symbols so the difference is
    formed INSIDE the affine form -- the shared dependence on the input
    perturbation cancels generator by generator before concretization. Bounding
    max V' and min V separately would be sound but would discard exactly that
    cancellation.
    """
    v1 = v_bound(e_next, vw)
    v0, q0 = v_bound(e_now, vw, want_quad=True)

    # (a) correlated: form the difference INSIDE the affine form so the shared
    #     dependence on the input perturbation cancels generator by generator.
    d = v1 - v0.scale(np.full(v0.c.shape, gamma))
    lo, hi = d.bounds()

    # (b) decorrelated but with a CLAMPED floor on V(e_l). The true V is >= its
    #     quadratic part and >= 0, so this floor is sound and is far better than
    #     the concretized lower bound wherever the ICNN's ReLUs are unstable.
    v0lo, v0hi = v0.bounds()
    v1lo, v1hi = v1.bounds()
    q0lo, _ = q0.bounds()
    floor = np.maximum(np.maximum(v0lo, q0lo), 0.0)
    hi_b = v1hi - gamma * floor

    # min of two sound upper bounds is sound.
    return {"d_hi": np.minimum(hi, hi_b)[:, 0], "d_lo": lo[:, 0],
            "d_hi_correlated": hi[:, 0], "d_hi_clamped": hi_b[:, 0],
            "v0_lo": v0lo[:, 0], "v0_hi": v0hi[:, 0],
            "v0_floor": floor[:, 0],
            "v1_lo": v1lo[:, 0], "v1_hi": v1hi[:, 0]}


def decrease_upper(e_now, e_next, vw, kappa=0.0):
    """Tight sound upper bound on  V(e_{l+1}) - (1-kappa) V(e_l).

    Both V's are propagated over the SAME noise symbols, so the difference is
    formed INSIDE the affine form -- the shared dependence on the input
    perturbation cancels generator-by-generator before concretization. Bounding
    max V' and min V separately would be sound but would discard exactly the
    cancellation that makes a decrease certificate provable at all.
    """
    v1 = v_bound(e_next, vw)
    v0 = v_bound(e_now, vw)
    d = v1 - v0.scale(np.full(v0.c.shape, 1.0 - kappa))
    lo, hi = d.bounds()
    v0lo, v0hi = v0.bounds()
    v1lo, v1hi = v1.bounds()
    return {"d_hi": hi[:, 0], "d_lo": lo[:, 0],
            "v0_lo": v0lo[:, 0], "v0_hi": v0hi[:, 0],
            "v1_lo": v1lo[:, 0], "v1_hi": v1hi[:, 0]}


# -------------------------------------------------------------------- training


def train_lyapunov(model, sae, prompts, U, steps=2500, batch=256, lr=2e-3,
                   rho=0.25, kappa=0.05, alpha=1e-3, widths=(64, 64), seed=0,
                   log_every=500, verbose=True, penalty=50.0):
    """Fit V to MINIMIZE the layer-wise growth factor gamma with V(e') <= gamma V(e).

    Not a decrease objective. The deviation dynamics of this network is expansive
    on the certified subspace (spectral radius 1.273), so a decrease objective is
    infeasible and optimizing it just drives the violation rate to ~50% while the
    loss looks small -- which is exactly what happened before this was diagnosed.
    Here gamma is an explicit trainable scalar minimized under a penalty on
    violations, so the number V is being fitted for is the number the verifier
    will later try to prove.

    Training remains heuristic; only the verifier makes claims. Any gap between
    the sampled gamma below and the certified gamma is the sampling-vs-proof gap
    and is reported as such.
    """
    torch.manual_seed(seed)
    d_model = model.embed.weight.shape[1]
    V = ICNNLyapunov(sae, d_model, prompts.shape[1], widths=widths, alpha=alpha)
    log_gamma = torch.nn.Parameter(torch.tensor(0.2))
    opt = torch.optim.Adam(list(V.parameters()) + [log_gamma], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator().manual_seed(seed)
    P = torch.from_numpy(prompts)

    with torch.no_grad():
        chunks = [[t.detach() for t in model.residual_trace(P[i:i + 1024])]
                  for i in range(0, len(P), 1024)]
        n_states = len(chunks[0])
        trace = [torch.cat([c[l] for c in chunks], 0) for l in range(n_states)]

    Ut = torch.as_tensor(U, dtype=torch.float32)
    k = Ut.shape[0]
    N, hist = P.shape[0], []
    for step in range(steps):
        idx = torch.randint(0, N, (batch,), generator=g)
        loss, viol = 0.0, 0.0
        for l in range(n_states - 1):
            xs, xn = trace[l][idx], trace[l + 1][idx]
            # Deviations are drawn from the SAME subspace the verifier will
            # certify over. Training on a distribution the prover never sees is
            # how sampled-violation numbers end up looking good while nothing is
            # provable. Log-uniform radii keep small deviations -- where strict
            # decrease is hardest -- from being under-sampled.
            scale = rho * torch.pow(10.0, -2.0 * torch.rand((batch, 1), generator=g))
            a = (torch.rand((batch, k), generator=g) * 2 - 1) * scale
            e0 = torch.einsum("bk,ktd->btd", a, Ut)
            with torch.no_grad():
                e1 = model.blocks[l](xs + e0) - xn
            v0, v1 = V(e0), V(e1)
            gamma = torch.exp(log_gamma)
            slack = F.relu(v1 - gamma * v0)
            loss = loss + penalty * slack.mean() / v0.detach().clamp_min(1e-9).mean()
            viol = viol + (slack > 0).float().mean()
        loss = loss + log_gamma
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(V.parameters()) + [log_gamma], 1.0)
        opt.step(); sched.step()
        if step % log_every == 0 or step == steps - 1:
            hist.append({"step": step, "loss": float(loss),
                         "gamma": float(torch.exp(log_gamma)),
                         "viol_frac": float(viol) / max(1, n_states - 1)})
            if verbose:
                print(f"    step {step:5d}  loss {float(loss):.5f}  "
                      f"gamma {float(torch.exp(log_gamma)):.4f}  "
                      f"sampled-violation {float(viol)/max(1,n_states-1):.4f}")
    return V, hist, float(torch.exp(log_gamma).item())
