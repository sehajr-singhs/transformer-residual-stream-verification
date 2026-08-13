"""c14 -- independent audit of the monolithic margin certificate.

The certificate under audit: for a fixed prompt anchor, the hybrid-zonotope BaB
decomposes the alpha box [-rho, rho]^4 into sub-boxes and discharges each one by
proving a sound upper bound s_hi < 0 on the unsafe-logit margin. The union of
discharged boxes is the whole box, so the certificate is exactly the conjunction
of those per-box claims.

This script attacks that conjunction three ways, none of which route through
src/bounds.py's zonotope mechanics:

  A. FALSIFICATION. Inside every discharged box, evaluate the TRUE margin with
     torch at corners, faces, interior samples, and PGD ascent witnesses. Any
     sample whose true margin exceeds that box's claimed bound is an unsoundness
     bug. This is the test that can actually kill the certificate.

  B. INDEPENDENT SOUND BOUND. Re-bound every discharged box with audit/ibp_ref,
     transcribed from the torch forward. Where IBP also closes, the box has a
     second, structurally different proof. Where it does not, that is reported
     as unconfirmed rather than quietly dropped.

  C. COVERAGE. Verify the discharged boxes actually tile [-rho, rho]^4 by volume
     and by containment of random points, so the conjunction is not vacuous.

Run:  python audit/c14_crosscheck.py [--rhos 0.005,0.01,0.02] [--samples 4096]
"""
import sys, os, json, time, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, torch
from src import toy_transformer as tt, task, sae as sae_mod, bounds as bd, verifier as vf
import ibp_ref as ref

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.path.join(ROOT, "checkpoints")
K_DIRS = 4
SEED = 0

ap = argparse.ArgumentParser()
ap.add_argument("--rhos", default="0.005,0.01,0.02,0.03,0.04")
ap.add_argument("--samples", type=int, default=4096)
ap.add_argument("--pgd-steps", type=int, default=40)
ap.add_argument("--max-boxes-audit", type=int, default=4096)
ap.add_argument("--out", default=os.path.join(ROOT, "results", "c14_audit.json"))
args = ap.parse_args()
RHOS = [float(r) for r in args.rhos.split(",")]

t_start = time.time()


def stamp(msg):
    print(f"[{time.time() - t_start:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------- 1. rebuild state
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
pick = np.random.default_rng(11).choice(Wdec.shape[0], size=K_DIRS, replace=False)
U = np.zeros((K_DIRS, w["seq_len"], w["d_model"]))
U[np.arange(K_DIRS), w["seq_len"] - 1, :] = Wdec[pick]

prompts, _ = task.enumerate_prompts(limit=256, rng=np.random.default_rng(SEED),
                                    safe_only=True)
with torch.no_grad():
    trace = model64.residual_trace(torch.from_numpy(prompts))
x_nom = [t[0].numpy().astype(np.float64) for t in trace]
x_nom_0 = x_nom[0]
iu, isf = task.margin_readout()

stamp(f"threat model rebuilt: feature_ids={pick.tolist()}  k={K_DIRS}")
BASE = json.load(open(os.path.join(ROOT, "results", "baseline.json")))
assert pick.tolist() == BASE["threat_model"]["feature_ids"], \
    "threat model does not reproduce baseline.json -- audit would be of a different object"
stamp("feature_ids match baseline.json exactly")


# ------------------------------------------------------- 2. ground-truth margin
U_t = torch.from_numpy(U)
xn_t = torch.from_numpy(x_nom_0)


def true_margin(alpha):
    """Exact margin from the float64 torch model. alpha: (B, k) -> (B,)."""
    a = torch.as_tensor(alpha, dtype=torch.float64)
    x = xn_t[None] + torch.einsum("bk,ktd->btd", a, U_t)
    with torch.no_grad():
        for blk in model64.blocks:
            x = blk(x)
        L = model64.logits_from_stream(x)[:, -1]
    return task.unsafe_margin(L.numpy())


def pgd_in_box(a_lo, a_hi, steps, restarts=3, seed=0):
    """Projected gradient ascent on the true margin, confined to one box."""
    rng = np.random.default_rng(seed)
    lo_t = torch.as_tensor(a_lo, dtype=torch.float64)
    hi_t = torch.as_tensor(a_hi, dtype=torch.float64)
    best = np.full(a_lo.shape[0], -np.inf)
    best_a = np.zeros_like(a_lo)
    span = hi_t - lo_t
    for r in range(restarts):
        init = lo_t + torch.as_tensor(rng.random(a_lo.shape)) * span
        a = init.clone().requires_grad_(True)
        step = span / max(steps // 4, 1)
        for _ in range(steps):
            x = xn_t[None] + torch.einsum("bk,ktd->btd", a, U_t)
            for blk in model64.blocks:
                x = blk(x)
            L = model64.logits_from_stream(x)[:, -1]
            um = L[:, torch.from_numpy(iu)].max(dim=-1).values
            sm = L[:, torch.from_numpy(isf)].max(dim=-1).values
            obj = (um - sm).sum()
            g, = torch.autograd.grad(obj, a)
            with torch.no_grad():
                a = torch.clamp(a + step * torch.sign(g), lo_t, hi_t)
            a.requires_grad_(True)
        with torch.no_grad():
            m = true_margin(a.detach().numpy())
        upd = m > best
        best = np.where(upd, m, best)
        best_a[upd] = a.detach().numpy()[upd]
    return best, best_a


def sample_box(a_lo, a_hi, n, rng):
    """Corners (up to 16 for k=4), plus uniform interior points."""
    k = a_lo.shape[0]
    corners = np.array(np.meshgrid(*[[0.0, 1.0]] * k, indexing="ij")).reshape(k, -1).T
    pts = [a_lo + corners * (a_hi - a_lo)]
    if n > len(corners):
        u = rng.random((n - len(corners), k))
        pts.append(a_lo + u * (a_hi - a_lo))
    return np.concatenate(pts, axis=0)


# ----------------------------------------------- 3. replay BaB, collect discharge
def hz_margin(a_lo, a_hi, chunk=128):
    """The prover's own bound, batched. Mirrors certify_margin_radius's ev()."""
    out = []
    for i in range(0, a_lo.shape[0], chunk):
        z = vf.alpha_to_zono(a_lo[i:i + chunk], a_hi[i:i + chunk], U) + x_nom_0[None]
        zL = vf._blocks_from(z, w, 0, w["n_layers"])
        s = bd.unsafe_margin_upper(bd.readout_logits(zL, w), iu, isf)
        out.append(s)
    return np.concatenate(out)


def replay(rho, max_boxes=4096, max_iters=20, min_width=1e-5):
    """Re-run the prover's BaB and record every box it discharges."""
    k = K_DIRS
    a_lo = np.full((1, k), -rho)
    a_hi = np.full((1, k), rho)
    disc_lo, disc_hi, disc_ub = [], [], []
    for it in range(max_iters):
        ub = hz_margin(a_lo, a_hi)
        ub = np.where(np.isfinite(ub), ub, np.inf)
        keep = ub >= 0.0
        d = ~keep
        if d.any():
            disc_lo.append(a_lo[d]); disc_hi.append(a_hi[d]); disc_ub.append(ub[d])
        a_lo, a_hi = a_lo[keep], a_hi[keep]
        if a_lo.shape[0] == 0:
            return True, (np.concatenate(disc_lo), np.concatenate(disc_hi),
                          np.concatenate(disc_ub)), it + 1
        wd = a_hi - a_lo
        if wd.max(axis=1).min() < min_width or a_lo.shape[0] * 2 > max_boxes:
            break
        dim = wd.argmax(axis=1); r = np.arange(a_lo.shape[0])
        mid = 0.5 * (a_lo[r, dim] + a_hi[r, dim])
        l1, h1 = a_lo.copy(), a_hi.copy(); h1[r, dim] = mid
        l2, h2 = a_lo.copy(), a_hi.copy(); l2[r, dim] = mid
        a_lo = np.concatenate([l1, l2]); a_hi = np.concatenate([h1, h2])
    packed = ((np.concatenate(disc_lo), np.concatenate(disc_hi),
               np.concatenate(disc_ub)) if disc_lo else
              (np.zeros((0, k)), np.zeros((0, k)), np.zeros(0)))
    return False, packed, max_iters


# ------------------------------------------------------------------- 4. the audit
report = {"config": {"rhos": RHOS, "samples_per_box": args.samples,
                     "pgd_steps": args.pgd_steps, "k": K_DIRS,
                     "feature_ids": pick.tolist(), "seed": SEED},
          "reference_engine": {"validated_against_torch_at_rho0": True},
          "per_rho": {}}

for rho in RHOS:
    stamp(f"rho={rho}: replaying prover BaB")
    proved, (dlo, dhi, dub), iters = replay(rho)
    n_boxes = dlo.shape[0]
    stamp(f"  prover proved={proved}  discharged boxes={n_boxes}  iters={iters}")

    base_claim = BASE["certification"]["monolithic_safety"].get(str(rho), {})
    agrees = base_claim.get("certified_safe", None) == proved

    # --- C. coverage: do the discharged boxes tile the original box?
    vol_total = (2 * rho) ** K_DIRS
    vol_disc = float(np.prod(dhi - dlo, axis=1).sum()) if n_boxes else 0.0
    rng = np.random.default_rng(12345)
    probe = rng.uniform(-rho, rho, size=(20000, K_DIRS))
    inside = np.zeros(len(probe), bool)
    for i in range(n_boxes):
        inside |= np.all((probe >= dlo[i] - 1e-15) & (probe <= dhi[i] + 1e-15), axis=1)
    cover_frac = float(inside.mean())

    # --- A + B per box
    audit_n = min(n_boxes, args.max_boxes_audit)
    sel = np.arange(n_boxes) if n_boxes <= audit_n else \
        np.random.default_rng(7).choice(n_boxes, audit_n, replace=False)

    worst_soundness = -np.inf     # max over samples of (true - claimed bound)
    worst_true = -np.inf          # max true margin found anywhere
    ibp_bounds = np.full(len(sel), np.nan)
    ibp_closed = 0
    srng = np.random.default_rng(99)

    for n, bi in enumerate(sel):
        blo, bhi_, claim = dlo[bi], dhi[bi], dub[bi]
        pts = sample_box(blo, bhi_, args.samples, srng)
        tm = true_margin(pts)
        worst_soundness = max(worst_soundness, float((tm - claim).max()))
        worst_true = max(worst_true, float(tm.max()))
        ib = ref.certify_margin(blo, bhi_, U, x_nom_0, w, iu, isf)
        ibp_bounds[n] = ib
        if ib < 0:
            ibp_closed += 1
        if (n + 1) % 256 == 0:
            stamp(f"    audited {n + 1}/{len(sel)} boxes")

    # PGD sweep over the discharged boxes (batched)
    stamp(f"  PGD ascent inside {len(sel)} discharged boxes")
    pgd_worst = -np.inf
    pgd_viol = -np.inf
    B = 256
    for i in range(0, len(sel), B):
        idx = sel[i:i + B]
        pm, _ = pgd_in_box(dlo[idx], dhi[idx], args.pgd_steps, seed=int(i))
        pgd_worst = max(pgd_worst, float(pm.max()))
        pgd_viol = max(pgd_viol, float((pm - dub[idx]).max()))
    worst_soundness = max(worst_soundness, pgd_viol)
    worst_true = max(worst_true, pgd_worst)

    row = {
        "prover_proved": bool(proved),
        "baseline_claimed": base_claim.get("certified_safe", None),
        "replay_agrees_with_baseline": bool(agrees),
        "discharged_boxes": int(n_boxes),
        "boxes_audited": int(len(sel)),
        "coverage": {
            "volume_fraction": vol_disc / vol_total if vol_total else None,
            "random_point_coverage": cover_frac,
            "tiles_the_box": bool(abs(vol_disc / vol_total - 1.0) < 1e-9
                                  and cover_frac == 1.0) if n_boxes else False,
        },
        "falsification": {
            "samples_per_box": int(args.samples),
            "max_true_margin_found": worst_true,
            "max_true_minus_claimed_bound": worst_soundness,
            "unsound": bool(worst_soundness > 0),
            "true_margin_stays_negative": bool(worst_true < 0),
        },
        "independent_ibp": {
            "boxes_closed": int(ibp_closed),
            "boxes_attempted": int(len(sel)),
            "fraction_closed": float(ibp_closed / len(sel)) if len(sel) else None,
            "median_bound": float(np.nanmedian(ibp_bounds)) if len(sel) else None,
            "min_bound": float(np.nanmin(ibp_bounds)) if len(sel) else None,
            "max_bound": float(np.nanmax(ibp_bounds)) if len(sel) else None,
        },
    }
    report["per_rho"][str(rho)] = row
    stamp(f"  proved={proved} agrees={agrees} | coverage {cover_frac:.4f} | "
          f"worst true {worst_true:+.4f} | soundness margin {worst_soundness:+.3e} | "
          f"IBP closed {ibp_closed}/{len(sel)}")

report["runtime_sec"] = time.time() - t_start
any_unsound = any(r["falsification"]["unsound"] for r in report["per_rho"].values())
report["VERDICT"] = {
    "no_unsoundness_found": not any_unsound,
    "note": ("Falsification failing to find a violation is evidence, not proof. "
             "The IBP column is the only genuinely independent SOUND bound here; "
             "where it does not close, the box rests on the native engine alone."),
}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(report, open(args.out, "w"), indent=2)
stamp(f"wrote {args.out}")
print(json.dumps(report["VERDICT"], indent=2))
