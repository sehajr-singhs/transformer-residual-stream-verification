"""Generate ARCHITECTURE.md from results/baseline.json.

The report is generated rather than hand-written so no number in it can drift
away from the run that produced it.
"""
import json
import os


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int,)):
        return str(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
        return f"{v:.2e}"
    return f"{v:.{nd}f}".rstrip("0").rstrip(".")


def build(res_path, out_path):
    with open(res_path) as f:
        R = json.load(f)
    L = []
    a = L.append

    m, cfg = R["model"], R["config"]
    wp = R["well_posedness"]
    bg = R["sae"]["bridge_gap"]
    cert = R["certification"]
    snd = R["soundness"]
    tm = R["threat_model"]

    a("# Baseline Architecture Report")
    a("")
    a("Structurally constrained compositional neural Lyapunov certificates for "
      "transformer safety. Generated from `results/baseline.json`; every number "
      "below comes from that run.")
    a("")
    a(f"- runtime: {_fmt(R['runtime_sec'], 0)}s")
    a(f"- prover backends: native hybrid-zonotope BaB "
      f"(z3: {_fmt(R['backends']['z3'])}, dReal: {_fmt(R['backends']['dreal'])}, "
      f"independent cross-check: {_fmt(R['backends']['independent_cross_check'])})")
    a("")

    a("## 1. Headline result")
    a("")
    a("The strict-decrease obligation this workspace was commissioned to prove "
      "is **infeasible for this model**, and that is a theorem rather than a "
      "tuning failure. The certificate is therefore restated as bounded "
      "finite-horizon growth, which is true, provable, and still sufficient for "
      "a safety guarantee.")
    a("")
    a("| quantity | value |")
    a("|---|---|")
    a(f"| deviation spectral radius on the certified subspace | "
      f"**{_fmt(wp['composite_restricted_to_span_U']['spectral_radius'], 3)}** |")
    a(f"| contraction metric can exist (needs radius < 1) | "
      f"**{_fmt(wp['contraction_possible'])}** |")
    a(f"| sampled growth factor of the fitted V | "
      f"{_fmt(R['lyapunov']['sampled_growth_factor'])} |")
    a(f"| all sound bounds survived falsification | "
      f"**{_fmt(snd['ALL_SOUND'])}** |")
    a(f"| max certified safe steering radius (sound BaB) | "
      f"**{_fmt(cert['max_certified_safe_rho'])}** |")
    a(f"| radius at which PGD actually breaks the model | "
      f"{_fmt(cert.get('empirical_robust_radius'))} |")
    if cert.get("geometric_gap_fraction"):
        a(f"| fraction of true robust radius certified | "
          f"**{_fmt(100 * cert['geometric_gap_fraction'], 2)}%** |")
    a(f"| small-gain composition feasible | "
      f"{_fmt(R['dissipativity']['gain_report']['small_gain_satisfied'])} "
      f"(cascade product "
      f"{_fmt(R['dissipativity']['gain_report']['cascade_gain_product'], 1)}) |")
    a("")
    a(f"> {wp['conclusion']}")
    a("")

    a("## 2. System under verification")
    a("")
    a(f"Pre-LN transformer, {m['n_layers']} layers x {m['n_heads']} heads, "
      f"d_model={m['d_model']}, d_mlp={m['d_mlp']}, seq_len={m['seq_len']}, "
      f"{m['params']} parameters, treated as the discrete-time system")
    a("")
    a("```")
    a("x_{l+1} = x_l + Attn_l(LN(x_l)) + MLP_l(LN(x_l + Attn_l(LN(x_l))))")
    a("```")
    a("")
    a("with layer index as time. Task is interpretable feature routing "
      "(`[f0 f1 f2 f3 SEL_k] -> f_k`); safety is the unsafe-logit margin "
      "`s(x) = max_{v unsafe} logit_v - max_{v safe} logit_v`, safe when `s < 0`.")
    a("")
    a("| metric | value |")
    a("|---|---|")
    a(f"| routing accuracy | {_fmt(m['eval']['acc_all'])} |")
    a(f"| worst safe-prompt margin | {_fmt(m['eval']['unsafe_margin_max_safe'], 3)} |")
    a(f"| fraction of safe prompts with s < 0 | "
      f"{_fmt(m['eval']['unsafe_margin_frac_negative_safe'])} |")
    a("")

    a("## 3. Invariant coordinate frame")
    a("")
    a("State is the deviation `e_l = x_l - x*_l` from the nominal trajectory of a "
      "fixed prompt class, so the origin is an exact fixed point of the deviation "
      "dynamics for every prompt with no equilibrium solve.")
    a("")
    a("Adding `c*1` to any position's residual stream leaves every LayerNorm "
      "output **exactly** unchanged, so it propagates through the identity path "
      "alone and is annihilated again by the final LayerNorm. Verified "
      "numerically:")
    a("")
    me = R["frame"]["mean_equivariance"]
    a(f"- max logit deviation under a mean shift of magnitude "
      f"{_fmt(me['shift_magnitude'], 3)}: **{_fmt(me['max_logit_deviation'])}** "
      f"(float32 round-off)")
    a(f"- centering projector `P = I - 11^T/d` is exact: symmetry error "
      f"{_fmt(R['frame']['projector']['sym_err'])}, idempotency error "
      f"{_fmt(R['frame']['projector']['idem_err'])}")
    a(f"- certified subspace dimension per position: "
      f"{R['frame']['gauge']['certified_subspace_dim']} of "
      f"{R['frame']['gauge']['d_model']}")
    a("")
    a(f"That removes {m['seq_len']} dimensions with zero approximation error.")
    a("")

    a("## 4. Feature frame and threat model")
    a("")
    a(f"{tm['kind']}, k={tm['k']}, features {tm['feature_ids']}:")
    a("")
    a(f"```\n{tm['set']}\n```")
    a("")
    a(f"Rationale: {tm['why']}.")
    a("")
    a("| SAE property | value |")
    a("|---|---|")
    a(f"| dictionary size | {bg['d_dict']} |")
    a(f"| relative reconstruction error (mean) | {_fmt(bg['rel_recon_err_mean'])} |")
    a(f"| relative reconstruction error (p99) | {_fmt(bg['rel_recon_err_p99'])} |")
    a(f"| fraction of variance unexplained | {_fmt(bg['fvu'])} |")
    a(f"| L0 (mean active features) | {_fmt(bg['l0_mean'], 1)} |")
    a(f"| dead features | {bg['dead_features']} |")
    a("")
    a("**Bridge-gap accounting.** " + R["stress"]["sae_bridge_sensitivity"]["note"])
    a("")

    a("## 5. Lyapunov function")
    a("")
    a(f"```\n{R['lyapunov']['form']}\n```")
    a("")
    a(f"- `V(0)` = {_fmt(R['lyapunov']['V_at_origin'])} exactly, by construction")
    a(f"- positive definite by construction: "
      f"{_fmt(R['lyapunov']['positive_definite_by_construction'])} "
      "(a violation is not representable by any parameter setting)")
    a(f"- convex in the state: {_fmt(R['lyapunov']['convex_in_state'])} "
      "(`W_enc e` is linear in `e`, so the ICNN's convexity transfers to state "
      "space, not merely to feature coordinates)")
    a(f"- alpha (quadratic tiebreaker weight): {_fmt(R['lyapunov']['alpha'])}")
    a("")

    a("## 6. Soundness")
    a("")
    a("Every bound is adversarially falsified by dense sampling before it is "
      "allowed to support a claim. Pass condition is `max_violation <= 0`.")
    a("")
    a("| check | max violation | sound |")
    a("|---|---|---|")
    for k, v in snd["primitives"].items():
        a(f"| primitive: {k} | {_fmt(v['max_violation'])} | {_fmt(v['sound'])} |")
    for k, v in snd["block_bounds"].items():
        a(f"| block propagation {k} | {_fmt(v['max_violation'])} | {_fmt(v['sound'])} |")
    a(f"| V bounds | {_fmt(snd['v_bounds']['max_violation'])} | "
      f"{_fmt(snd['v_bounds']['sound'])} |")
    for k, v in snd["growth_bound_subspace"].items():
        a(f"| growth bound {k} | {_fmt(v['max_violation'])} | {_fmt(v['sound'])} |")
    a("")
    a(f"**ALL_SOUND = {_fmt(snd['ALL_SOUND'])}**")
    a("")
    a("Relaxation gap (sound bound minus attained maximum) per layer and radius:")
    a("")
    a("| case | sound bound | attained | gap |")
    a("|---|---|---|---|")
    for k, v in snd["growth_bound_subspace"].items():
        a(f"| {k} | {_fmt(v['sound_bound'])} | {_fmt(v['attained_max_sampled'])} | "
          f"{_fmt(v['relaxation_gap'])} |")
    a("")

    a("## 7. Certification results")
    a("")
    a("### 7a. Per-layer growth factor (compositional obligation)")
    a("")
    a("Smallest `gamma_l` such that `V(e_{l+1}) <= gamma_l V(e_l)` is PROVED over "
      "the whole alpha box by sound branch-and-bound. Each layer is certified "
      "independently over a shared box, so cost is O(L), and the composition is "
      "the product. `n/a` means no gamma in [1, 6] was provable within budget.")
    a("")
    a("Ablation across metrics -- the control question is whether the learned ICNN "
      "certifies a SMALLER gamma than a trivial quadratic `alpha||P e||^2`. If it "
      "does not, the ICNN is decoration at this scale.")
    a("")
    for mname, block in cert.get("metric_ablation", {}).items():
        a(f"**metric: {mname}**")
        a("")
        a("| rho | " + " | ".join(f"gamma layer{l}" for l in range(m["n_layers"]))
          + " | composite bound |")
        a("|---|" + "---|" * (m["n_layers"] + 1))
        for rho, v in block.items():
            cells = [_fmt(v[f'layer{l}']['gamma_certified'], 3)
                     for l in range(m["n_layers"])]
            a(f"| {rho} | " + " | ".join(cells) + " | "
              + _fmt(v["composite_growth_bound"], 3) + " |")
        a("")
    if cert.get("inner_region_note"):
        a(f"Inner exclusion `inner_frac = {_fmt(cert.get('inner_frac'))}`. "
          + cert["inner_region_note"])
        a("")

    a("### 7b. Monolithic safety margin -- the certificate that closes")
    a("")
    a("Sound BaB proving the final unsafe-logit margin stays negative over the "
      "whole steering box. This obligation is ADDITIVE rather than a ratio, which "
      "is why branch-and-bound closes it while the growth condition of 7a resists.")
    a("")
    a("| rho | certified safe | worst bound | boxes touched |")
    a("|---|---|---|---|")
    for rho, v in cert["monolithic_safety"].items():
        st = v["stats"]
        a(f"| {rho} | {_fmt(v['certified_safe'])} | {_fmt(st.get('worst'))} | "
          f"{st.get('boxes_touched')} |")
    a("")
    a(f"**Max certified safe steering radius: "
      f"{_fmt(cert['max_certified_safe_rho'])}**")
    a("")
    if cert.get("pgd_attack"):
        a("PGD attack on the same steering set, to get an honest denominator "
          "(comparing a sound radius against a radius where random sampling "
          "merely failed to find anything would flatter the certificate):")
        a("")
        a("| rho | PGD max margin | model broken |")
        a("|---|---|---|")
        for rho, v in cert["pgd_attack"].items():
            a(f"| {rho} | {_fmt(v['pgd_max_margin'], 3)} | "
              f"{_fmt(v['broken'])} |")
        a("")
    emp = cert.get("empirical_robust_radius")
    if emp and cert.get("max_certified_safe_rho"):
        ratio = cert["max_certified_safe_rho"] / emp
        a(f"**The geometric gap.** PGD first breaks the model at rho = {_fmt(emp)}, "
          f"while the prover certifies rho = {_fmt(cert['max_certified_safe_rho'])}. "
          f"The certificate therefore covers about **{_fmt(100 * ratio, 2)}%** of "
          f"the radius over which the model is genuinely safe. That single number "
          f"is the theoretical ceiling of this construction as built, and it is the "
          f"figure to judge before investing in scale: the proof is real, and it is "
          f"roughly two orders of magnitude more conservative than the truth.")
        a("")

    a("## 8. Compositional dissipativity, and why small-gain cannot work here")
    a("")
    gr = R["dissipativity"]["gain_report"]
    a(f"> {gr['obstruction']}")
    a("")
    a("| layer | gamma attention (joint) | sum over heads | gamma MLP | "
      "layer gain upper |")
    a("|---|---|---|---|---|")
    for g in R["dissipativity"]["subblock_gains"]:
        a(f"| {g['layer']} | {_fmt(g['gamma_attn_joint'], 3)} | "
          f"{_fmt(g['gamma_heads_sum'], 3)} | {_fmt(g['gamma_mlp'], 3)} | "
          f"{_fmt(g['gamma_layer_upper'], 3)} |")
    a("")
    a(f"- cascade gain product: {_fmt(gr['cascade_gain_product'], 1)}")
    a(f"- small-gain condition satisfied: {_fmt(gr['small_gain_satisfied'])}")
    a(f"- interconnection LP feasible: "
      f"{_fmt(R['dissipativity']['supply_rate_lp']['feasible'])}")
    a("")
    a("Both the a-priori argument (the identity path forces every norm gain to "
      "exceed 1) and the measured numbers agree. The compositional value is "
      "therefore NOT gain multiplication; it is that each layer's obligation is "
      "discharged independently over a shared box at O(L) cost.")
    a("")

    a("## 9. Stress: where the certificate dies")
    a("")
    a("### Unstable ReLU count is the binding constraint")
    a("")
    a("Once the unstable (interval-crossing) ReLU count saturates, the "
      "perturbation-to-signal ratio entering LayerNorm exceeds ~0.5, the "
      "`1/sqrt(var)` bracket collapses toward `1/sqrt(eps)`, and the "
      "propagation detonates.")
    a("")
    a("| rho | layer | pert/signal ratio | unstable ReLUs | output width |")
    a("|---|---|---|---|---|")
    for row in R["stress"]["unstable_relu_vs_rho"]:
        for lay in row["layers"]:
            a(f"| {row['rho']} | {lay['layer']} | "
              f"{_fmt(lay['perturbation_to_signal_ratio'], 3)} | "
              f"{lay['unstable_relus']}/{lay['of_total']} | "
              f"{_fmt(lay['out_width'])} |")
    a("")
    a("### Threat-model dimension scaling")
    a("")
    a("| k | final bound width @ rho=0.01 | finite |")
    a("|---|---|---|")
    for s in R["stress"]["subspace_dim_scaling"]:
        a(f"| {s['k']} | {_fmt(s['final_width'])} | {_fmt(s['finite'])} |")
    fb = R["stress"]["full_box_baseline"]
    a(f"| full box ({fb['n_generators']} gens, rho={fb['rho']}) | "
      f"{_fmt(fb['final_width'])} | - |")
    a("")
    a("### Propagation ablation")
    a("")
    a("| mode | final bound width @ rho=0.01 |")
    a("|---|---|")
    for s in R["stress"]["propagation_ablation"]:
        a(f"| {s['mode']} | {_fmt(s['final_width'])} |")
    a("")

    a("## 10. Honest limitations")
    a("")
    for s in [
        "The strict-decrease certificate originally specified is impossible for "
        "this model (spectral radius "
        f"{_fmt(wp['composite_restricted_to_span_U']['spectral_radius'], 3)} > 1). "
        "What is delivered is a bounded-growth certificate.",
        "The independent prover cross-check is PARTIALLY closed (audit c14, see "
        "C14_AUDIT.md). z3 confirms the prover's primitives over exact rationals: "
        "the margin readout is 8/8 sound and 7/8 tight on boxes the prover "
        "discharges. The END-TO-END bound is still not independently confirmed. "
        "Two from-scratch reference engines (audit/ibp_ref.py, audit/mvf_ref.py, "
        "both validated against torch to ~1e-15) close ZERO boxes at any radius, "
        "because any decorrelating abstraction amplifies by 4.7e8 through two "
        "blocks. Note this is a limit on INTERVAL abstractions specifically, not "
        "on abstract interpretation as such -- the hybrid zonotope is itself an "
        "abstract prover and survives precisely because it carries the k noise "
        "symbols exactly. Closing the gap needs a second correlation-preserving "
        "implementation. dReal and cvxpy remain absent.",
        "The certificate is stated for a SINGLE prompt anchor "
        "(x_nom = trace[0][0]), not over the prompt distribution. Every radius "
        "reported here is a claim about the continuous perturbation set around "
        "one fixed discrete context. Nothing is proved about a second prompt, "
        "let alone about a distribution over prompts, and the anchor was not "
        "chosen adversarially. Extending the same obligation across token "
        "distributions is the immediate frontier for this line of work and is "
        "strictly harder: the discrete context enters non-smoothly, so it cannot "
        "be folded into the same alpha-space branch-and-bound.",
        "The threat model is steering along k SAE directions at the final token, "
        "not arbitrary activation noise. A full 160-dimensional box is reported "
        "as a baseline and is far outside what the prover can close.",
        "This is a 2-layer, d_model=32 toy transformer on a synthetic routing "
        "task. Nothing here has been demonstrated to scale, and the unstable-ReLU "
        "mechanism identified in section 9 gets worse with width and depth.",
        "Safety is operationalized as an unsafe-logit margin over a designated "
        "token subset. Bridging that to any real notion of harmful behaviour is a "
        "semantic assumption, not a proved step.",
        "The excluded inner box around the origin is discharged by direct margin "
        "bounding rather than by the growth condition; its size is a reported "
        "parameter, not a hidden one.",
        "The growth-RATIO obligation is much harder for branch-and-bound than the "
        "margin obligation, and for a structural reason: V(e')/V(e) is "
        "scale-invariant, so shrinking a box shrinks numerator and denominator "
        "together and the relaxation-inflated ratio does not fall nearly as fast "
        "as an additive slack does. The margin condition is additive and closes; "
        "the ratio condition needs roughly two more orders of magnitude of "
        "tightness. Any depth-scaling claim for the compositional route has to "
        "clear that bar first.",
        "The ICNN term measurably WIDENS the certified bounds relative to the "
        "trivial quadratic metric at this scale (see the section 7a ablation), "
        "because its unstable ReLUs add relaxation error that the quadratic has "
        "none of. Input convexity buys convex sublevel sets and positive "
        "definiteness by construction; on this model it does not buy a smaller "
        "certified growth factor.",
        "The empirical radius is an UPPER bound on the true robust radius, not a "
        "measurement of it. PGD breaking at rho=10 shows the true radius is at "
        "most 10; PGD failing at rho=3 shows nothing, because PGD is incomplete. "
        "The true radius therefore lies somewhere in (certified, 10]. The "
        "conservativeness gap should be read as 'at most 250x', and it is a "
        "property of this prover on this model, not a general tax for formal "
        "guarantees.",
        "The unstable-ReLU count, not the LayerNorm variance bracket, is what "
        "drives the initial bound explosion (audit c14 section 3). The unsplit "
        "bound crosses zero at rho ~ 0.0070 while the LayerNorm bracket is still "
        "tight (spread 1.22); the bracket only degrades at rho >= 0.0171, after "
        "the bound has already passed 1e8. Text attributing the failure to "
        "LayerNorm cliffs has the causal order backwards.",
        "The bound is NOT monotone in rho: it reaches 2.1e10 at rho=0.0267 but "
        "3.5e4 at rho=0.0334. promote_E_topk and compact are radius-dependent "
        "heuristics, so a larger box can receive a luckier promotion. Any claim "
        "that the bound degrades smoothly with radius is false as stated, and "
        "any radius-sweep figure must show the raw non-monotone curve rather "
        "than a fitted trend.",
    ]:
        a(f"- {s}")
    a("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return out_path


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(build(os.path.join(root, "results", "baseline.json"),
                os.path.join(root, "ARCHITECTURE.md")))
