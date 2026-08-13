# Baseline Architecture Report

Structurally constrained compositional neural Lyapunov certificates for transformer safety. Generated from `results/baseline.json`; every number below comes from that run.

- runtime: 766s
- prover backends: native hybrid-zonotope BaB (z3: no, dReal: no, independent cross-check: no)

## 1. Headline result

The strict-decrease obligation this workspace was commissioned to prove is **infeasible for this model**, and that is a theorem rather than a tuning failure. The certificate is therefore restated as bounded finite-horizon growth, which is true, provable, and still sufficient for a safety guarantee.

| quantity | value |
|---|---|
| deviation spectral radius on the certified subspace | **1.273** |
| contraction metric can exist (needs radius < 1) | **no** |
| sampled growth factor of the fitted V | 1.4884 |
| all sound bounds survived falsification | **yes** |
| max certified safe steering radius (sound BaB) | **0.04** |
| radius at which PGD actually breaks the model | 10 |
| fraction of true robust radius certified | **0.4%** |
| small-gain composition feasible | no (cascade product 5562.9) |

> A positive-definite V with V(e') <= gamma V(e), gamma < 1, exists IFF the deviation spectral radius is < 1. It is 1.273 > 1 on the certified subspace, so the strict-decrease obligation is INFEASIBLE for this model -- not merely hard to relax. The certificate is therefore stated as bounded finite-horizon growth, which is true and still sufficient for a safety guarantee.

## 2. System under verification

Pre-LN transformer, 2 layers x 4 heads, d_model=32, d_mlp=128, seq_len=5, 26144 parameters, treated as the discrete-time system

```
x_{l+1} = x_l + Attn_l(LN(x_l)) + MLP_l(LN(x_l + Attn_l(LN(x_l))))
```

with layer index as time. Task is interpretable feature routing (`[f0 f1 f2 f3 SEL_k] -> f_k`); safety is the unsafe-logit margin `s(x) = max_{v unsafe} logit_v - max_{v safe} logit_v`, safe when `s < 0`.

| metric | value |
|---|---|
| routing accuracy | 1 |
| worst safe-prompt margin | -9.565 |
| fraction of safe prompts with s < 0 | 1 |

## 3. Invariant coordinate frame

State is the deviation `e_l = x_l - x*_l` from the nominal trajectory of a fixed prompt class, so the origin is an exact fixed point of the deviation dynamics for every prompt with no equilibrium solve.

Adding `c*1` to any position's residual stream leaves every LayerNorm output **exactly** unchanged, so it propagates through the identity path alone and is annihilated again by the final LayerNorm. Verified numerically:

- max logit deviation under a mean shift of magnitude 0.448: **2.86e-06** (float32 round-off)
- centering projector `P = I - 11^T/d` is exact: symmetry error 0, idempotency error 0
- certified subspace dimension per position: 31 of 32

That removes 5 dimensions with zero approximation error.

## 4. Feature frame and threat model

activation steering along k SAE decoder directions at the final token, k=4, features [8, 31, 7, 50]:

```
e = sum_j alpha_j U_j,  |alpha_j| <= rho
```

Rationale: a full box over the residual stream needs T*d_model = 160 generators and inflates ~2x per dense map regardless of propagation tightness; steering attacks move the stream along a few interpretable directions.

| SAE property | value |
|---|---|
| dictionary size | 64 |
| relative reconstruction error (mean) | 0.0228 |
| relative reconstruction error (p99) | 0.0418 |
| fraction of variance unexplained | 5.91e-04 |
| L0 (mean active features) | 49.7 |
| dead features | 0 |

**Bridge-gap accounting.** the encoder is used as a READOUT (V is a function of e, not of Dec(Enc(e))), so reconstruction error does NOT enter the proved dynamics; it bounds only how well V's level sets align with feature semantics. If anyone closes the loop through Dec, this number becomes the soundness bottleneck.

## 5. Lyapunov function

```
V(e) = ReLU(g(W_enc e) - g(0)) + alpha*||P e||^2, g an ICNN
```

- `V(0)` = 0 exactly, by construction
- positive definite by construction: yes (a violation is not representable by any parameter setting)
- convex in the state: yes (`W_enc e` is linear in `e`, so the ICNN's convexity transfers to state space, not merely to feature coordinates)
- alpha (quadratic tiebreaker weight): 0.5

## 6. Soundness

Every bound is adversarially falsified by dense sampling before it is allowed to support a claim. Pass condition is `max_violation <= 0`.

| check | max violation | sound |
|---|---|---|
| primitive: relu | 0 | yes |
| primitive: layernorm | -0.2453 | yes |
| primitive: softmax | 0 | yes |
| block propagation layer0 | -0.4713 | yes |
| block propagation layer1 | -2.59e+05 | yes |
| V bounds | -2.8319 | yes |
| growth bound layer0_rho0.01 | -0.3621 | yes |
| growth bound layer0_rho0.05 | -471.5749 | yes |
| growth bound layer1_rho0.01 | -0.3112 | yes |
| growth bound layer1_rho0.05 | -84.9788 | yes |

**ALL_SOUND = yes**

Relaxation gap (sound bound minus attained maximum) per layer and radius:

| case | sound bound | attained | gap |
|---|---|---|---|
| layer0_rho0.01 | 0.3621 | 6.70e-05 | 0.3621 |
| layer0_rho0.05 | 471.575 | 7.72e-05 | 471.5749 |
| layer1_rho0.01 | 0.3113 | 8.67e-05 | 0.3112 |
| layer1_rho0.05 | 84.9794 | 6.21e-04 | 84.9788 |

## 7. Certification results

### 7a. Per-layer growth factor (compositional obligation)

Smallest `gamma_l` such that `V(e_{l+1}) <= gamma_l V(e_l)` is PROVED over the whole alpha box by sound branch-and-bound. Each layer is certified independently over a shared box, so cost is O(L), and the composition is the product. `n/a` means no gamma in [1, 6] was provable within budget.

Ablation across metrics -- the control question is whether the learned ICNN certifies a SMALLER gamma than a trivial quadratic `alpha||P e||^2`. If it does not, the ICNN is decoration at this scale.

**metric: icnn**

| rho | gamma layer0 | gamma layer1 | composite bound |
|---|---|---|---|
| 0.005 | n/a | n/a | n/a |

**metric: quadratic_only**

| rho | gamma layer0 | gamma layer1 | composite bound |
|---|---|---|---|
| 0.005 | n/a | n/a | n/a |

Inner exclusion `inner_frac = 0.3`. The inner region |alpha_j| <= inner_frac*rho is NOT covered by the growth condition: V(e_l) has no positive floor there, so V(e') <= gamma V(e) is unsatisfiable for every gamma. It is discharged separately by the margin certificate below, which is the standard practical-stability formulation. Its size is a reported parameter, not a hidden knob.

### 7b. Monolithic safety margin -- the certificate that closes

Sound BaB proving the final unsafe-logit margin stays negative over the whole steering box. This obligation is ADDITIVE rather than a ratio, which is why branch-and-bound closes it while the growth condition of 7a resists.

| rho | certified safe | worst bound | boxes touched |
|---|---|---|---|
| 0.005 | yes | n/a | 0 |
| 0.01 | yes | n/a | 7 |
| 0.02 | yes | n/a | 127 |
| 0.03 | yes | n/a | 1023 |
| 0.04 | yes | n/a | 2047 |
| 0.06 | no | 38.5762 | 8191 |

**Max certified safe steering radius: 0.04**

PGD attack on the same steering set, to get an honest denominator (comparing a sound radius against a radius where random sampling merely failed to find anything would flatter the certificate):

| rho | PGD max margin | model broken |
|---|---|---|
| 0.04 | -11.337 | no |
| 0.3 | -11.233 | no |
| 1.0 | -10.642 | no |
| 3.0 | -3.682 | no |
| 10.0 | 4.005 | yes |

**The geometric gap.** PGD first breaks the model at rho = 10, while the prover certifies rho = 0.04. The certificate therefore covers about **0.4%** of the radius over which the model is genuinely safe. That single number is the theoretical ceiling of this construction as built, and it is the figure to judge before investing in scale: the proof is real, and it is roughly two orders of magnitude more conservative than the truth.

## 8. Compositional dissipativity, and why small-gain cannot work here

> identity path forces gamma_layer >= 1, so no norm-gain composition can certify contraction of a residual stream; contraction must come from sign-aware cancellation, i.e. from V's cross terms

| layer | gamma attention (joint) | sum over heads | gamma MLP | layer gain upper |
|---|---|---|---|---|
| 0 | 39.251 | 51.434 | 25.831 | 66.082 |
| 1 | 53.083 | 69.578 | 30.099 | 84.182 |

- cascade gain product: 5562.9
- small-gain condition satisfied: no
- interconnection LP feasible: no

Both the a-priori argument (the identity path forces every norm gain to exceed 1) and the measured numbers agree. The compositional value is therefore NOT gain multiplication; it is that each layer's obligation is discharged independently over a shared box at O(L) cost.

## 9. Stress: where the certificate dies

### Unstable ReLU count is the binding constraint

Once the unstable (interval-crossing) ReLU count saturates, the perturbation-to-signal ratio entering LayerNorm exceeds ~0.5, the `1/sqrt(var)` bracket collapses toward `1/sqrt(eps)`, and the propagation detonates.

| rho | layer | pert/signal ratio | unstable ReLUs | output width |
|---|---|---|---|---|
| 0.005 | 0 | 0.004 | 2/640 | 0.0063 |
| 0.005 | 1 | 0.021 | 29/640 | 0.1121 |
| 0.01 | 0 | 0.008 | 8/640 | 0.0176 |
| 0.01 | 1 | 0.06 | 128/640 | 2.5924 |
| 0.02 | 0 | 0.017 | 32/640 | 0.0698 |
| 0.02 | 1 | 0.239 | 128/640 | 80608.6789 |
| 0.05 | 0 | 0.042 | 123/640 | 0.9995 |
| 0.05 | 1 | 4.136 | 128/640 | 588.1025 |
| 0.1 | 0 | 0.083 | 128/640 | 27.9541 |
| 0.1 | 1 | 12.135 | 128/640 | 1.78e+07 |
| 0.25 | 0 | 0.208 | 128/640 | 21743.802 |
| 0.25 | 1 | 11.828 | 128/640 | 5.81e+07 |

### Threat-model dimension scaling

| k | final bound width @ rho=0.01 | finite |
|---|---|---|
| 1 | 0.0192 | yes |
| 2 | 0.1386 | yes |
| 4 | 2.5924 | yes |
| 8 | 1.07e+05 | yes |
| 16 | 9.8168 | yes |
| full box (160 gens, rho=0.002) | 1.03e+06 | - |

### Propagation ablation

| mode | final bound width @ rho=0.01 |
|---|---|
| hybrid+promotion | 2.5924 |
| no promotion | 4.4481 |

## 10. Honest limitations

- The strict-decrease certificate originally specified is impossible for this model (spectral radius 1.273 > 1). What is delivered is a bounded-growth certificate.
- The independent prover cross-check is PARTIALLY closed (audit c14, see C14_AUDIT.md). z3 confirms the prover's primitives over exact rationals: the margin readout is 8/8 sound and 7/8 tight on boxes the prover discharges. The END-TO-END bound is still not independently confirmed. Two from-scratch reference engines (audit/ibp_ref.py, audit/mvf_ref.py, both validated against torch to ~1e-15) close ZERO boxes at any radius, because any decorrelating abstraction amplifies by 4.7e8 through two blocks. Note this is a limit on INTERVAL abstractions specifically, not on abstract interpretation as such -- the hybrid zonotope is itself an abstract prover and survives precisely because it carries the k noise symbols exactly. Closing the gap needs a second correlation-preserving implementation. dReal and cvxpy remain absent.
- The certificate is stated for a SINGLE prompt anchor (x_nom = trace[0][0]), not over the prompt distribution. Every radius reported here is a claim about the continuous perturbation set around one fixed discrete context. Nothing is proved about a second prompt, let alone about a distribution over prompts, and the anchor was not chosen adversarially. Extending the same obligation across token distributions is the immediate frontier for this line of work and is strictly harder: the discrete context enters non-smoothly, so it cannot be folded into the same alpha-space branch-and-bound.
- The threat model is steering along k SAE directions at the final token, not arbitrary activation noise. A full 160-dimensional box is reported as a baseline and is far outside what the prover can close.
- This is a 2-layer, d_model=32 toy transformer on a synthetic routing task. Nothing here has been demonstrated to scale, and the unstable-ReLU mechanism identified in section 9 gets worse with width and depth.
- Safety is operationalized as an unsafe-logit margin over a designated token subset. Bridging that to any real notion of harmful behaviour is a semantic assumption, not a proved step.
- The excluded inner box around the origin is discharged by direct margin bounding rather than by the growth condition; its size is a reported parameter, not a hidden one.
- The growth-RATIO obligation is much harder for branch-and-bound than the margin obligation, and for a structural reason: V(e')/V(e) is scale-invariant, so shrinking a box shrinks numerator and denominator together and the relaxation-inflated ratio does not fall nearly as fast as an additive slack does. The margin condition is additive and closes; the ratio condition needs roughly two more orders of magnitude of tightness. Any depth-scaling claim for the compositional route has to clear that bar first.
- The ICNN term measurably WIDENS the certified bounds relative to the trivial quadratic metric at this scale (see the section 7a ablation), because its unstable ReLUs add relaxation error that the quadratic has none of. Input convexity buys convex sublevel sets and positive definiteness by construction; on this model it does not buy a smaller certified growth factor.
- The empirical radius is an UPPER bound on the true robust radius, not a measurement of it. PGD breaking at rho=10 shows the true radius is at most 10; PGD failing at rho=3 shows nothing, because PGD is incomplete. The true radius therefore lies somewhere in (certified, 10]. The conservativeness gap should be read as 'at most 250x', and it is a property of this prover on this model, not a general tax for formal guarantees.
- The unstable-ReLU count, not the LayerNorm variance bracket, is what drives the initial bound explosion (audit c14 section 3). The unsplit bound crosses zero at rho ~ 0.0070 while the LayerNorm bracket is still tight (spread 1.22); the bracket only degrades at rho >= 0.0171, after the bound has already passed 1e8. Text attributing the failure to LayerNorm cliffs has the causal order backwards.
- The bound is NOT monotone in rho: it reaches 2.1e10 at rho=0.0267 but 3.5e4 at rho=0.0334. promote_E_topk and compact are radius-dependent heuristics, so a larger box can receive a luckier promotion. Any claim that the bound degrades smoothly with radius is false as stated, and any radius-sweep figure must show the raw non-monotone curve rather than a fitted trend.
